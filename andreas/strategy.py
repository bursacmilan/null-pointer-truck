#!/usr/bin/env python3
"""
Hybrid RampRush strategy — Gemini via Vertex AI + Milan-derived supplier matching.

Supplier resolution pipeline:
  Tier 0: exact normalized match (dict lookup)
  Tier 1: min(token_sort_ratio, WRatio) >= 72  [Milan's conservative scoring]
  Tier 2: jaro_winkler on Metaphone codes >= 70 [phonetic / audio variants]
  Tier 3: token overlap >= 2 tokens             [partial audio names]
  Tier 4: best fuzzy regardless of threshold    [last resort + warning]
"""

import json
import logging
import os
import re
import sys

import jellyfish
import requests
from extract import _find_damage as _text_damage, _norm as _text_norm
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, HttpOptions
from rapidfuzz import fuzz, process

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from strategies.base import Decision, Strategy

log = logging.getLogger(__name__)

API_BASE     = "https://truckgenerator-production.up.railway.app"
GCP_PROJECT  = os.environ.get("GCP_PROJECT", "dg-ml-dev")
GCP_LOCATION = "europe-west1"
MODEL        = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

SUPPLIER_MATCH_THRESHOLD = 72

# Noise tokens stripped before matching (articles + industry-generic words).
# Corporate suffixes (GmbH, AG, Inc, Corp, …) are deliberately KEPT —
# they distinguish entities sharing a common root (Meridian Corp ≠ Meridian Holdings).
_NOISE_TOKENS = {
    "the", "le", "la", "les", "il", "der", "die", "das",
    "logistics", "logistik", "logistique", "logistica",
    "transport", "transports", "transporte", "trasporti",
    "spedition", "spedizione", "freight", "shipping", "express",
}
_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)


def _normalize(name: str) -> str:
    """Lowercase, umlaut-fold, strip punctuation, remove noise tokens."""
    s = name.lower()
    s = (s.replace("ä", "a").replace("ö", "o").replace("ü", "u")
           .replace("ß", "ss").replace("é", "e").replace("è", "e")
           .replace("ê", "e").replace("à", "a").replace("ç", "c"))
    s = _NON_WORD.sub(" ", s)
    tokens = [t for t in s.split() if t and t not in _NOISE_TOKENS]
    return " ".join(tokens)


def _phonetic_key(name: str) -> str:
    """Metaphone fingerprint per word — collapses Muller/Mueller, ue/ü, etc."""
    words = _normalize(name).split()
    codes = []
    for w in words:
        try:
            c = jellyfish.metaphone(w)
        except Exception:
            c = ""
        if c:
            codes.append(c)
    return " ".join(codes)


SYSTEM_PROMPT = """\
You are a warehouse logistics agent. Analyze the provided truck documentation \
and extract delivery information as JSON.

Rules:
- raw_supplier_name: Find the supplier name using these locations IN ORDER: \
  1. EMAIL Subject line — format is "Subject: [type] – SUPPLIER NAME" or \
     "Subject: [type] — SUPPLIER NAME", extract everything after the dash exactly. \
  2. EMAIL intro phrase — name follows one of these fixed phrases: \
     DE: "Firma", "informieren Sie über den heutigen Eingang von" \
     FR: "de la part de", "l'arrivée prévue de", "Société" \
     IT: "azienda", "vi comunichiamo che" \
     ES: "empresa", "en nombre de" \
     EN: "company", "on behalf of" \
  3. AUDIO — supplier name is stated in the first sentence/introduction. \
  4. EMAIL signature — company name at the very end. \
  Copy VERBATIM including legal suffixes (GmbH, AG, S.A., Inc., Ltd., KG, etc.). \
  Do NOT extract numbers, dates, stock indices, or reference codes. \
  Do NOT paraphrase or translate. Return empty string if truly not found.
- has_damage: true ONLY if a clear damage signal is present. \
  PHOTO — look for: torn/crushed packaging, wet or stained boxes, spilled/scattered \
  goods, broken pallet planks, cracked or dented containers, deformed cardboard, \
  visible debris. Undamaged goods appear intact, dry, and neatly stacked. \
  EMAIL/AUDIO urgency markers (only appear in damage alerts, never in normal messages): \
  Alert, Alerta, Attenzione, Urgent, Urgente, Kritisch, Achtung, Warnung, Atencion, \
  Avviso urgente, Attention. \
  EMAIL/AUDIO explicit damage vocabulary: \
  beschädigt, Transportschaden, damaged, cargo damage, damage detected, \
  endommagé, dégâts, dégâts importants, détérioration, avarie, \
  danneggiato, danni alla merce, gravi danni, avaria, \
  danos graves, danos en la carga, defekt, deterioration. \
  When any of these signals is present, set true. When in doubt → true. \
  Only set false when all documentation is clearly undamaged.
- goods_type:
    "perishable" = food, beverages, medicine, fresh produce, dairy, frozen, cold chain, \
    refrigerated. Keywords: Kühlware, Frischeware, frisch, gefroren, Lebensmittel, \
    verderblich, alimentaire, frigo, surgelé, réfrigéré, deperibile, farmaci, perecedero.
    "oversized" = heavy machinery, vehicles, construction equipment, bulk raw materials. \
    Keywords: Schwergut, sperrig, Sperrgut, Übermaß, encombrant, ingombrante, bulky.
    "standard" = everything else.
- unit: "parcels" = individual boxes/packages/Pakete/colis/colli/pacco/pacchi. \
  "pallets" = Paletten/palettes/bancali/paletas.
- parcel_count: the integer quantity of parcels/pallets in this shipment. \
  It appears ADJACENT to the unit word. \
  IGNORE: dates, reference numbers, phone numbers, stock indices (S&P 500, MIDCAP 400).
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "required": ["raw_supplier_name", "parcel_count", "unit", "goods_type", "has_damage"],
    "properties": {
        "raw_supplier_name": {"type": "STRING"},
        "parcel_count":      {"type": "INTEGER"},
        "unit":              {"type": "STRING", "enum": ["parcels", "pallets"]},
        "goods_type":        {"type": "STRING",
                              "enum": ["standard", "oversized", "perishable"]},
        "has_damage":        {"type": "BOOLEAN"},
    },
}


class ClaudeRampStrategy(Strategy):

    def __init__(self) -> None:
        self._client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
            http_options=HttpOptions(timeout=60000),
        )
        self._suppliers      = self._fetch_suppliers()
        self._norm_names     = [_normalize(s["supplier_name"]) for s in self._suppliers]
        self._norm_to_idx    = {name: i for i, name in enumerate(self._norm_names)}
        self._phonetic_keys  = [_phonetic_key(s["supplier_name"]) for s in self._suppliers]
        log.info("ClaudeRampStrategy ready — model=%s  suppliers=%d",
                 MODEL, len(self._suppliers))

    def _fetch_suppliers(self) -> list[dict]:
        resp = requests.get(f"{API_BASE}/suppliers", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _resolve_supplier(self, raw_name: str) -> tuple[int, str]:
        norm_raw = _normalize(raw_name)

        # Tier 0: exact normalized match
        if norm_raw and norm_raw in self._norm_to_idx:
            sup = self._suppliers[self._norm_to_idx[norm_raw]]
            log.debug("Supplier exact: '%s' → '%s' (%d)", raw_name, sup["supplier_name"], sup["supplier_id"])
            return sup["supplier_id"], sup["supplier_name"]

        # Tier 1: min(token_sort_ratio, WRatio) — Milan's conservative approach
        sort_hits = process.extract(norm_raw, self._norm_names,
                                    scorer=fuzz.token_sort_ratio, limit=10)
        scored = []
        for choice, s, idx in sort_hits:
            blended = min(s, fuzz.WRatio(norm_raw, choice))
            scored.append((blended, idx))
        scored.sort(reverse=True)

        best_score = scored[0][0] if scored else 0.0
        best_idx   = scored[0][1] if scored else 0

        if best_score >= SUPPLIER_MATCH_THRESHOLD:
            sup = self._suppliers[best_idx]
            log.debug("Supplier fuzzy: '%s' → '%s' (%d) score=%.0f",
                      raw_name, sup["supplier_name"], sup["supplier_id"], best_score)
            return sup["supplier_id"], sup["supplier_name"]

        # Tier 2: jaro_winkler on Metaphone codes — catches audio transcription variants
        raw_phon = _phonetic_key(raw_name)
        if raw_phon:
            best_phon_score, best_phon_idx = 0.0, 0
            for i, pk in enumerate(self._phonetic_keys):
                if not pk:
                    continue
                try:
                    sim = jellyfish.jaro_winkler_similarity(raw_phon, pk)
                except Exception:
                    continue
                score = sim * 100.0
                if score > best_phon_score:
                    best_phon_score, best_phon_idx = score, i
            if best_phon_score >= 70:
                sup = self._suppliers[best_phon_idx]
                log.debug("Supplier phonetic: '%s' → '%s' (%d) jw=%.0f",
                          raw_name, sup["supplier_name"], sup["supplier_id"], best_phon_score)
                return sup["supplier_id"], sup["supplier_name"]

        # Tier 3: token overlap — partial audio names
        raw_tokens = set(norm_raw.split())
        best_tok_idx, best_tok_score = 0, 0
        if raw_tokens:
            for i, norm in enumerate(self._norm_names):
                score = len(raw_tokens & set(norm.split()))
                if score > best_tok_score:
                    best_tok_score, best_tok_idx = score, i
        if best_tok_score >= 2:
            sup = self._suppliers[best_tok_idx]
            log.debug("Supplier token: '%s' → '%s' (%d) tok=%d",
                      raw_name, sup["supplier_name"], sup["supplier_id"], best_tok_score)
            return sup["supplier_id"], sup["supplier_name"]

        # Tier 4: last resort — best fuzzy regardless of threshold
        sup = self._suppliers[best_idx]
        log.warning("Low-confidence supplier (score=%.0f): '%s' → '%s' (%d)",
                    best_score, raw_name, sup["supplier_name"], sup["supplier_id"])
        return sup["supplier_id"], sup["supplier_name"]

    def decide(self, truck: dict) -> Decision:
        docs      = truck.get("documentation", [])
        ramp_list = truck.get("ramp_status", [])

        parts = self._build_parts(docs)
        data  = self._extract_with_gemini(parts)

        sid, sname = self._resolve_supplier(data["raw_supplier_name"])

        # Gemini handles photo + audio damage; regex handles email text (proven, exact)
        has_damage = data["has_damage"]
        for doc in docs:
            if doc.get("type") == "email":
                text = doc.get("text", doc.get("body", ""))
                if text and _text_damage(_text_norm(text)):
                    has_damage = True
                    log.info("Regex damage detected in email text")

        log.info("raw='%s' → id=%s  count=%s unit=%s damage=%s goods=%s",
                 data["raw_supplier_name"], sid,
                 data["parcel_count"], data["unit"], has_damage, data["goods_type"])

        if has_damage:
            return Decision(
                endpoint="reject-truck", supplier_id=sid, supplier_name=sname,
                parcel_count=data["parcel_count"], has_damage=True,
                unit=data["unit"], assigned_ramp=None,
            )

        ramp = self._select_ramp(data["goods_type"], data["unit"],
                                 data["parcel_count"], ramp_list)
        log.info("Selected ramp: %s", ramp)

        return Decision(
            endpoint="assign-ramp", supplier_id=sid, supplier_name=sname,
            parcel_count=data["parcel_count"], has_damage=False,
            unit=data["unit"], assigned_ramp=ramp,
        )

    def _abs_url(self, url: str) -> str:
        return url if url.startswith("http") else f"{API_BASE}{url}"

    def _build_parts(self, docs: list[dict]) -> list:
        parts = []
        for doc in docs:
            dtype = doc.get("type", "")
            if dtype in ("photo", "image"):
                try:
                    img_bytes = requests.get(self._abs_url(doc["url"]), timeout=15).content
                    parts.append(types.Part(
                        inline_data=types.Blob(data=img_bytes, mime_type="image/jpeg")
                    ))
                    log.debug("Image added (%d bytes)", len(img_bytes))
                except Exception as exc:
                    log.warning("Could not download image %s: %s", doc.get("url"), exc)
            elif dtype == "email":
                text = doc.get("text", doc.get("body", ""))
                if text:
                    parts.append(types.Part(text=f"[EMAIL]\n{text}"))
            elif dtype == "audio":
                try:
                    audio_bytes = requests.get(self._abs_url(doc["url"]), timeout=15).content
                    parts.append(types.Part(
                        inline_data=types.Blob(data=audio_bytes, mime_type="audio/mpeg")
                    ))
                    log.debug("Audio added (%d bytes)", len(audio_bytes))
                except Exception as exc:
                    log.warning("Could not download audio %s: %s", doc.get("url"), exc)
        parts.append(types.Part(text="Extract the truck delivery data as JSON."))
        return parts

    def _extract_with_gemini(self, parts: list) -> dict:
        config = GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
        )
        response = self._client.models.generate_content(
            model=MODEL, contents=parts, config=config,
        )
        return json.loads(response.text)

    def _select_ramp(self, goods_type: str, unit: str,
                     count: int, raw_ramp_list: list[dict]) -> str | None:
        ramp_info = {
            r["ramp"]: {"status": r.get("status", "unknown"),
                        "queue":  r.get("queue_length", r.get("queue", 0))}
            for r in raw_ramp_list
        }

        def best_from(candidates: list[str]) -> str | None:
            present  = [r for r in candidates if r in ramp_info]
            free     = [r for r in present if ramp_info[r]["status"] == "free"]
            if free:
                return free[0]
            by_queue = sorted(present, key=lambda r: ramp_info[r]["queue"])
            return by_queue[0] if by_queue else None

        if goods_type == "perishable":
            groups = [["R07"], ["R01", "R02"], ["R03", "R04", "R05", "R06", "R08"]]
        elif unit == "parcels":
            groups = [["R01", "R02"], ["R03", "R04", "R05", "R06", "R07", "R08"]]
        elif unit == "pallets" and count > 32:
            groups = [["R08"], ["R05", "R06"], ["R03", "R04"]]
        elif goods_type == "oversized":
            groups = [["R05", "R06"], ["R03", "R04"], ["R07", "R08"]]
        else:
            groups = [["R03", "R04"], ["R05", "R06"], ["R07", "R08"]]

        for group in groups:
            ramp = best_from(group)
            if ramp:
                return ramp

        log.warning("No ramp found — ramp_info=%s", ramp_info)
        return None
