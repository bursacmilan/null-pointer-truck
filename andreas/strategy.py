#!/usr/bin/env python3
"""
GeminiRampStrategy — Gemini 2.5 Flash via company Vertex AI.

Supplier resolution: Gemini extracts the raw name it hears/sees; we fuzzy-match
locally against all 9169 suppliers using normalized names + SequenceMatcher.
"""

import difflib
import json
import logging
import os
import re
import sys

import requests
from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig, HttpOptions

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from strategies.base import Decision, Strategy

log = logging.getLogger(__name__)

API_BASE     = "https://truckgenerator-production.up.railway.app"
GCP_PROJECT  = os.environ.get("GCP_PROJECT", "dg-ml-dev")
GCP_LOCATION = "europe-west1"
MODEL        = "gemini-2.5-flash"

# Company suffixes stripped before fuzzy matching
_SUFFIX_RE = re.compile(
    r"\b(inc|corp|co|llc|ltd|gmbh|ag|sa|bv|nv|plc|kg|kgaa|oy|ab|as|srl|spa|sas|se"
    r"|group|holding|holdings|international|global|fund|financial|capital)\b\.?",
    re.IGNORECASE,
)

SYSTEM_PROMPT = """\
You are a warehouse logistics agent. Analyze the provided truck documentation \
(photo, supplier email, and optionally a driver audio note) and extract delivery \
information as JSON.

Rules:
- raw_supplier_name: the supplier name EXACTLY as mentioned (email sender/signature, \
  spoken in audio). Copy verbatim — do NOT paraphrase or look up alternatives.
- has_damage: true if ANY signal indicates damage: visible dents/cracks in photo, \
  text words like "beschädigt / damaged / endommagé / danneggiato / defekt", \
  or audio mentions damage. Be conservative — doubt → true.
- goods_type:
    "perishable" = food, beverages, medicine, fresh produce, dairy, frozen goods, \
    cold chain, refrigerated. Keywords: Kühlware, frisch, gefroren, Lebensmittel, \
    alimentaire, frigo, surgelé, deperibile, farmaci.
    "oversized" = heavy machinery, vehicles, construction equipment, bulk raw materials, \
    exceptionally large/heavy freight. Keywords: Schwergut, sperrig, Übermaß, \
    encombrante, ingombrante, vrac.
    "standard" = everything else (electronics, textiles, auto parts, general goods).
- unit: "parcels" = individual boxes/packages/Pakete/colis/colli counted as pieces. \
  "pallets" = Paletten/palettes/pallet loads.
- parcel_count: the integer quantity stated. Extract exactly — do not estimate.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "required": ["raw_supplier_name", "parcel_count", "unit", "has_damage", "goods_type"],
    "properties": {
        "raw_supplier_name": {
            "type": "STRING",
            "description": "Supplier name verbatim from the documentation",
        },
        "parcel_count": {"type": "INTEGER"},
        "unit":         {"type": "STRING", "enum": ["parcels", "pallets"]},
        "has_damage":   {"type": "BOOLEAN"},
        "goods_type":   {"type": "STRING",
                         "enum": ["standard", "oversized", "perishable"]},
    },
}


def _normalize(name: str) -> str:
    """Lowercase, strip company suffixes and punctuation for fuzzy matching."""
    name = _SUFFIX_RE.sub(" ", name)
    name = re.sub(r"[^\w\s]", " ", name)
    return " ".join(name.lower().split())


class ClaudeRampStrategy(Strategy):

    def __init__(self) -> None:
        self._client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
            http_options=HttpOptions(timeout=60000),
        )
        self._suppliers = self._fetch_suppliers()
        # Pre-compute normalized names for fast matching
        self._norm_names = [_normalize(s["supplier_name"]) for s in self._suppliers]
        log.info("GeminiRampStrategy ready — model=%s  suppliers=%d",
                 MODEL, len(self._suppliers))

    def _fetch_suppliers(self) -> list[dict]:
        resp = requests.get(f"{API_BASE}/suppliers", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _resolve_supplier(self, raw_name: str) -> tuple[int, str]:
        """Best-effort fuzzy match: normalized ratio → token overlap → first match."""
        norm_raw = _normalize(raw_name)

        # Strategy 1: SequenceMatcher ratio on normalized full names
        best_idx, best_ratio = 0, 0.0
        for i, norm in enumerate(self._norm_names):
            r = difflib.SequenceMatcher(None, norm_raw, norm, autojunk=False).ratio()
            if r > best_ratio:
                best_ratio, best_idx = r, i

        # Strategy 2: token overlap on normalized names (helps with partial matches)
        raw_tokens = set(norm_raw.split())
        best_tok_idx, best_tok_score = 0, 0
        if raw_tokens:
            for i, norm in enumerate(self._norm_names):
                score = len(raw_tokens & set(norm.split()))
                if score > best_tok_score:
                    best_tok_score, best_tok_idx = score, i

        # Pick best: prefer ratio match, fall back to token if ratio is weak
        if best_ratio >= 0.5:
            idx = best_idx
            log.debug("Supplier ratio-match (%.2f): '%s' → '%s'",
                      best_ratio, raw_name, self._suppliers[idx]["supplier_name"])
        elif best_tok_score >= 2:
            idx = best_tok_idx
            log.debug("Supplier token-match (%d tokens): '%s' → '%s'",
                      best_tok_score, raw_name, self._suppliers[idx]["supplier_name"])
        else:
            idx = best_idx
            log.warning("Low-confidence supplier match (ratio=%.2f tok=%d): '%s' → '%s'",
                        best_ratio, best_tok_score, raw_name,
                        self._suppliers[idx]["supplier_name"])

        sup = self._suppliers[idx]
        return sup["supplier_id"], sup["supplier_name"]

    def decide(self, truck: dict) -> Decision:
        docs      = truck.get("documentation", [])
        ramp_list = truck.get("ramp_status", [])

        parts = self._build_parts(docs)
        data  = self._extract_with_gemini(parts)

        supplier_id, supplier_name = self._resolve_supplier(data["raw_supplier_name"])

        log.info("raw='%s' → id=%s  count=%s  unit=%s  damage=%s  goods=%s",
                 data["raw_supplier_name"], supplier_id,
                 data["parcel_count"], data["unit"],
                 data["has_damage"], data["goods_type"])

        if data["has_damage"]:
            return Decision(
                endpoint      = "reject-truck",
                supplier_id   = supplier_id,
                supplier_name = supplier_name,
                parcel_count  = data["parcel_count"],
                has_damage    = True,
                unit          = data["unit"],
                assigned_ramp = None,
            )

        ramp = self._select_ramp(data["goods_type"], data["unit"],
                                 data["parcel_count"], ramp_list)
        log.info("Selected ramp: %s", ramp)

        return Decision(
            endpoint      = "assign-ramp",
            supplier_id   = supplier_id,
            supplier_name = supplier_name,
            parcel_count  = data["parcel_count"],
            has_damage    = False,
            unit          = data["unit"],
            assigned_ramp = ramp,
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
            model=MODEL,
            contents=parts,
            config=config,
        )
        return json.loads(response.text)

    def _select_ramp(self, goods_type: str, unit: str,
                     count: int, raw_ramp_list: list[dict]) -> str | None:
        ramp_info: dict[str, dict] = {
            r["ramp"]: {
                "status": r.get("status", "unknown"),
                "queue":  r.get("queue_length", r.get("queue", 0)),
            }
            for r in raw_ramp_list
        }

        def best_from(candidates: list[str]) -> str | None:
            present = [r for r in candidates if r in ramp_info]
            free    = [r for r in present if ramp_info[r]["status"] == "free"]
            if free:
                return free[0]
            by_queue = sorted(present, key=lambda r: ramp_info[r]["queue"])
            return by_queue[0] if by_queue else None

        if goods_type == "perishable":
            groups = [["R07"], ["R01", "R02"], ["R03", "R04", "R05", "R06", "R08"]]
        elif unit == "parcels":
            # parcels that are perishable must go R07 — but goods_type already covers that
            groups = [["R01", "R02"], ["R03", "R04", "R05", "R06", "R07", "R08"]]
        elif unit == "pallets" and count > 32:
            groups = [["R08"], ["R05", "R06"], ["R03", "R04"]]
        elif goods_type == "oversized":
            groups = [["R05", "R06"], ["R03", "R04"], ["R07", "R08"]]
        else:
            # standard pallets ≤ 32
            groups = [["R03", "R04"], ["R05", "R06"], ["R07", "R08"]]

        for group in groups:
            ramp = best_from(group)
            if ramp:
                return ramp

        log.warning("No ramp found — ramp_info=%s", ramp_info)
        return None
