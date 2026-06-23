#!/usr/bin/env python3
"""
GeminiRampStrategy — Gemini 2.5 Flash via company Vertex AI.

Supplier resolution is done locally with fuzzy matching so we never need
to send 9169 supplier names to the LLM. Gemini just extracts the raw name
it hears/sees; we match it against the full list with difflib.
"""

import difflib
import json
import logging
import os
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

SYSTEM_PROMPT = """\
You are a warehouse logistics agent. Analyze the provided truck documentation \
(photo, supplier email, and optionally a driver audio note) and extract delivery \
information as JSON.

Rules:
- raw_supplier_name: copy the supplier name EXACTLY as mentioned in the documentation \
  (email sender/signature, spoken in audio). Do not paraphrase or abbreviate.
- has_damage: true if ANY source (visual damage in photo, text mentions damage, \
  audio mentions damage). Be conservative — when in doubt, mark as damaged.
- goods_type: "perishable" = food, medicine, refrigerated, cold chain; \
  "oversized" = heavy machinery, bulk freight, exceptionally large items; \
  "standard" = everything else.
- unit: "parcels" if individual packages/boxes are counted; "pallets" if pallets/Paletten.
- parcel_count: the integer quantity stated in the documentation.
"""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "required": ["raw_supplier_name", "parcel_count", "unit", "has_damage", "goods_type"],
    "properties": {
        "raw_supplier_name": {
            "type": "STRING",
            "description": "Supplier name exactly as it appears in the documentation",
        },
        "parcel_count": {"type": "INTEGER"},
        "unit":         {"type": "STRING", "enum": ["parcels", "pallets"]},
        "has_damage":   {"type": "BOOLEAN"},
        "goods_type":   {"type": "STRING",
                         "enum": ["standard", "oversized", "perishable"]},
    },
}


class ClaudeRampStrategy(Strategy):

    def __init__(self) -> None:
        self._client = genai.Client(
            vertexai=True,
            project=GCP_PROJECT,
            location=GCP_LOCATION,
            http_options=HttpOptions(timeout=60000),  # milliseconds
        )
        self._suppliers     = self._fetch_suppliers()
        self._supplier_names = [s["supplier_name"] for s in self._suppliers]
        self._name_to_sup   = {s["supplier_name"]: s for s in self._suppliers}
        log.info("GeminiRampStrategy ready — model=%s  suppliers=%d",
                 MODEL, len(self._suppliers))

    def _fetch_suppliers(self) -> list[dict]:
        resp = requests.get(f"{API_BASE}/suppliers", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _resolve_supplier(self, raw_name: str) -> tuple[int, str]:
        """Fuzzy-match raw_name against full supplier list locally."""
        matches = difflib.get_close_matches(
            raw_name, self._supplier_names, n=1, cutoff=0.3
        )
        if matches:
            sup = self._name_to_sup[matches[0]]
            log.debug("Supplier match: '%s' → '%s' (%d)", raw_name, sup["supplier_name"], sup["supplier_id"])
            return sup["supplier_id"], sup["supplier_name"]

        # Fallback: token-based partial match
        raw_tokens = set(raw_name.lower().split())
        best, best_score = None, 0
        for name, sup in self._name_to_sup.items():
            tokens = set(name.lower().split())
            score = len(raw_tokens & tokens)
            if score > best_score:
                best, best_score = sup, score

        if best and best_score > 0:
            log.debug("Supplier token-match: '%s' → '%s' (%d)", raw_name, best["supplier_name"], best["supplier_id"])
            return best["supplier_id"], best["supplier_name"]

        log.warning("No supplier match for '%s' — using placeholder", raw_name)
        return self._suppliers[0]["supplier_id"], self._suppliers[0]["supplier_name"]

    def decide(self, truck: dict) -> Decision:
        docs      = truck.get("documentation", [])
        ramp_list = truck.get("ramp_status", [])

        parts = self._build_parts(docs)
        data  = self._extract_with_gemini(parts)

        supplier_id, supplier_name = self._resolve_supplier(data["raw_supplier_name"])

        log.info("Extracted: raw_name='%s' → supplier_id=%s  count=%s  unit=%s  damage=%s  goods=%s",
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
