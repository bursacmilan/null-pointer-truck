"""Supplier list loading + name → canonical id resolution.

Documentation (esp. emails) contains the EXACT canonical supplier name, so we
try exact matching first and only fall back to fuzzy matching for garbled audio
transcripts. Fuzzy uses a length-aware scorer (WRatio) so that a full name like
"Actavis plc (NYSE:ACT) added to S&P 500" is not collapsed onto a shorter
near-duplicate "Actavis".
"""

import json
import logging
import re
import urllib.request

from rapidfuzz import fuzz, process

from config import API_BASE

log = logging.getLogger("suppliers")

_WS = re.compile(r"\s+")


def _light(name: str) -> str:
    """Lower-case + collapse whitespace + trim edge punctuation. Keeps inner
    punctuation so exact canonical strings still match exactly."""
    s = _WS.sub(" ", name.lower()).strip()
    return s.strip(" .,:;-–—")


class SupplierIndex:
    def __init__(self, suppliers: list[dict]):
        self.suppliers = suppliers
        self.by_id = {s["supplier_id"]: s["supplier_name"] for s in suppliers}

        self._exact: dict[str, int] = {}
        self._light_names: list[str] = []
        self._ids: list[int] = []
        for s in suppliers:
            key = _light(s["supplier_name"])
            # first occurrence wins for exact lookup
            self._exact.setdefault(key, s["supplier_id"])
            self._light_names.append(key)
            self._ids.append(s["supplier_id"])
        log.info("Indexed %d suppliers (%d unique exact keys)",
                 len(suppliers), len(self._exact))

    def match(self, raw_name: str) -> tuple[int, str, float]:
        """Return (supplier_id, canonical_name, confidence 0-100)."""
        if not raw_name or not raw_name.strip():
            sid = self._ids[0]
            return sid, self.by_id[sid], 0.0

        key = _light(raw_name)

        # 1) exact match (documentation usually carries the canonical name verbatim)
        if key in self._exact:
            sid = self._exact[key]
            return sid, self.by_id[sid], 100.0

        # 2) length-aware fuzzy fallback (garbled audio names)
        hit = process.extractOne(
            key, self._light_names, scorer=fuzz.WRatio, score_cutoff=0
        )
        if hit is None:
            sid = self._ids[0]
            return sid, self.by_id[sid], 0.0
        _, score, idx = hit
        sid = self._ids[idx]
        return sid, self.by_id[sid], float(score)


def fetch_suppliers() -> list[dict]:
    log.info("Fetching supplier list from %s/suppliers", API_BASE)
    with urllib.request.urlopen(f"{API_BASE}/suppliers") as r:
        data = json.loads(r.read())
    log.info("Fetched %d suppliers", len(data))
    return data


def load_index() -> SupplierIndex:
    return SupplierIndex(fetch_suppliers())
