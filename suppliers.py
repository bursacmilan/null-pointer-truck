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

import numpy as np
from rapidfuzz import fuzz, process

try:
    import jellyfish
    _HAVE_PHONETIC = True
except ImportError:
    _HAVE_PHONETIC = False

from config import API_BASE

log = logging.getLogger("suppliers")

_WS    = re.compile(r"\s+")
_TOKEN = re.compile(r"[^a-z0-9]+")

# Weight of lexical (WRatio) vs phonetic (metaphone) similarity in the fuzzy
# fallback. TTS+whisper garble names phonetically (Carvana→"Karane",
# Columbia→"Klambia"), so a phonetic component recovers names pure edit-distance
# misses. 0.65 chosen as a robust mid-point (not tuned to a single run).
_LEX_WEIGHT = 0.65


def _light(name: str) -> str:
    """Lower-case + collapse whitespace + trim edge punctuation. Keeps inner
    punctuation so exact canonical strings still match exactly."""
    s = _WS.sub(" ", name.lower()).strip()
    return s.strip(" .,:;-–—")


def _phonetic(name: str) -> str:
    """Metaphone code per significant token, joined — a phonetic fingerprint."""
    if not _HAVE_PHONETIC:
        return ""
    toks = [t for t in _TOKEN.split(_light(name)) if len(t) > 1]
    return " ".join(jellyfish.metaphone(t) for t in toks)


class SupplierIndex:
    def __init__(self, suppliers: list[dict]):
        self.suppliers = suppliers
        self.by_id = {s["supplier_id"]: s["supplier_name"] for s in suppliers}

        self._exact: dict[str, int] = {}
        self._light_names: list[str] = []
        self._phon_names: list[str] = []
        self._ids: list[int] = []
        for s in suppliers:
            key = _light(s["supplier_name"])
            self._exact.setdefault(key, s["supplier_id"])   # first occurrence wins
            self._light_names.append(key)
            self._phon_names.append(_phonetic(s["supplier_name"]))
            self._ids.append(s["supplier_id"])
        log.info("Indexed %d suppliers (%d unique exact keys, phonetic=%s)",
                 len(suppliers), len(self._exact), _HAVE_PHONETIC)

    def match(self, raw_name: str) -> tuple[int, str, float]:
        """Return (supplier_id, canonical_name, confidence 0-100).

        Exact match first (documentation carries the canonical name verbatim);
        otherwise a blended lexical+phonetic fuzzy match for garbled audio names.
        """
        if not raw_name or not raw_name.strip():
            sid = self._ids[0]
            return sid, self.by_id[sid], 0.0

        key = _light(raw_name)
        if key in self._exact:                       # 1) exact
            sid = self._exact[key]
            return sid, self.by_id[sid], 100.0

        # 2) blended fuzzy fallback
        if _HAVE_PHONETIC:
            lex = process.cdist([key], self._light_names, scorer=fuzz.WRatio,
                                workers=-1)[0]
            phon = process.cdist([_phonetic(raw_name)], self._phon_names,
                                 scorer=fuzz.token_sort_ratio, workers=-1)[0]
            blended = _LEX_WEIGHT * lex + (1.0 - _LEX_WEIGHT) * phon
            idx = int(np.argmax(blended))
            score = float(blended[idx])
        else:
            hit = process.extractOne(key, self._light_names, scorer=fuzz.WRatio)
            idx, score = hit[2], float(hit[1])

        sid = self._ids[idx]
        return sid, self.by_id[sid], score


def fetch_suppliers() -> list[dict]:
    log.info("Fetching supplier list from %s/suppliers", API_BASE)
    with urllib.request.urlopen(f"{API_BASE}/suppliers") as r:
        data = json.loads(r.read())
    log.info("Fetched %d suppliers", len(data))
    return data


def load_index() -> SupplierIndex:
    return SupplierIndex(fetch_suppliers())
