"""
Supplier directory: pulled once at startup from /suppliers, then resolves
free-text supplier names (extracted from emails/audio/photos) to the
canonical supplier_id via fuzzy matching.
"""

from __future__ import annotations

import logging
import re

import httpx
import jellyfish
from rapidfuzz import fuzz, process

from config import SUPPLIERS_URL, SUPPLIER_MATCH_THRESHOLD

log = logging.getLogger(__name__)


# Common legal-form / filler tokens we strip before matching so that
# "Müller Logistics AG" ↔ "muller logistik" still aligns.
_NOISE_TOKENS = {
    # Articles only. We deliberately KEEP corporate suffixes (corp, inc, plc,
    # ltd, holdings, group, …) because they distinguish entities sharing a
    # common root, e.g. "Meridian Corp" vs "Meridian Holdings".
    "the", "le", "la", "les", "il", "der", "die", "das",
    # Industry-generic words that are pure filler in supplier names.
    "logistics", "logistik", "logistique", "logistica",
    "transport", "transports", "transporte", "trasporti",
    "spedition", "spedizione", "freight", "shipping", "express",
}
_NON_WORD = re.compile(r"[^\w\s]+", re.UNICODE)


def _normalize(name: str) -> str:
    s = name.lower()
    s = (s.replace("ä", "a").replace("ö", "o").replace("ü", "u")
           .replace("ß", "ss").replace("é", "e").replace("è", "e")
           .replace("ê", "e").replace("à", "a").replace("ç", "c"))
    s = _NON_WORD.sub(" ", s)
    tokens = [t for t in s.split() if t and t not in _NOISE_TOKENS]
    return " ".join(tokens)


def _metaphone(text: str) -> str:
    """Concatenate metaphone codes of each word so multi-word company names
    keep their word-level phonetic signature. Empty for inputs jellyfish
    can't encode."""
    if not text:
        return ""
    parts = []
    for word in text.split():
        try:
            code = jellyfish.metaphone(word)
        except Exception:
            code = ""
        if code:
            parts.append(code)
    return " ".join(parts)


class SupplierIndex:
    def __init__(self, suppliers: list[dict]):
        self.suppliers = suppliers
        self._by_normalized: dict[str, int] = {}
        self._choices: list[str] = []
        self._choice_to_id: dict[str, int] = {}
        self._phonetic_codes: dict[str, str] = {}

        for s in suppliers:
            sid  = int(s["supplier_id"])
            name = str(s["supplier_name"])
            norm = _normalize(name)
            if not norm:
                continue
            self._by_normalized.setdefault(norm, sid)
            self._choices.append(norm)
            self._choice_to_id[norm] = sid
            self._phonetic_codes[norm] = _metaphone(norm)

    @classmethod
    async def load(cls) -> "SupplierIndex":
        import asyncio as _asyncio
        log.info("Loading suppliers from %s", SUPPLIERS_URL)
        last_exc: Exception | None = None
        for attempt in range(1, 5):
            try:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    r = await client.get(SUPPLIERS_URL)
                    r.raise_for_status()
                    data = r.json()
                log.info("Loaded %d suppliers (attempt %d)", len(data), attempt)
                return cls(data)
            except (httpx.RemoteProtocolError, httpx.ReadError,
                    httpx.ConnectError, httpx.ReadTimeout) as e:
                last_exc = e
                wait = 1.5 * attempt
                log.warning("Suppliers load attempt %d failed (%s) — retrying in %.1fs",
                            attempt, type(e).__name__, wait)
                await _asyncio.sleep(wait)
        raise RuntimeError("Failed to load /suppliers after 4 attempts") from last_exc

    _PLACEHOLDER_NAMES = {
        "unbekannt", "unknown", "n a", "na", "nicht bekannt", "inconnu",
        "sconosciuto", "desconocido", "anonymous", "?", "",
    }

    def top_k(self, raw_name: str, k: int = 5) -> list[tuple[int, str, float]]:
        """
        Return up to k candidate suppliers ranked by the BEST of:
          - string fuzzy: min(token_sort_ratio, WRatio)
          - phonetic: jaro_winkler on metaphone codes × 100
        Used by the LLM-disambiguation step. Phonetic catches whisper errors
        like "Schwerbus" / "Airbus" that string fuzzy misses entirely.
        """
        if not raw_name or not raw_name.strip():
            return []
        norm = _normalize(raw_name)
        if not norm or len(norm) < 3 or norm in self._PLACEHOLDER_NAMES:
            return []

        scored: dict[str, float] = {}

        # 1. Top string-fuzzy candidates (conservative — min of two scorers).
        sort_hits = process.extract(
            norm, self._choices,
            scorer=fuzz.token_sort_ratio,
            limit=max(k * 4, 15),
        )
        for choice, s, _ in sort_hits:
            blended = min(s, fuzz.WRatio(norm, choice))
            if blended > scored.get(choice, 0):
                scored[choice] = blended

        # 2. Phonetic candidates: jaro_winkler over metaphone codes of all
        # suppliers. ~9k iterations, microseconds each — well under 100 ms.
        raw_phon = _metaphone(norm)
        if raw_phon:
            for choice, choice_phon in self._phonetic_codes.items():
                if not choice_phon:
                    continue
                try:
                    sim = jellyfish.jaro_winkler_similarity(raw_phon, choice_phon)
                except Exception:
                    continue
                score = sim * 100.0
                if score >= 70 and score > scored.get(choice, 0):
                    scored[choice] = score

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        out: list[tuple[int, str, float]] = []
        for choice, score in ranked[:k]:
            sid = self._choice_to_id[choice]
            out.append((sid, self._canonical(sid), float(score)))
        return out

    def resolve(self, raw_name: str) -> tuple[int, str, float]:
        """
        Return (supplier_id, canonical_name, score).
        Falls back to first supplier id with score=0 if nothing matches —
        we still have to send *some* id, but log a warning.

        Scoring strategy:
          - exact normalized match → 100 (and we're done)
          - else compute token_sort_ratio AND WRatio, take the MIN (require
            both scorers to be confident — avoids partial_ratio / token_set
            blowups like "C.A.P.A." → "BETTERWARE…S.A.P.I."
          - if no candidate clears SUPPLIER_MATCH_THRESHOLD, log top 3 and
            fall back to placeholder
        """
        if not raw_name or not raw_name.strip():
            sid = int(self.suppliers[0]["supplier_id"])
            return sid, self.suppliers[0]["supplier_name"], 0.0

        norm = _normalize(raw_name)

        if norm in self._PLACEHOLDER_NAMES or len(norm) < 3:
            log.warning("Supplier name is a placeholder/too short (raw=%r norm=%r) — "
                        "skipping fuzzy match", raw_name, norm)
            sid = int(self.suppliers[0]["supplier_id"])
            return sid, self.suppliers[0]["supplier_name"], 0.0

        if norm in self._by_normalized:
            sid = self._by_normalized[norm]
            return sid, self._canonical(sid), 100.0

        sort_hits  = process.extract(norm, self._choices, scorer=fuzz.token_sort_ratio, limit=5)
        wratio_map = {c: fuzz.WRatio(norm, c) for c, _, _ in sort_hits}

        scored = [(c, min(s, wratio_map[c])) for c, s, _ in sort_hits]
        scored.sort(key=lambda x: x[1], reverse=True)

        log.debug("Supplier candidates for %r (norm=%r): %s",
                  raw_name, norm,
                  [(c, s) for c, s in scored[:3]])

        if scored and scored[0][1] >= SUPPLIER_MATCH_THRESHOLD:
            best_choice, best_score = scored[0]
            sid = self._choice_to_id[best_choice]
            return sid, self._canonical(sid), float(best_score)

        log.warning("Supplier fuzzy-match below threshold %d (raw=%r best=%s) — "
                    "falling back to first supplier",
                    SUPPLIER_MATCH_THRESHOLD, raw_name,
                    scored[0] if scored else None)
        sid = int(self.suppliers[0]["supplier_id"])
        return sid, self.suppliers[0]["supplier_name"], float(scored[0][1] if scored else 0)

    def _canonical(self, sid: int) -> str:
        for s in self.suppliers:
            if int(s["supplier_id"]) == sid:
                return str(s["supplier_name"])
        return ""
