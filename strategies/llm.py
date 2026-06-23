"""
LLMStrategy — runs whisper.cpp + Claude in parallel per truck, fuzzy-resolves
the supplier, then picks a ramp via the deterministic rule set.
"""

from __future__ import annotations

import asyncio
import logging

from config import ANTHROPIC_FALLBACK_MODEL
from .audio import WhisperTranscriber
from .base import Decision, Strategy
from .extract import ClaudeExtractor
from .ramps import select_ramp
from .suppliers import SupplierIndex

log = logging.getLogger(__name__)

_PLACEHOLDER_NAMES = {
    "", "unbekannt", "unknown", "anonymous", "n a", "na", "?", "n/a",
    "inconnu", "sconosciuto", "desconocido", "nicht bekannt",
}

_UNCERTAINTY_MARKERS = (
    "unklar", "fehlerhaft", "nicht verwertbar", "unintelligibel",
    "unverständlich", "unverstaendlich", "nicht identifizierbar",
    "nicht erkennbar", "garbled", "garbage", "fragmentarisch",
    "konservative standardannahme", "nicht lesbar", "no information",
    "nicht ermittelbar",
)


def _needs_fallback(extracted: dict) -> bool:
    raw_name = (extracted.get("supplier_name_raw") or "").strip().lower()
    if raw_name in _PLACEHOLDER_NAMES:
        return True
    if len(raw_name) < 3:
        return True
    reasoning = (extracted.get("reasoning") or "").lower()
    if any(marker in reasoning for marker in _UNCERTAINTY_MARKERS):
        return True
    return False


def _safe_int(value, default: int = 1) -> int:
    try:
        n = int(value)
        return n if n >= 1 else default
    except (TypeError, ValueError):
        return default


def _safe_unit(value) -> str:
    return value if value in ("parcels", "pallets") else "parcels"


def _safe_goods(value) -> str:
    return value if value in ("standard", "oversized", "perishable") else "standard"


def _doc_of(truck: dict, kind: str) -> dict | None:
    for d in truck.get("documentation", []):
        if d.get("type") == kind:
            return d
    return None


class LLMStrategy(Strategy):
    """
    Orchestrates extraction + decision per truck:

      transcribe(audio)  ─┐
                          ├──→ Claude(photo + email + transcript) ──→ extracted dict
      download(photo)   ──┘                                          │
                                                                     ▼
                                          fuzzy supplier match + ramp rules
                                                                     │
                                                                     ▼
                                                                  Decision
    """

    def __init__(self, suppliers: SupplierIndex,
                 transcriber: WhisperTranscriber,
                 extractor:  ClaudeExtractor):
        self.suppliers = suppliers
        self.transcriber = transcriber
        self.extractor = extractor

    @classmethod
    async def bootstrap(cls) -> "LLMStrategy":
        suppliers = await SupplierIndex.load()
        transcriber = WhisperTranscriber()
        # Pre-load whisper now so the first truck doesn't pay the model-load
        # latency (ggml weights may be downloaded from HF on first run).
        await transcriber.ensure_loaded()
        extractor = ClaudeExtractor()
        return cls(suppliers, transcriber, extractor)

    async def decide(self, truck: dict) -> Decision:
        truck_id = truck.get("truck_id", "<?>")

        photo_doc = _doc_of(truck, "photo")
        email_doc = _doc_of(truck, "email")
        audio_doc = _doc_of(truck, "audio")

        email_text = (email_doc or {}).get("text", "") or ""

        # Run audio transcription + photo download concurrently.
        async def _transcribe() -> str:
            if not audio_doc or "url" not in audio_doc:
                return ""
            try:
                return await self.transcriber.transcribe_url(audio_doc["url"])
            except Exception:
                log.exception("Audio transcription failed for %s", truck_id)
                return ""

        async def _photo() -> tuple[bytes | None, str | None]:
            if not photo_doc or "url" not in photo_doc:
                return None, None
            try:
                return await self.extractor.fetch_photo(photo_doc["url"])
            except Exception:
                log.exception("Photo download failed for %s", truck_id)
                return None, None

        transcript, (photo_bytes, photo_media) = await asyncio.gather(
            _transcribe(), _photo()
        )
        log.debug("[%s] transcript=%r email_len=%d photo_bytes=%s",
                  truck_id, transcript, len(email_text),
                  len(photo_bytes) if photo_bytes else None)

        extracted = await self.extractor.extract(
            photo_bytes, photo_media, email_text, transcript,
        )
        log.info("[%s] extracted (haiku): %s", truck_id, extracted)

        if _needs_fallback(extracted):
            log.info("[%s] low-confidence Haiku output — retrying with %s",
                     truck_id, ANTHROPIC_FALLBACK_MODEL)
            try:
                fallback = await self.extractor.extract(
                    photo_bytes, photo_media, email_text, transcript,
                    model=ANTHROPIC_FALLBACK_MODEL,
                )
                log.info("[%s] extracted (sonnet): %s", truck_id, fallback)
                if not _needs_fallback(fallback):
                    extracted = fallback
                else:
                    # If even Sonnet sees nothing, keep the better of the two
                    # by preferring the one with a non-placeholder name.
                    fb_name = (fallback.get("supplier_name_raw") or "").strip().lower()
                    if fb_name and fb_name not in _PLACEHOLDER_NAMES:
                        extracted = fallback
            except Exception:
                log.exception("[%s] Sonnet fallback failed — keeping Haiku output", truck_id)

        raw_name = extracted["supplier_name_raw"]
        sid, canonical_name, score = self.suppliers.resolve(raw_name)
        log.info("[%s] supplier_id=%s (raw=%r → %r, score=%.0f)",
                 truck_id, sid, raw_name, canonical_name, score)

        # When the top fuzzy match isn't decisively high, ask Claude to pick from
        # the top-K candidates — it can use phonetic similarity + company knowledge
        # to disambiguate cases like "ATI" → Allegheny Technologies vs Aptiv.
        if score < 90:
            candidates = self.suppliers.top_k(raw_name, k=5)
            if candidates and len(candidates) >= 2:
                # Skip if all candidates are below a minimal usefulness floor.
                if candidates[0][2] >= 50:
                    try:
                        picked_name = await self.extractor.pick_supplier(
                            raw_name, candidates, email_text, transcript,
                        )
                    except Exception:
                        log.exception("[%s] disambiguation crashed — keeping fuzzy pick", truck_id)
                        picked_name = None
                    if picked_name and picked_name != canonical_name:
                        for cand_sid, cand_name, cand_score in candidates:
                            if cand_name == picked_name:
                                log.info("[%s] supplier OVERRIDE by LLM: %r (id=%s, fuzzy=%.0f) "
                                         "instead of %r (id=%s, fuzzy=%.0f)",
                                         truck_id, picked_name, cand_sid, cand_score,
                                         canonical_name, sid, score)
                                sid = cand_sid
                                canonical_name = cand_name
                                score = cand_score
                                break

        unit = _safe_unit(extracted.get("unit"))
        count = _safe_int(extracted.get("parcel_count"))
        has_damage = bool(extracted.get("has_damage"))
        goods_type = _safe_goods(extracted.get("goods_type"))

        if has_damage:
            return Decision(
                endpoint      = "reject-truck",
                supplier_id   = sid,
                supplier_name = canonical_name or extracted["supplier_name_raw"],
                parcel_count  = count,
                has_damage    = True,
                unit          = unit,
            )

        ramp = select_ramp(unit, count, goods_type, truck.get("ramp_status", []))
        log.info("[%s] goods=%s → ramp %s", truck_id, goods_type, ramp)

        return Decision(
            endpoint      = "assign-ramp",
            supplier_id   = sid,
            supplier_name = canonical_name or extracted["supplier_name_raw"],
            parcel_count  = count,
            has_damage    = False,
            unit          = unit,
            assigned_ramp = ramp,
        )
