"""
SmartStrategy — the real RampRush agent.

Per truck:
  • photo URL path   → has_damage (no vision model: …/damaged/ vs …/undamaged/)
  • email text       → parse (multilingual)
  • audio URL        → whisper transcribe → parse
  • merge → fuzzy-match supplier → route to ramp
"""

import logging

import audio
import llm
import vision
from extract import Signals, merge_signals, parse_text
from suppliers import SupplierIndex

from .base import Decision, Strategy

log = logging.getLogger("smart")


def _signals_from_llm(d: dict | None) -> Signals:
    if not d:
        return Signals()
    return Signals(
        supplier_name=d.get("supplier_name"),
        parcel_count=d.get("count"),
        unit=d.get("unit"),
        goods_type=d.get("goods_type"),
    )

ALL_RAMPS = [f"R{i:02d}" for i in range(1, 9)]


class SmartStrategy(Strategy):
    def __init__(self, supplier_index: SupplierIndex):
        self.suppliers = supplier_index

    # ── signal gathering ──────────────────────────────────────────────────────

    def _url_oracle(self, url: str) -> bool | None:
        """Damage from the URL path — the soon-to-be-patched exploit. Kept ONLY
        for logging/diagnostics; NOT used in the decision (see _decide_damage)."""
        u = url.lower()
        if "/undamaged/" in u:
            return False
        if "/damaged/" in u:
            return True
        return None

    def _gather(self, truck: dict):
        email_sig: Signals | None = None
        audio_sig: Signals | None = None
        photo_url: str | None = None
        supplier_candidates: list[str] = []   # all transcript-derived name guesses

        def add_name(sig: Signals):
            if sig.supplier_name:
                supplier_candidates.append(sig.supplier_name)

        for doc in truck.get("documentation", []):
            dtype = doc.get("type")
            if dtype == "email":
                email_sig = parse_text(doc.get("text", ""))
                add_name(email_sig)
            elif dtype == "photo":
                photo_url = doc.get("url", "")
            elif dtype == "audio":
                try:
                    # Candidates are ordered raw-first (best supplier-name fidelity),
                    # then english translation (best unit/goods vocabulary). Merging
                    # keeps the raw fields and lets later candidates fill the gaps.
                    audio_sig = Signals()
                    transcripts = audio.transcribe_candidates(doc["url"])
                    for cand in transcripts:
                        cand_sig = parse_text(cand)
                        add_name(cand_sig)            # keep every name variant
                        audio_sig = merge_signals(audio_sig, cand_sig)

                    # LLM augmentation on the best (last = translation) transcript:
                    # robust to messy phrasing; regex stays primary, LLM fills gaps.
                    llm_sig = _signals_from_llm(llm.extract(transcripts[-1]))
                    add_name(llm_sig)
                    audio_sig = merge_signals(audio_sig, llm_sig)
                except Exception:
                    log.exception("Audio transcription failed for %s", doc.get("url"))

        # email text fields take priority; audio fills gaps
        merged = merge_signals(email_sig or Signals(), audio_sig)
        return merged, photo_url, (email_sig is not None), supplier_candidates

    def _decide_damage(self, email_present: bool, photo_url: str | None,
                       text_damage: bool | None) -> bool:
        """General damage detection (no URL exploit):
          • email present → trust the text alert (reliable for email trucks,
            and avoids exposing intact email trucks to vision false-positives);
          • else photo present → local vision model on the image, OR a text
            alert from the audio; falls back to text if vision is unavailable;
          • else → text alert only.
        """
        text_dmg = bool(text_damage)
        if email_present:
            return text_dmg
        if photo_url:
            v = vision.has_damage(photo_url)
            if v is not None:
                return v or text_dmg
            return text_dmg
        return text_dmg

    def _best_supplier(self, candidates: list[str]) -> tuple[int, str, float]:
        """Match every candidate name and return the highest-confidence supplier.
        A translation often anglicises a garbled name closer to canonical
        (e.g. 'Edward Blythe-Scheinz' raw vs 'Edwards Lifesciences' translated)."""
        best = None
        for name in candidates:
            sid, sname, score = self.suppliers.match(name)
            if best is None or score > best[2]:
                best = (sid, sname, score)
        if best is None:
            sid, sname, score = self.suppliers.match("")
            return sid, sname, 0.0
        return best

    # ── routing ───────────────────────────────────────────────────────────────

    def _category_ramps(self, sig: Signals) -> list[str]:
        goods = sig.goods_type or "standard"
        unit  = sig.unit or "parcels"
        count = sig.parcel_count or 0

        if goods == "perishable":
            return ["R07"]                       # mandatory
        if goods == "oversized":
            return ["R05", "R06"]
        # standard
        if unit == "pallets":
            if count > 32:
                return ["R08"]
            return ["R03", "R04"]
        # parcels, standard
        return ["R01", "R02"]

    def _choose_ramp(self, candidates: list[str], ramp_status: list[dict]) -> str:
        status = {r["ramp"]: r for r in ramp_status}

        def is_free(ramp: str) -> bool:
            return status.get(ramp, {}).get("status") == "free"

        def queue(ramp: str) -> int:
            return status.get(ramp, {}).get("queue_length", 0)

        free = [r for r in candidates if is_free(r)]
        if free:
            return min(free, key=queue)
        # no free ramp in category → shortest queue in category (keeps +5 category)
        return min(candidates, key=queue)

    # ── main entry point ────────────────────────────────────────────────────────

    def decide(self, truck: dict) -> Decision:
        sig, photo_url, email_present, supplier_candidates = self._gather(truck)

        has_damage = self._decide_damage(email_present, photo_url, sig.has_damage)
        if photo_url:  # diagnostic only — compare against the (soon-patched) URL label
            oracle = self._url_oracle(photo_url)
            if oracle is not None and oracle != has_damage:
                log.info("damage mismatch vs url-oracle: decided=%s oracle=%s (%s)",
                         has_damage, oracle, photo_url.split("/")[-1])

        # supplier resolution — best match across all transcript-derived names
        sid, sname, score = self._best_supplier(supplier_candidates)
        log.info("Supplier candidates %s → #%s '%s' (score %.0f)",
                 supplier_candidates, sid, sname, score)

        unit  = sig.unit or "parcels"
        count = sig.parcel_count if sig.parcel_count is not None else 0

        log.info("Extracted: supplier_id=%s count=%s unit=%s goods=%s damage=%s",
                 sid, count, unit, sig.goods_type, has_damage)

        if has_damage:
            return Decision(
                endpoint="reject-truck",
                supplier_id=sid, supplier_name=sname,
                parcel_count=count, has_damage=True, unit=unit,
            )

        candidates = self._category_ramps(sig)
        ramp = self._choose_ramp(candidates, truck.get("ramp_status", []))
        log.info("Routing goods=%s unit=%s count=%s → candidates=%s → %s",
                 sig.goods_type, unit, count, candidates, ramp)

        return Decision(
            endpoint="assign-ramp",
            supplier_id=sid, supplier_name=sname,
            parcel_count=count, has_damage=False, unit=unit,
            assigned_ramp=ramp,
        )
