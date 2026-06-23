"""
Deterministic ramp selector. No LLM — pure rules + queue minimization.

Priority order (most specific first):
  1. perishable                → R07           (must, regardless of unit)
  2. unit=pallets, count > 32  → R08           (only ramp for double trucks)
  3. goods_type=oversized      → R05 / R06     (heavy lanes)
  4. unit=parcels              → R01 / R02
  5. unit=pallets, count ≤ 32  → R03 / R04 primary, R05/R06/R07 fallback

Selection within the candidate list:
  - prefer "free" ramps; among those, the one with the shortest queue
  - if all primaries busy and a secondary is free → take the free secondary
    (+7 free beats +5 category-bonus alone)
  - if everything busy → shortest queue among primaries
    (keeps the +5 category bonus and earns +7 for "no free alternative")
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def _candidates(unit: str, count: int, goods_type: str) -> tuple[list[str], list[str]]:
    # Perishable always wins — cold chain has nowhere else to go.
    if goods_type == "perishable":
        return ["R07"], []
    # Oversized wins over count: heavy lanes (R05/R06) are primary for it
    # regardless of pallet count, per spec ("Primär für übergroße Güter").
    if goods_type == "oversized":
        return ["R05", "R06"], []
    # Non-oversized double truck → only R08 accepts >32 pallets.
    if unit == "pallets" and count > 32:
        return ["R08"], []
    if unit == "parcels":
        return ["R01", "R02"], []
    # standard pallets ≤ 32
    return ["R03", "R04"], ["R05", "R06", "R07"]


def select_ramp(unit: str, count: int, goods_type: str, ramp_status: list[dict]) -> str:
    status = {r["ramp"]: (r.get("status", "occupied"), r.get("queue_length", 99))
              for r in ramp_status}

    def free_sorted(cands: list[str]) -> list[str]:
        return sorted(
            [c for c in cands if status.get(c, ("occupied", 99))[0] == "free"],
            key=lambda c: status[c][1],
        )

    def shortest_queue(cands: list[str]) -> str:
        return min(cands, key=lambda c: status.get(c, ("occupied", 99))[1])

    primary, secondary = _candidates(unit, count, goods_type)
    log.debug("Ramp candidates for unit=%s count=%d goods=%s → primary=%s secondary=%s",
              unit, count, goods_type, primary, secondary)

    free_primary = free_sorted(primary)
    if free_primary:
        return free_primary[0]

    free_secondary = free_sorted(secondary)
    if free_secondary:
        log.debug("Primary all busy, falling back to free secondary %s", free_secondary[0])
        return free_secondary[0]

    chosen = shortest_queue(primary)
    log.debug("All candidates busy — picking shortest-queue primary %s", chosen)
    return chosen
