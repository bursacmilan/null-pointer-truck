#!/usr/bin/env python3
"""
RampRush agent — WebSocket plumbing + pluggable strategy.
"""

import asyncio
import json
import logging
import os
import urllib.request

import websockets

from config import API_BASE, TEAM_ID, WS_URL
from strategies import DummyRejectStrategy, LLMStrategy, Strategy
from strategies.base import Decision

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
# Silence the noisy HTTP / SDK logs — they otherwise dump base64 image payloads.
for noisy in ("httpx", "httpcore", "anthropic", "anthropic._base_client",
              "urllib3", "websockets.client", "websockets.protocol", "asyncio"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log = logging.getLogger("agent")


# ── HTTP ──────────────────────────────────────────────────────────────────────

def post_json(path: str, payload: dict) -> dict:
    log.debug("POST /%s  payload=%s", path, json.dumps(payload))
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        f"{API_BASE}/{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req) as r:
        body = json.loads(r.read())
    log.debug("Response: %s", json.dumps(body))
    return body


def build_payload(truck_id: str, d: Decision) -> dict:
    payload = {
        "truck_id":      truck_id,
        "team_id":       TEAM_ID,
        "supplier_id":   d.supplier_id,
        "supplier_name": d.supplier_name,
        "parcel_count":  d.parcel_count,
        "has_damage":    d.has_damage,
        "unit":          d.unit,
    }
    if d.assigned_ramp is not None:
        payload["assigned_ramp"] = d.assigned_ramp
    return payload


# ── Agent loop ────────────────────────────────────────────────────────────────

async def run(strategy: Strategy) -> None:
    log.info("Connecting to %s", WS_URL)

    async with websockets.connect(WS_URL) as ws:
        log.info("WebSocket connected. Team: %s  Strategy: %s",
                 TEAM_ID, type(strategy).__name__)

        async for raw in ws:
            log.debug("Raw WS message: %s", raw[:300])

            truck = json.loads(raw)

            if "truck_id" not in truck:
                log.info("Non-truck message received (end-of-round summary?): %s", truck)
                continue

            truck_id = truck["truck_id"]
            priority = truck.get("priority", "normal")
            docs     = [d["type"] for d in truck.get("documentation", [])]
            ramps    = {r["ramp"]: r["status"] for r in truck.get("ramp_status", [])}

            log.info("TRUCK %s  priority=%s  docs=%s", truck_id, priority, docs)
            log.debug("Ramp status: %s", ramps)

            try:
                decision = await strategy.decide(truck)
                log.info("Decision → endpoint=%s  supplier_id=%s  parcel_count=%s  "
                         "has_damage=%s  unit=%s  ramp=%s",
                         decision.endpoint, decision.supplier_id, decision.parcel_count,
                         decision.has_damage, decision.unit, decision.assigned_ramp)
            except Exception:
                log.exception("Strategy raised an exception for truck %s — skipping", truck_id)
                continue

            payload  = build_payload(truck_id, decision)
            response = post_json(decision.endpoint, payload)

            total = response.get("total", "?")
            log.info("Score for %s: total=%s  extraction=%s  decision=%s  throughput=%s",
                     truck_id,
                     total,
                     response.get("extraction_score"),
                     response.get("decision_score"),
                     response.get("throughput_bonus"))

            breakdown = response.get("breakdown", {})
            for field, info in breakdown.items():
                earned = info.get("earned", "?")
                maxi   = info.get("max", "?")
                result = info.get("result", "")
                log.debug("  %-14s %s/%s  %s", field, earned, maxi, result)


async def _main_async() -> None:
    strategy: Strategy
    if os.environ.get("STRATEGY", "llm").lower() == "dummy":
        strategy = DummyRejectStrategy()
    else:
        strategy = await LLMStrategy.bootstrap()
    log.info("Starting agent with strategy: %s", type(strategy).__name__)
    await run(strategy)


def main() -> None:
    try:
        asyncio.run(_main_async())
    except KeyboardInterrupt:
        log.info("Interrupted by user.")


if __name__ == "__main__":
    main()
