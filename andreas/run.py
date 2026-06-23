#!/usr/bin/env python3
"""
Andreas entry point — team nulltruckpoint-ak using ClaudeRampStrategy.

Run from repo root:
    conda activate rampRush
    export ANTHROPIC_API_KEY=sk-...
    python andreas/run.py
"""

import asyncio
import logging
import os
import sys

# Ensure repo root is on path regardless of where this script is called from
_HERE   = os.path.dirname(os.path.abspath(__file__))
_ROOT   = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Override team identity BEFORE importing main (which copies these at import time) ──
TEAM_ID = "nulltruckpoint-ak"
WS_URL  = f"wss://truckgenerator-production.up.railway.app/ws?team_id={TEAM_ID}"

import config
config.TEAM_ID = TEAM_ID
config.WS_URL  = WS_URL

import main as _main
_main.TEAM_ID = TEAM_ID
_main.WS_URL  = WS_URL

from strategy import ClaudeRampStrategy

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%H:%M:%S",
)
log = logging.getLogger("andreas")

if __name__ == "__main__":
    import time
    log.info("Starting Andreas agent — team_id=%s", TEAM_ID)
    strategy = ClaudeRampStrategy()
    while True:
        try:
            asyncio.run(_main.run(strategy))
            log.info("Session ended — reconnecting in 3s...")
            time.sleep(3)
        except KeyboardInterrupt:
            log.info("Stopped.")
            break
        except Exception as exc:
            log.warning("Connection lost (%s) — reconnecting in 3s...", exc)
            time.sleep(3)
