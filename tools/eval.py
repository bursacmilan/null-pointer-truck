#!/usr/bin/env python3
"""
Fetch /team/{TEAM_ID}/log and report per-field extraction accuracy + decision
quality against the server's ground truth. Use this to tune the strategy.

Usage:  .venv/bin/python -m tools.eval [N]      # last N trucks (default: all)
"""

import json
import sys
import urllib.request
from collections import Counter

sys.path.insert(0, ".")
from config import API_BASE, TEAM_ID

FIELDS = ["supplier_id", "parcel_count", "has_damage", "unit"]


def fetch_log() -> list[dict]:
    url = f"{API_BASE}/team/{TEAM_ID}/log"
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read()).get("log", [])


def main() -> None:
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    log = fetch_log()
    if limit:
        log = log[:limit]
    if not log:
        print("No log entries for team", TEAM_ID)
        return

    n = len(log)
    correct = Counter()
    wrong_samples: dict[str, list[str]] = {f: [] for f in FIELDS}
    total_points = 0
    decision_correct = 0
    decision_points = 0

    for entry in log:
        total_points += entry.get("points", 0)
        bd = entry.get("breakdown", {})
        for f in FIELDS:
            info = bd.get(f, {})
            if info.get("earned") == info.get("max"):
                correct[f] += 1
            else:
                if len(wrong_samples[f]) < 8:
                    wrong_samples[f].append(f"{entry['truck_id']}: {info.get('result','')}")
        dinfo = bd.get("decision", {})
        decision_points += dinfo.get("earned", 0)
        if dinfo.get("earned", 0) > 0:
            decision_correct += 1

    print(f"=== Team {TEAM_ID} — {n} trucks ===")
    print(f"Total points: {total_points}   (avg {total_points/n:.1f}/truck)\n")
    print("Extraction accuracy:")
    for f in FIELDS:
        pct = 100 * correct[f] / n
        print(f"  {f:<14} {correct[f]:>3}/{n}  ({pct:5.1f}%)")
    print(f"\nDecision: {decision_correct}/{n} positive  "
          f"(avg {decision_points/n:.1f} pts/truck)\n")

    for f in FIELDS:
        if wrong_samples[f]:
            print(f"--- {f} misses ---")
            for s in wrong_samples[f]:
                print("   ", s)


if __name__ == "__main__":
    main()
