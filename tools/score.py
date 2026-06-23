#!/usr/bin/env python3
"""
Compute the canonical performance metric for a strategy from a run log.

Metric = average points/truck (the number we compare across strategies), plus
per-field accuracy for diagnosis. Reads the breakdown lines emitted by main.py.

Usage:  .venv/bin/python -m tools.score /tmp/val_run.log [N]
        N = score only the first N events (e.g. 100 for one full testmode cycle).
"""

import re
import sys
from collections import Counter

FIELDS = ["supplier_id", "parcel_count", "has_damage", "unit", "ramp_category"]
MAXES  = {"supplier_id": 15, "parcel_count": 10, "has_damage": 10, "unit": 5,
          "ramp_category": 5}


def score_log(path: str, limit: int | None = None) -> dict:
    correct = Counter()
    total   = Counter()
    points  = []
    # Track per-truck so a limit counts whole events, and field lines attach to
    # the right (kept) event.
    for line in open(path):
        m = re.search(r"Score for TRK-\d+: total=(-?\d+)", line)
        if m:
            # Break at the (N+1)-th event; the N-th event's field lines (which
            # precede this score line) have already been counted.
            if limit is not None and len(points) >= limit:
                break
            points.append(int(m.group(1)))
            continue
        for f in FIELDS:
            m = re.search(rf"\b{f}\s+(-?\d+)/(\d+)\b", line)
            if m:
                total[f] += 1
                if int(m.group(1)) == int(m.group(2)):
                    correct[f] += 1
    n = len(points)
    return {
        "trucks": n,
        "total_points": sum(points),
        "avg_per_truck": (sum(points) / n) if n else 0.0,
        "accuracy": {f: (correct[f], total[f]) for f in FIELDS},
    }


def main() -> None:
    path  = sys.argv[1] if len(sys.argv) > 1 else "/tmp/val_run.log"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    r = score_log(path, limit)
    print(f"=== {path} ===")
    print(f"trucks:        {r['trucks']}")
    print(f"total points:  {r['total_points']}")
    print(f"AVG/TRUCK:     {r['avg_per_truck']:.2f}   <-- comparison metric")
    print("accuracy:")
    for f, (c, t) in r["accuracy"].items():
        pct = 100 * c / t if t else 0
        print(f"  {f:<14} {c:>3}/{t:<3} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
