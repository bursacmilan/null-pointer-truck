#!/usr/bin/env python3
"""Compute extraction accuracy from a run's logged response breakdowns."""
import re, sys
from collections import Counter
path = sys.argv[1]
FIELDS=["supplier_id","parcel_count","has_damage","unit"]
correct=Counter(); total=Counter(); misses={f:[] for f in FIELDS}
cur=None
for line in open(path):
    m=re.search(r"TRUCK (TRK-\d+)",line)
    if m: cur=m.group(1)
    # breakdown debug:  "  supplier_id    -15/15  <result>"
    m=re.search(r"\b(supplier_id|parcel_count|has_damage|unit)\s+(-?\d+)/(\d+)\s+(.*)",line)
    if m:
        f,earned,mx,res=m.group(1),int(m.group(2)),int(m.group(3)),m.group(4)
        total[f]+=1
        if earned==mx: correct[f]+=1
        elif len(misses[f])<12: misses[f].append(f"{cur}: {res.strip()}")
n=max(total.values()) if total else 0
print(f"trucks with breakdown: {n}")
for f in FIELDS:
    if total[f]: print(f"  {f:<14} {correct[f]:>3}/{total[f]}  ({100*correct[f]/total[f]:5.1f}%)")
for f in FIELDS:
    if misses[f]:
        print(f"--- {f} misses ---")
        for s in misses[f]: print("   ",s)
