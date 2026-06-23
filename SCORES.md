# RampRush — Strategy Score Log

The **`seb-best`** branch always holds the highest-performing strategy. We only
push to `seb-best` when a measured run beats the current record below.

**Two numbers, don't confuse them:**
- **Total points** (e.g. ~3450) = cumulative leaderboard score. Grows with the
  number of trucks processed, so it is NOT comparable across runs of different
  length. This is what the live dashboard shows.
- **Avg points/truck** (e.g. 42.59) = total ÷ trucks. This IS the fair
  comparison metric across strategies and the one that gates `seb-best`.

Measured live against the test server (`testmode`), computed with
`tools/score.py <run.log>`. Field accuracy shown for diagnosis. Scores vary
run-to-run (truck cycling, ramp occupancy, audio noise) — treat ≥1.5 pts/truck
as a real gain.

## 🏆 Current record (on `seb-best`)

| Metric | Value |
|---|---|
| **Avg points/truck** | **43.44** |
| Total points | 3779 |
| Trucks measured | 87 |
| supplier_id | 73.6% |
| parcel_count | 87.4% |
| has_damage | 100.0% |
| unit | 94.3% |
| ramp_category | 83.9% |

## History

| Date | Version | Avg/truck | trucks | supplier | count | damage | unit | category | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-23 | v1 — smart, token_set supplier, no audio translate | 36.06 | 117 | 50.4% | 83.8% | 100% | 93.2% | 85.5% | First real strategy |
| 2026-06-23 | v2 — exact-match supplier + audio raw+translate merge | 42.59 | 81 | 70.4% | 87.7% | 100% | 93.8% | 84.0% | Big supplier jump |
| 2026-06-23 | **v3 — multi-candidate supplier (match raw + translated names, keep best)** | **43.44** | 87 | 73.6% | 87.4% | 100% | 94.3% | 83.9% | 🏆 current best; translated names recover anglicised garbled audio names (e.g. Edwards Lifesciences) |

## Known ceiling / remaining losses

- **Audio (~27% of trucks)** is the weak point. Email extraction is ~100% on
  every field; all `ramp_category` and most `supplier_id`/`count` misses are
  audio trucks.
- ~25% of audio clips are deliberately destroyed (whisper hallucinates, e.g.
  "Thank you for watching this video") — an inherent floor no preprocessing fixes.
- `has_damage` is 100% (read from photo URL path `/damaged/` vs `/undamaged/`).
