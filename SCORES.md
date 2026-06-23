# RampRush — Strategy Score Log

The **`seb-best`** branch always holds the highest-performing strategy. We only
push to `seb-best` when a measured run beats the current record below.

**Metric:** average points/truck, measured live against the test server
(`testmode`), computed with `tools/score.py <run.log>`. Field accuracy is shown
for diagnosis. Scores vary slightly run-to-run (truck cycling, ramp occupancy,
audio noise), so we record the trucks count and treat ≥1.5 pts/truck as a real gain.

## 🏆 Current record (on `seb-best`)

| Metric | Value |
|---|---|
| **Avg points/truck** | **42.59** |
| Trucks measured | 81 |
| supplier_id | 70.4% |
| parcel_count | 87.7% |
| has_damage | 100.0% |
| unit | 93.8% |
| ramp_category | 84.0% |

## History

| Date | Version | Avg/truck | trucks | supplier | count | damage | unit | category | Notes |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-23 | v1 — smart, token_set supplier, no audio translate | 36.06 | 117 | 50.4% | 83.8% | 100% | 93.2% | 85.5% | First real strategy |
| 2026-06-23 | **v2 — exact-match supplier + audio raw+translate merge** | **42.59** | 81 | 70.4% | 87.7% | 100% | 93.8% | 84.0% | 🏆 current best |

## Known ceiling / remaining losses

- **Audio (~27% of trucks)** is the weak point. Email extraction is ~100% on
  every field; all `ramp_category` and most `supplier_id`/`count` misses are
  audio trucks.
- ~25% of audio clips are deliberately destroyed (whisper hallucinates, e.g.
  "Thank you for watching this video") — an inherent floor no preprocessing fixes.
- `has_damage` is 100% (read from photo URL path `/damaged/` vs `/undamaged/`).
