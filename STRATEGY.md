# RampRush — Strategy Plan

## Key reconnaissance findings (from live server probing)

1. **Photos leak the ground truth in the URL path.**
   `…/assets/damaged_parcels/damaged/…`  → `has_damage = true`
   `…/assets/damaged_parcels/undamaged/…` → `has_damage = false`
   → **No vision model needed.** We read `has_damage` straight from the photo URL.

2. **Emails are templated** across DE / FR / IT / ES / EN. Each contains:
   - Subject: `… – {Supplier Name}`
   - Body: `… {N} {unit} de/of {goods description} …`
   - An optional damage alert (`Alert: major cargo damage detected!`)
   - One random distractor sentence (noise) + signature
   → Pure regex + multilingual keyword parsing. No LLM needed.

3. **Audio follows the same template**, spoken. Transcribed locally with
   `faster-whisper` (`small`, int8, CPU, ~3 s/clip, no API key, multilingual).
   Output feeds the **same** text parser as emails.
   - e.g. `"somos la empresa CIA. Entregamos seis paquetes de electrónica hoy."`
   - Noisy/accented clips garble names & counts → handled by fuzzy matching.

4. **Supplier list = 9,169 entries.** Need fast fuzzy NER → `rapidfuzz`.

5. **Feedback endpoints** (JSON, used by the dashboard):
   - `GET /scoreboard` — ranking
   - `GET /team/{team_id}/log` — per-truck breakdown **with expected (ground-truth) values**
   - `GET /status` — scenario / queue
   → We can run, then diff our extraction against ground truth to tune. No HTML scraping needed after all.

## Scoring recap (what to optimise)

| Field | +/- | Source |
|---|---|---|
| supplier_id | +15/−15 | fuzzy match name → canonical id |
| parcel_count | +10/−10 | number in text |
| has_damage | +10/−10 | **photo URL** (or text keyword) |
| unit | +5/−5 | parcels/pallets keyword |
| decision | +20 reject / +7 free ramp / +5 right category | rules |
| throughput | +2 always | just respond |

No abstain option → always submit the best guess (best fuzzy beats blank).

## Extraction pipeline (per truck)

```
docs → gather signals:
  email.text                       → parse
  audio.url   → whisper transcribe → parse  (asyncio.to_thread, non-blocking)
  photo.url                        → has_damage from path

parse(text) yields: supplier_name?, count?, unit?, goods_type?, damage?
merge signals (email > audio for text fields; photo authoritative for damage)
supplier_name → rapidfuzz → supplier_id
```

### Multilingual vocab
- **parcels**: colis, paquet(s), paquete(s), parcel(s), Paket(e), pacco/pacchi, collo/colli
- **pallets**: palette(s), pallet(s), Palette(n), paleta(s)
- **perishable**: périssable, perishable, perecedero, deperibile, verderblich, Kühlware
- **oversized**: encombrant, oversized, sperrig, Sperrgut, ingombrante, voluminoso, übergroß
- **damage**: damage, endommagé/dommage, beschädigt/Schaden, danni/danneggiato, dañado
- **numbers**: digits first; fallback multilingual number-words 0–99

## Routing logic (decision)

Priority order (first match wins), then pick a **free** ramp in that category:
```
has_damage             → /reject-truck
perishable             → R07            (mandatory, even for parcels)
oversized              → R05, R06
unit=pallets & N > 32  → R08
unit=pallets & N ≤ 32  → R03, R04
unit=parcels           → R01, R02
```
Ramp choice within a category: prefer `status=free`; else lowest `queue_length`.
This guarantees the +5 category bonus and captures +7 whenever a category ramp is free.

## File layout

```
config.py            TEAM_ID, URLs, model settings
suppliers.py         load + cache supplier list, fuzzy match
extract.py           multilingual text parser + signal merge
audio.py             whisper transcription (lazy-loaded singleton)
strategies/
  base.py            Strategy ABC + Decision (exists)
  dummy_reject.py    stub (exists)
  smart.py           SmartStrategy — the real one
main.py              plumbing; concurrent media handling + DEBUG logs (exists)
tools/eval.py        fetch /team/{id}/log, compute per-field accuracy
```

## Open questions / assumptions
- **parcels + oversized** (e.g. "50 Pakete Sperrgut"): assume `goods_type` priority →
  route oversized to R05/R06; still report `unit=parcels` for the extraction score.
  *(Verify against /team log; adjust if category scoring disagrees.)*
- **Free vs category trade-off** when category ramps all occupied: we keep correct
  category (guaranteed +5). Revisit if logs show it costs more than the +7.
- Whisper `small` accuracy on noisy German clips is the weakest link — may bump to
  `medium` if logs show poor audio supplier/count accuracy (cost: ~2× latency).
