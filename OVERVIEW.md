# RampRush Agent — How It Works

A client agent that connects to the RampRush event server, parses noisy
multilingual/multimodal truck documentation, extracts structured data, and
assigns each truck to the right unloading ramp in real time.

---

## The problem

Each truck arrives with messy documentation in **any combination** of:
- 📧 **Email** — free text, DE/FR/IT/ES/EN, typos, abbreviations, + a random
  distractor sentence designed to mislead.
- 🎙️ **Audio** — a spoken message, accented and noisy.
- 📷 **Photo** — a parcel image that may show transport damage.

From these we must extract **5 fields** and make **1 decision**:

| Field | Source | Scoring |
|---|---|---|
| `supplier_id` | name → canonical ID (9,169 suppliers) | +15 / −15 |
| `parcel_count` | number in text | +10 / −10 |
| `has_damage` | photo / text alert | +10 / −10 |
| `unit` (parcels/pallets) | keyword | +5 / −5 |
| ramp assignment | rules over the above | +20 reject / +12 correct |

---

## Pipeline (per truck)

```
        ┌─ email  → regex parser ─────────────────────────────┐
WS msg ─┼─ audio  → Whisper STT (raw + EN translation)         │→ merge signals
        │            → regex parser → local-LLM augmentation    │      │
        └─ photo  → local Vision model (damage)                 ┘      ▼
                                                       supplier match (exact→fuzzy+phonetic)
                                                                       │
                                                                       ▼
                                                        ramp rules → POST assign / reject
```

The server gates the next truck on our response, so we process one fully, reply,
and repeat. All models run **locally** — no cloud API key required.

---

## Key components & the clever bits

**1. Email parsing — regex, not LLM (≈100% accurate).**
Emails are templated. We pull the supplier from the subject/intro phrases, the
count from the digit next to the unit word, and the goods type from keywords —
all scoped to the *delivery sentence* so the random distractor ("…my car was
broken again…") can't trigger false matches.

**2. Damage detection — image-first, no exploit.**
The photo URL used to leak the answer (`/damaged/` vs `/undamaged/`); that's
being patched, so we **don't use it**. Instead:
- email trucks → trust the text alert (urgency markers like *Alert/Kritisch/
  Attenzione* — these never appear in normal mail or distractors);
- audio+photo / photo-only → a **local vision model** (llava) with a deliberately
  conservative prompt (a false "damaged" wrongly rejects a good truck, which is
  costly). Live damage accuracy ≈ 98%.

**3. Audio — transcribe twice, then LLM.**
Whisper runs a raw pass (best for the supplier name) **and** an English
translation pass (clean "parcels/pallets/oversized/perishable" vocabulary that
recovers fields foreign/garbled transcripts miss). A local LLM then re-extracts
from the messy transcript as a robustness layer. ~25% of audio clips are
deliberately destroyed (the model hallucinates) — an inherent floor.

**4. Supplier resolution — exact, then lexical + phonetic.**
Documentation usually carries the *exact* canonical name → exact match first
(emails ≈100%). For garbled audio names we blend **lexical** similarity
(rapidfuzz WRatio) with **phonetic** similarity (metaphone/jellyfish), because
TTS+speech-to-text errors are phonetic: *Carvana→"Karane"*, *Columbia→"Klambia"*.
We try every transcript variant and keep the highest-confidence match.

**5. Ramp routing — spec-correct rules.**
Priority: perishable→R07 (mandatory) ▸ oversized→R05/R06 ▸ >32 pallets→R08 ▸
parcels→R01/R02 ▸ standard pallets ≤32→R03/R04 (**fallback R05/R06/R07**, which
also accept normal pallets). Prefer a *free* ramp (earns +7) before an occupied
one; this maximises both the category bonus and the free-lane bonus.

---

## Model-agnostic by design

Every model is swappable via environment variables — no code changes:

| Component | Default (local) | Override |
|---|---|---|
| Speech-to-text | faster-whisper `small` | `WHISPER_MODEL` |
| Audio LLM | ollama `llama3.1` | `LLM_MODEL` |
| Vision (damage) | ollama `llava:7b` | `VISION_MODEL` |

`USE_LLM=0` / `USE_VISION=0` toggle components off for A/B testing. Pointing at a
cloud provider (e.g. Claude) only touches `llm.py` / `vision.py`.

---

## Measuring performance

`tools/score.py <run.log> 100` reports avg points/truck + per-field accuracy.
**Important:** the server **regenerates truck content per connection** (the
TRK-IDs cycle within a run but differ across runs), so scores are compared as
*averages over a full 100-truck cycle*, never truck-by-truck across runs. See
`SCORES.md` for the log and `STRATEGY.md` for the full reconnaissance notes.
