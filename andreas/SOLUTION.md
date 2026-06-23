# Andreas — RampRush Solution

Team: `nulltruckpoint-ak`

## How it works

Each truck arrives via WebSocket with up to three documents: a photo, an email, and an audio clip.

### 1. Extraction (Gemini via Vertex AI)
All documents are sent to Gemini in a single multimodal call:
- **Photo** → raw JPEG bytes (vision)
- **Email** → plain text
- **Audio** → raw MP3 bytes (Gemini transcribes internally)

Gemini returns structured JSON: `raw_supplier_name`, `parcel_count`, `unit`, `goods_type`, `has_damage`.

### 2. Damage detection (hybrid)
- **Email text** → Seb's proven regex from `extract.py`: exact multilingual keyword matching (urgency markers like `Alert`, `Kritisch`, `Attenzione` + damage terms in DE/FR/IT/ES/EN). Zero ambiguity, never misses a structured alert line.
- **Photo / audio** → Gemini's `has_damage` output from vision/audio understanding.
- Result: `true` if **either** source signals damage → truck is rejected.

### 3. Supplier matching (4-tier pipeline)
The raw name Gemini extracts is matched against 9169 suppliers using Milan's normalization approach:
- Umlaut folding (`ü→u`, `ö→o`, `ä→a`, `ß→ss`, accents)
- Noise token removal (`logistics`, `transport`, `freight`, articles, etc.)
- Corporate suffixes kept (GmbH, AG, Inc → disambiguate entities)

**Tier 0** — Exact normalized match (dict lookup)  
**Tier 1** — `min(token_sort_ratio, WRatio) ≥ 72` — conservative fuzzy (both scorers must agree)  
**Tier 2** — Jaro-Winkler on Metaphone phonetic codes ≥ 70 — catches audio variants (Müller/Muller/Mueller)  
**Tier 3** — Token overlap ≥ 2 — catches partial audio names  
**Tier 4** — Best fuzzy regardless of threshold + warning log  

### 4. Ramp routing
| Condition | Ramp(s) |
|---|---|
| Perishable goods | R07 → R01/R02 fallback |
| Parcels | R01/R02 → others |
| Pallets > 32 | R08 → R05/R06 → R03/R04 |
| Oversized | R05/R06 → R03/R04 |
| Standard pallets ≤ 32 | R03/R04 → R05/R06 |

Within each group: prefer free ramps, then shortest queue.

## Running

```bash
conda run -n rampRush --no-capture-output \
  GEMINI_MODEL=gemini-2.5-pro \
  TEAM_ID=nulltruckpoint-ak \
  python andreas/run.py
```
