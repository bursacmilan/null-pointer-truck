# Architecture — moving parts

All models are **local and model-agnostic**: swap any of them via environment
variables (see `config.py`) without touching the pipeline. Cloud models would
only require new clients in `llm.py` / `vision.py`.

```
WebSocket truck ─┐
                 │  documentation:
                 ├─ email   → extract.parse_text        (multilingual regex, noise-hardened)
                 ├─ audio   → audio.transcribe_candidates(raw + EN translate, faster-whisper)
                 │              → extract.parse_text
                 │              → llm.extract           (ollama LLM, fills regex gaps)   [USE_LLM]
                 └─ photo   → vision.has_damage          (ollama VLM)                     [USE_VISION]

  merge signals → suppliers.match (exact → blended lexical+phonetic) → route → POST
```

## Components & how to swap models

| Part | File | Default model | Env override |
|---|---|---|---|
| Speech-to-text | `audio.py` | faster-whisper `small` | `WHISPER_MODEL` (e.g. `medium`, `large-v3`) |
| Text extraction | `extract.py` | regex (no model) | — |
| Audio LLM augment | `llm.py` | ollama `llama3.1` | `LLM_MODEL` (e.g. `qwen2.5`, `mistral-small`) |
| Photo damage | `vision.py` | ollama `llava:7b` | `VISION_MODEL` (e.g. `llama3.2-vision`, `qwen2-vl`) |
| Supplier NER | `suppliers.py` | exact + WRatio + metaphone | — |
| ollama endpoint | — | `localhost:11434` | `OLLAMA_HOST` |

Toggle whole components off for A/B testing: `USE_LLM=0`, `USE_VISION=0`.
A stronger vision model is the most impactful upgrade (see SCORES.md / damage).

## Key design decisions

- **Damage without the URL exploit.** The `…/damaged/` vs `…/undamaged/` URL path
  is being patched, so it is NOT used in the decision (only logged as an oracle
  for diagnostics). Instead: email trucks → text alert (reliable, avoids vision
  false-positives on intact boxes); audio+photo / photo-only → vision model.
- **Supplier matching** is exact-first (emails carry the canonical name verbatim,
  ~100%), then a blended lexical (WRatio) + phonetic (metaphone) fuzzy fallback
  for garbled audio names (TTS+whisper errors are phonetic: Carvana→"Karane").
- **Audio** runs raw + English-translation passes; the LLM adds a robustness
  layer for messy phrasing. The hardest ~25% of audio clips are deliberately
  destroyed (whisper hallucinations) — an inherent floor.

See `STRATEGY.md` for the full reconnaissance and `SCORES.md` for the score log.
