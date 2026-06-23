import os

TEAM_ID  = "null-pointer"
API_BASE = "https://truckgenerator-production.up.railway.app"
WS_URL   = f"wss://truckgenerator-production.up.railway.app/ws?team_id={TEAM_ID}"

# ── Models ────────────────────────────────────────────────────────────────────
# Everything is local-by-default and MODEL-AGNOSTIC: override any model via an
# environment variable to plug in a stronger one without touching code.
#   WHISPER_MODEL   — faster-whisper STT  (base|small|medium|large-v3, or a path)
#   LLM_MODEL       — ollama text model   (e.g. llama3.1, qwen2.5, mistral-small)
#   VISION_MODEL    — ollama vision model (e.g. llava:7b, llama3.2-vision, qwen2-vl)
#   OLLAMA_HOST     — ollama endpoint (default http://localhost:11434)
# To point at a cloud provider instead, only llm.py / vision.py need new clients;
# the rest of the pipeline is unaffected.

# Whisper (faster-whisper) — local STT, no API key required.
WHISPER_MODEL     = os.environ.get("WHISPER_MODEL", "small")  # small ≈ 3s/clip CPU
WHISPER_DEVICE    = os.environ.get("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE   = os.environ.get("WHISPER_COMPUTE", "int8")
WHISPER_BEAM_SIZE = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))

# ollama-served local LLM / VLM (no API key). Swap freely via env.
OLLAMA_HOST  = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
LLM_MODEL    = os.environ.get("LLM_MODEL", "llama3.1:latest")
VISION_MODEL = os.environ.get("VISION_MODEL", "llava:7b")

# Feature toggles (set to "0" to disable) — lets a teammate isolate components.
USE_LLM    = os.environ.get("USE_LLM", "1") != "0"
USE_VISION = os.environ.get("USE_VISION", "1") != "0"
