import os

TEAM_ID  = os.environ.get("TEAM_ID", "null-pointer")
API_BASE = "https://truckgenerator-production.up.railway.app"
WS_URL   = f"wss://truckgenerator-production.up.railway.app/ws?team_id={TEAM_ID}"

SUPPLIERS_URL = f"{API_BASE}/suppliers"

# Anthropic
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
ANTHROPIC_FALLBACK_MODEL = os.environ.get("ANTHROPIC_FALLBACK_MODEL", "claude-sonnet-4-6")
ANTHROPIC_MAX_TOKENS = int(os.environ.get("ANTHROPIC_MAX_TOKENS", "1024"))

# whisper.cpp (pywhispercpp)
#   tiny | base | small | medium | large-v3 | large-v3-turbo
# "medium" gives noticeably better multilingual transcription on noisy/garbled
# clips at the cost of a ~1.5 GB one-time download and ~2-3× transcription
# latency (which is fine — the server waits for our response anyway).
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
WHISPER_THREADS = int(os.environ.get("WHISPER_THREADS", "6"))

# Supplier fuzzy matching
SUPPLIER_MATCH_THRESHOLD = int(os.environ.get("SUPPLIER_MATCH_THRESHOLD", "72"))
