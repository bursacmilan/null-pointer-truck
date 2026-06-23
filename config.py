TEAM_ID  = "null-pointer"
API_BASE = "https://truckgenerator-production.up.railway.app"
WS_URL   = f"wss://truckgenerator-production.up.railway.app/ws?team_id={TEAM_ID}"

# Whisper (faster-whisper) settings — local STT, no API key required.
WHISPER_MODEL     = "small"   # base | small | medium ; small ≈ 3s/clip on CPU
WHISPER_DEVICE    = "cpu"
WHISPER_COMPUTE   = "int8"
WHISPER_BEAM_SIZE = 5
