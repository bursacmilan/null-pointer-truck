"""Local speech-to-text via faster-whisper (no API key, multilingual, CPU)."""

import logging
import tempfile
import urllib.request
from urllib.parse import urljoin

from config import (
    API_BASE,
    WHISPER_BEAM_SIZE,
    WHISPER_COMPUTE,
    WHISPER_DEVICE,
    WHISPER_MODEL,
)

log = logging.getLogger("audio")

_model = None


def _get_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel  # lazy import (heavy)
        log.info("Loading whisper model '%s' (%s/%s)...",
                 WHISPER_MODEL, WHISPER_DEVICE, WHISPER_COMPUTE)
        _model = WhisperModel(
            WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
        )
        log.info("Whisper model loaded.")
    return _model


def transcribe_url(url: str) -> str:
    """Download an audio URL and return the transcript text (blocking — run in a thread)."""
    url = urljoin(API_BASE + "/", url)   # server sends relative paths like /assets/...
    log.debug("Downloading audio: %s", url)
    with urllib.request.urlopen(url) as r:
        audio_bytes = r.read()

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        f.write(audio_bytes)
        f.flush()
        model = _get_model()
        segments, info = model.transcribe(f.name, beam_size=WHISPER_BEAM_SIZE)
        text = " ".join(s.text for s in segments).strip()

    log.info("Transcribed (%s, p=%.2f): %s",
             info.language, info.language_probability, text)
    return text


def warm_up() -> None:
    """Load the model ahead of the first truck so the first response isn't slow."""
    _get_model()
