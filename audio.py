"""Local speech-to-text via faster-whisper (no API key, multilingual, CPU).

Supplier audio is templated in DE/FR/IT/ES/EN. On clean clips a single raw pass
is best. Heavily-noised clips make whisper misdetect the language (Arabic /
Romanian / Javanese) and emit garbage; for those we do ONE fallback pass on
loudness-normalised audio. Blanket preprocessing is avoided because it degrades
clean speech (e.g. turns "seis"=6 into "el paquete"=1)."""

import logging
import subprocess
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

# Languages the documentation is actually spoken in. Anything else from whisper
# is a misdetection on noise → trigger the preprocessing fallback.
EXPECTED_LANGS = {"de", "fr", "it", "es", "en"}

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


def _transcribe_file(path: str, language: str | None = None,
                     task: str = "transcribe") -> tuple[str, str]:
    """Run whisper on a local file → (detected_language, text).
    task='translate' emits an English translation regardless of source language."""
    model = _get_model()
    segments, info = model.transcribe(
        path, beam_size=WHISPER_BEAM_SIZE, language=language, task=task
    )
    text = " ".join(s.text for s in segments).strip()
    return info.language, text


def _loudnorm(src_path: str) -> str | None:
    """Loudness-normalise to 16 kHz mono WAV (gentle — no band-pass/denoise that
    would distort clean speech). Returns the new path, or None if ffmpeg fails."""
    dst = src_path + ".norm.wav"
    proc = subprocess.run(
        ["ffmpeg", "-y", "-i", src_path,
         "-af", "loudnorm=I=-16:TP=-1.5", "-ar", "16000", "-ac", "1", dst],
        capture_output=True,
    )
    return dst if proc.returncode == 0 else None


def transcribe_candidates(url: str) -> list[str]:
    """Download an audio URL and return one or more transcript candidates to parse.

    Always returns the raw transcript (keeps the supplier name closest to its
    canonical spelling). For non-English clips it also returns an English
    translation, whose normalised vocabulary ("parcels", "oversized",
    "perishable", …) reliably recovers unit/goods that foreign or garbled raw
    transcripts miss. Blocking — run in a thread."""
    url = urljoin(API_BASE + "/", url)   # server sends relative paths like /assets/...
    log.debug("Downloading audio: %s", url)
    with urllib.request.urlopen(url) as r:
        audio_bytes = r.read()

    candidates: list[str] = []
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
        f.write(audio_bytes)
        f.flush()

        lang, raw = _transcribe_file(f.name)
        log.info("Transcribed (%s): %s", lang, raw)
        if raw:
            candidates.append(raw)

        # Misdetected language (noise) → retry raw on loudness-normalised audio.
        if lang not in EXPECTED_LANGS:
            norm = _loudnorm(f.name)
            if norm:
                lang2, raw2 = _transcribe_file(norm)
                log.info("Fallback transcribe (%s): %s", lang2, raw2)
                if raw2:
                    candidates.append(raw2)
                    lang = lang2 if lang2 in EXPECTED_LANGS else lang

        # English translation pass for any non-English clip — robust unit/goods.
        if lang != "en":
            _, trans = _transcribe_file(f.name, task="translate")
            log.info("Translated (en): %s", trans)
            if trans:
                candidates.append(trans)

    return candidates or [""]


def transcribe_url(url: str) -> str:
    """Backwards-compatible single-string transcript (first candidate)."""
    return transcribe_candidates(url)[0]


def warm_up() -> None:
    """Load the model ahead of the first truck so the first response isn't slow."""
    _get_model()
