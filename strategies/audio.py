"""
Audio transcription via whisper.cpp (pywhispercpp).

The model is loaded once and reused across trucks. Since the server only sends
the next truck after our response, transcribe calls are inherently serial — so
we don't need a lock around the underlying whisper context.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

import httpx

from config import WHISPER_MODEL, WHISPER_THREADS
from ._urls import normalize_url

log = logging.getLogger(__name__)


class WhisperTranscriber:
    def __init__(self, model_size: str = WHISPER_MODEL, n_threads: int = WHISPER_THREADS):
        self.model_size = model_size
        self.n_threads = n_threads
        self._model = None

    async def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        log.info("Loading whisper.cpp model %r (n_threads=%d) — first run downloads ggml weights",
                 self.model_size, self.n_threads)
        from pywhispercpp.model import Model
        self._model = await asyncio.to_thread(
            Model, self.model_size, n_threads=self.n_threads, print_progress=False,
        )
        log.info("whisper.cpp model loaded.")

    async def transcribe_url(self, url: str) -> str:
        await self.ensure_loaded()

        normalized = normalize_url(url) or url
        log.debug("Audio download URL: raw=%r normalized=%r", url, normalized)
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(normalized)
            r.raise_for_status()
            audio_bytes = r.content
        log.debug("Audio downloaded: %d bytes from %s", len(audio_bytes), normalized)

        suffix = os.path.splitext(normalized.split("?")[0])[1] or ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(audio_bytes)
            raw_path = f.name

        cleaned_path = await _preprocess_audio(raw_path)
        path_for_whisper = cleaned_path or raw_path

        try:
            segments = await asyncio.to_thread(
                self._model.transcribe,
                path_for_whisper,
                language="auto",
            )
            text = " ".join(s.text for s in segments).strip()
            log.debug("Transcript: %s", text)
            return text
        finally:
            for p in (raw_path, cleaned_path):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass


async def _preprocess_audio(src_path: str) -> str | None:
    """
    Run the raw download through ffmpeg to produce a 16 kHz mono WAV that's
    been highpass-filtered, denoised, and loudness-normalized. This usually
    yields a noticeably better whisper transcript on accented/noisy clips.
    Returns the cleaned WAV path on success, None to fall back to the raw file.
    """
    dst_path = src_path + ".clean.wav"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", src_path,
        "-af",
        # 80 Hz highpass kills HVAC rumble; 8 kHz lowpass cuts hiss above the
        # speech band; afftdn does spectral denoise; loudnorm to a consistent
        # target so whisper isn't seeing wildly different gain per clip.
        "highpass=f=80,lowpass=f=8000,afftdn=nf=-25,"
        "loudnorm=I=-16:LRA=11:TP=-1.5",
        "-ar", "16000", "-ac", "1",
        dst_path,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        rc = await proc.wait()
    except FileNotFoundError:
        log.warning("ffmpeg not on PATH — skipping audio preprocessing")
        return None

    if rc != 0 or not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
        log.debug("ffmpeg preprocessing failed (rc=%s) — falling back to raw audio", rc)
        try:
            os.unlink(dst_path)
        except OSError:
            pass
        return None
    return dst_path
