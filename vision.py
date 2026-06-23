"""Local vision model (via ollama) for parcel-damage detection from the photo.

This replaces the photo-URL heuristic (`…/damaged/` vs `…/undamaged/`), which is
an exploit that will be patched. Damage must be read from the image content.
Degrades to None (unknown) if the model/server is unavailable, so callers can
fall back to text-based damage signals.
"""

import base64
import json
import logging
import urllib.request
from urllib.parse import urljoin

from config import API_BASE, OLLAMA_HOST, USE_VISION, VISION_MODEL

log = logging.getLogger("vision")

TIMEOUT = 60

# Conservative prompt — parcels are usually fine, so we only flag CLEAR severe
# damage. This keeps the costly false-positive rate down (a wrongly-flagged
# intact truck loses both the assignment and the (wrong) reject). Benchmarked:
# llava:7b → 81% acc, FP 3/12 with this phrasing (vs FP 9/12 on a naive prompt).
PROMPT = ("This is a photo of a shipping parcel. Most parcels are FINE. "
          "Only report damage if you clearly see SEVERE damage: large rips, the "
          "box crushed or collapsed, or holes with contents spilling out. Minor "
          "tape, labels, or a normal closed box = not damaged. "
          "Answer with one word: DAMAGED or FINE.")

_available: bool | None = None


def available() -> bool:
    global _available
    if _available is None:
        if not USE_VISION:
            _available = False
            return _available
        try:
            with urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3) as r:
                tags = json.loads(r.read())
            names = {m["name"] for m in tags.get("models", [])}
            _available = any(n.split(":")[0] == VISION_MODEL.split(":")[0] for n in names)
            log.info("vision model %s available=%s", VISION_MODEL, _available)
        except Exception:
            _available = False
            log.info("vision model unavailable — falling back to text damage signals")
    return _available


def _download(url: str) -> bytes:
    url = urljoin(API_BASE + "/", url)
    with urllib.request.urlopen(url, timeout=TIMEOUT) as r:
        return r.read()


def has_damage(photo_url: str) -> bool | None:
    """True/False from the image, or None if vision is unavailable / errors."""
    if not available():
        return None
    try:
        img = _download(photo_url)
        b64 = base64.b64encode(img).decode()
        data = json.dumps({
            "model": VISION_MODEL, "prompt": PROMPT, "images": [b64],
            "stream": False, "options": {"temperature": 0, "num_predict": 12},
        }).encode()
        req = urllib.request.Request(f"{OLLAMA_HOST}/api/generate", data=data,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
            ans = json.loads(r.read())["response"].strip().upper()
    except Exception:
        log.exception("vision damage check failed for %s", photo_url)
        return None

    log.info("vision damage(%s) → %r", photo_url.split("/")[-1], ans[:20])
    if "DAMAG" in ans or "YES" in ans:
        return True
    if "FINE" in ans or "NO" in ans or "INTACT" in ans:
        return False
    return None
