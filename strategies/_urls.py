"""
URL normalization: the server sometimes emits documentation URLs without
an http/https scheme (relative paths or scheme-less). Coerce them against
the configured API_BASE so httpx accepts them.
"""

from __future__ import annotations

from urllib.parse import urlparse

from config import API_BASE


def normalize_url(url: str | None) -> str | None:
    if not url:
        return url
    parsed = urlparse(url)
    if parsed.scheme in ("http", "https"):
        return url
    if url.startswith("//"):
        return "https:" + url
    if url.startswith("/"):
        return f"{API_BASE}{url}"
    return f"{API_BASE}/{url}"
