"""Optional local-LLM extraction via ollama (no cloud API key needed).

Used only as an augmentation for AUDIO transcripts, whose translated/garbled
phrasing varies more than the templated emails. Regex stays primary and
authoritative; the LLM fills gaps the regex leaves. Degrades gracefully to a
no-op if the ollama server is unreachable.
"""

import json
import logging
import urllib.request

from config import LLM_MODEL, OLLAMA_HOST, USE_LLM

log = logging.getLogger("llm")

OLLAMA_URL = f"{OLLAMA_HOST}/api/generate"
MODEL      = LLM_MODEL
TIMEOUT    = 20

_available: bool | None = None

_PROMPT = """You extract delivery info from a logistics voice message that may be \
noisy or partially mistranslated. Reply with ONLY a compact JSON object, no prose:
{{"supplier_name": string or null, "count": integer or null, \
"unit": "parcels" or "pallets" or null, \
"goods_type": "standard" or "oversized" or "perishable" or null}}

Rules:
- count = number of items being delivered.
- unit = "pallets" if pallets/pallet are mentioned, else "parcels" for parcels/packages/boxes.
- goods_type = "perishable" (chilled/frozen/fresh/cold), "oversized" (bulky/over-dimensional/sperrgut), else "standard".
- supplier_name = the sending company only, stripped of filler like "the company".

Message: "{text}"
JSON:"""


def available() -> bool:
    global _available
    if _available is None:
        if not USE_LLM:
            _available = False
            return _available
        try:
            urllib.request.urlopen(f"{OLLAMA_HOST}/api/tags", timeout=3)
            _available = True
            log.info("ollama LLM available (%s)", MODEL)
        except Exception:
            _available = False
            log.info("ollama LLM not available — regex-only extraction")
    return _available


def _call(prompt: str) -> str:
    data = json.dumps({
        "model": MODEL, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "num_predict": 120},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())["response"]


def extract(text: str) -> dict | None:
    """Return {'supplier_name','count','unit','goods_type'} or None on failure."""
    if not text or not available():
        return None
    try:
        raw = _call(_PROMPT.format(text=text.replace('"', "'")))
    except Exception:
        log.exception("LLM call failed")
        return None

    # pull the first JSON object out of the response (model may wrap in ```)
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end < 0:
        log.warning("LLM returned no JSON: %s", raw[:120])
        return None
    try:
        obj = json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        log.warning("LLM JSON parse failed: %s", raw[start:end + 1][:120])
        return None

    out: dict = {}
    name = obj.get("supplier_name")
    if isinstance(name, str) and name.strip():
        out["supplier_name"] = name.strip()
    if isinstance(obj.get("count"), int):
        out["count"] = obj["count"]
    if obj.get("unit") in ("parcels", "pallets"):
        out["unit"] = obj["unit"]
    if obj.get("goods_type") in ("standard", "oversized", "perishable"):
        out["goods_type"] = obj["goods_type"]
    log.debug("LLM extract → %s", out)
    return out or None
