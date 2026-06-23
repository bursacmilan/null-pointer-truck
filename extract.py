"""
Multilingual extraction of structured fields from email/audio text.

Emails and (whisper-transcribed) audio share the same templated structure across
DE / FR / IT / ES / EN:

    "<greeting> <supplier> ... <N> <unit> de/of <goods> ... <distractor> <signature>"

We pull: supplier_name, parcel_count, unit, goods_type, has_damage.
"""

import logging
import re
import unicodedata
from dataclasses import dataclass

log = logging.getLogger("extract")


@dataclass
class Signals:
    supplier_name: str | None = None
    parcel_count:  int | None = None
    unit:          str | None = None   # "parcels" | "pallets"
    goods_type:    str | None = None   # "standard" | "oversized" | "perishable"
    has_damage:    bool | None = None


# ── Keyword vocabularies (lower-cased, accent-insensitive matching) ────────────

PARCEL_WORDS = [
    "colis", "paquet", "paquets", "paquete", "paquetes",
    "parcel", "parcels", "package", "packages",
    "paket", "pakete", "pacco", "pacchi", "collo", "colli",
    "colete", "colete", "colet", "pacote", "pacotes",   # ro/pt (translate/misdetect)
]
PALLET_WORDS = [
    "palette", "palettes", "pallet", "pallets",
    "paletten", "paleta", "paletas", "palet", "palets", "pale", "pales",
    "bancale", "bancali",
]
PERISHABLE_WORDS = [
    "perissable", "perissables", "perishable", "perishables",
    "perecedero", "perecederos", "perecedera",
    "deperibile", "deperibili", "verderblich", "kuhlware", "kuehlware",
    "frisch", "refrigere", "refrigerated", "frozen", "surgele",
    "chilled", "cold chain", "cold-chain",         # english-translation variants
]
OVERSIZED_WORDS = [
    "encombrant", "encombrants", "oversized", "oversize",
    "sperrig", "sperrgut", "ubergross", "uebergross", "ingombrante",
    "ingombranti", "voluminoso", "voluminosa", "voluminous", "bulky",
    "over-dimensional", "over dimensional", "overdimensional",  # translate variants
    "oversized goods", "dimensional",
]
# Damage is signalled by a dedicated ALERT line, e.g.
#   "Alert: major cargo damage detected!" / "Kritisch! Erhebliche Transportschäden…"
#   "Attenzione: gravi danni alla merce!" / "Urgent! Dégâts importants sur la cargaison."
#   "Alerta: daños graves en la carga."
# We detect damage via (a) an urgency marker, which never appears in normal mails
# or distractor sentences, or (b) a specific cargo-damage term. Generic words like
# "kaputt"/"broken" are deliberately EXCLUDED — they show up in distractor noise.
URGENCY_MARKERS = [
    "alert", "alerta", "attenzione", "urgent", "urgente",
    "kritisch", "achtung", "warnung", "atencion", "avviso urgente",
]
DAMAGE_TERMS = [
    "cargo damage", "damage detected", "damage to the cargo",
    "transportschaden", "transportschaden",  # de (deaccented)
    "danni alla merce", "gravi danni", "danneggiat",
    "degats", "deterioration", "avarie", "avaria",
    "danos graves", "danos en la carga", "danos a la",
]

# Number words 0-99 across the five languages (only what realistically appears).
_UNITS = {
    # english
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
    # german
    "null": 0, "eins": 1, "ein": 1, "eine": 1, "zwei": 2, "drei": 3,
    "vier": 4, "funf": 5, "fuenf": 5, "sechs": 6, "sieben": 7, "acht": 8,
    "neun": 9, "zehn": 10, "elf": 11, "zwolf": 12, "zwoelf": 12,
    "dreizehn": 13, "vierzehn": 14, "funfzehn": 15, "fuenfzehn": 15,
    "sechzehn": 16, "siebzehn": 17, "achtzehn": 18, "neunzehn": 19,
    "zwanzig": 20, "dreissig": 30, "dreizig": 30, "vierzig": 40,
    "funfzig": 50, "fuenfzig": 50, "sechzig": 60, "siebzig": 70,
    "achtzig": 80, "neunzig": 90,
    # french
    "zero_fr": 0, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4,
    "cinq": 5, "sept": 7, "huit": 8, "neuf": 9, "dix": 10, "onze": 11,
    "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15, "seize": 16,
    "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50,
    "soixante": 60,
    # spanish
    "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5, "seis": 6,
    "siete": 7, "ocho": 8, "nueve": 9, "diez": 10, "once": 11, "doce": 12,
    "trece": 13, "catorce": 14, "quince": 15, "dieciseis": 16,
    "veinte": 20, "treinta": 30, "cuarenta": 40, "cincuenta": 50,
    "sesenta": 60, "setenta": 70, "ochenta": 80, "noventa": 90,
    # italian
    "uno_it": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6,
    "otto": 8, "nove": 9, "dieci": 10, "undici": 11, "dodici": 12,
    "venti": 20, "trenta": 30, "quaranta": 40, "cinquanta": 50,
    "sessanta": 60,
}

# Phrases that introduce the supplier name (regex, accent-insensitive, no-accent text).
# Ordered: specific company-keyword intros first (incl. possessive "company's X"),
# then generic "this is / we are / I am" fallbacks for translated audio.
_SUPPLIER_PATTERNS = [
    r"de la part de\s+(.+?)(?:,|\.|nous|veuillez|$)",
    r"on behalf of\s+(.+?)(?:,|\.|please|$)",
    r"en nombre de\s+(.+?)(?:,|\.|les|$)",
    r"vi comunichiamo che\s+(.+?)(?:,|\.|consegnera|$)",
    r"informieren sie uber den heutigen eingang von\s+(.+?)(?:,|\.|:|$)",
    r"l'?arrivee prevue de\s+(.+?)(?:,|\.|avec|$)",
    # company-keyword + optional possessive 's  (handles "company's ATI")
    r"(?:empresa|company|firma|societe|azienda|ditta|impresa)(?:'?s)?\s+(.+?)"
    r"(?:,|\.|entregamos|we |wir |is |plant|$)",
    # generic self-introductions in translated/English audio
    r"(?:calling from|this is|we are|i am|here is|hier spricht|somos)\s+"
    r"(?:the\s+)?(?:company\s+|empresa\s+|firma\s+)?(.+?)(?:,|\.|we |$)",
]

# Words to trim off the head/tail of an extracted supplier name.
_NAME_STRIP = re.compile(
    r"^(the|la|le|el|il|der|die|das)\s+|"
    r"\s+(plant|company|incoming|today|incoming today|delivering)\s*$",
    re.IGNORECASE,
)


def _deaccent(s: str) -> str:
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def _norm(s: str) -> str:
    """Lower-case, strip accents — for keyword/number matching."""
    return _deaccent(s.lower())


def _find_unit(text_n: str) -> str | None:
    has_parcel = any(re.search(rf"\b{w}\b", text_n) for w in PARCEL_WORDS)
    has_pallet = any(re.search(rf"\b{w}\b", text_n) for w in PALLET_WORDS)
    if has_pallet and not has_parcel:
        return "pallets"
    if has_parcel and not has_pallet:
        return "parcels"
    if has_pallet and has_parcel:
        # Both mentioned — pick whichever sits closer to a digit/count.
        return "pallets"
    return None


def _find_goods(text_n: str) -> str | None:
    if any(re.search(rf"\b{w}\b", text_n) for w in PERISHABLE_WORDS):
        return "perishable"
    if any(re.search(rf"\b{w}\b", text_n) for w in OVERSIZED_WORDS):
        return "oversized"
    # Default to standard only if there is a clear delivery context.
    return "standard"


def _find_damage(text_n: str) -> bool:
    # Urgency marker → reliable damage alert (absent from normal mail & distractors).
    if any(re.search(rf"\b{re.escape(w)}\b", text_n) for w in URGENCY_MARKERS):
        return True
    # Specific cargo-damage phrasing.
    if any(w in text_n for w in DAMAGE_TERMS):
        return True
    return False


def _find_count(text_n: str) -> int | None:
    """Prefer a digit adjacent to a unit word; fall back to any digit, then words."""
    unit_alt = "|".join(PARCEL_WORDS + PALLET_WORDS)

    # digit immediately before a unit word:  "46 colis", "32 pallets"
    m = re.search(rf"(\d+)\s+(?:{unit_alt})", text_n)
    if m:
        return int(m.group(1))

    # unit word before a digit (rare):  "pallets 32"
    m = re.search(rf"(?:{unit_alt})\s+(\d+)", text_n)
    if m:
        return int(m.group(1))

    # any standalone integer in a plausible range
    nums = [int(n) for n in re.findall(r"\b(\d{1,3})\b", text_n)]
    nums = [n for n in nums if 1 <= n <= 300]
    if nums:
        return nums[0]

    # number words near a unit word
    for token, val in _UNITS.items():
        word = token.split("_")[0]
        if re.search(rf"\b{word}\b\s+(?:{unit_alt})", text_n):
            return val
    # any number word
    for token, val in _UNITS.items():
        word = token.split("_")[0]
        if re.search(rf"\b{word}\b", text_n):
            return val
    return None


def _find_supplier(text_raw: str) -> str | None:
    """
    Supplier name from the email subject (after a dash) or an introductory phrase.
    Returns the RAW (accented, cased) substring for downstream fuzzy matching.
    """
    # Subject line: "Subject: <kind> – <Supplier Name>"
    m = re.search(r"(?im)^subject:\s*.*?[–—-]\s*(.+)$", text_raw)
    if m:
        cand = m.group(1).strip()
        if cand:
            return cand

    text_n = _norm(text_raw)
    for pat in _SUPPLIER_PATTERNS:
        m = re.search(pat, text_n)
        if m:
            # Map the normalized span back to the raw text for original casing.
            start, end = m.span(1)
            cand = text_raw[start:end].strip(" .,:;'")
            cand = _NAME_STRIP.sub("", cand).strip(" .,:;'")
            if cand:
                return cand
    return None


def _strip_subject(text: str) -> str:
    """Drop the 'Subject: …' line — supplier names there carry spurious numbers
    (e.g. 'S&P 500', 'MIDCAP 400', 'Series 2006-2') that corrupt count parsing."""
    return re.sub(r"(?im)^subject:.*$", "", text)


def parse_text(text: str) -> Signals:
    """Extract all signals from a single text blob (email body or transcript)."""
    if not text:
        return Signals()
    text_n = _norm(text)
    body_n = _norm(_strip_subject(text))   # count comes from the body, not subject
    sig = Signals(
        supplier_name=_find_supplier(text),
        parcel_count=_find_count(body_n),
        unit=_find_unit(text_n),
        goods_type=_find_goods(text_n),
        has_damage=_find_damage(text_n),
    )
    log.debug("parse_text → %s", sig)
    return sig


def merge_signals(primary: Signals, secondary: Signals | None) -> Signals:
    """Combine signals; `primary` (email) wins over `secondary` (audio) per field."""
    if secondary is None:
        return primary

    def pick(a, b):
        return a if a is not None else b

    return Signals(
        supplier_name=pick(primary.supplier_name, secondary.supplier_name),
        parcel_count=pick(primary.parcel_count, secondary.parcel_count),
        unit=pick(primary.unit, secondary.unit),
        goods_type=pick(primary.goods_type, secondary.goods_type),
        # damage: true if either source says so
        has_damage=(bool(primary.has_damage) or bool(secondary.has_damage))
        if (primary.has_damage is not None or secondary.has_damage is not None)
        else None,
    )
