"""
Multimodal extraction via Anthropic Claude.

Single call per truck combining: photo (vision), email body (text), audio
transcript (text). Output is forced through a tool_use schema so we always
get well-typed fields.
"""

from __future__ import annotations

import base64
import logging
import os

import anthropic
import httpx

from config import ANTHROPIC_MODEL, ANTHROPIC_MAX_TOKENS
from ._urls import normalize_url

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
Du bist ein Wareneingangs-Assistent in einem Logistikzentrum. Du erhältst pro
LKW bis zu drei Informationsquellen:

  1. Ein Foto der Lieferung / des LKWs (kann Transportschäden zeigen).
  2. Eine E-Mail vom Lieferanten (Freitext, mehrsprachig — beliebige Sprache
     möglich: DE / FR / IT / EN / ES / NL / PT / SV / AR / ... — mit Tippfehlern,
     Abkürzungen, gemischten Sprachen).
  3. Eine Audio-Transkription einer Sprachnachricht (beliebige Sprache,
     evtl. fehlerhaft, mit Akzent oder Rauschen).

Extrahiere strikt und faktenbasiert genau die Felder, die das `extract_truck_data`
Tool verlangt. Halte dich an folgende Regeln:

• `supplier_name_raw`: Name des Lieferanten **wörtlich und vollständig** —
  inklusive Rechtsform-Suffixen wie "Inc.", "plc", "Corp.", "Ltd.", "AG",
  "GmbH", "Holdings", "Trust" etc. Diese Suffixe sind WICHTIG: das System
  unterscheidet z. B. "Actavis", "Actavis Inc." und "Actavis plc" als
  verschiedene Lieferanten. Wenn die Quellen abweichen, nimm die Variante
  aus der primären Quelle (E-Mail-Subject > E-Mail-Body > Audio).

• `parcel_count`: ganze Zahl > 0. Wenn mehrere Quellen sich widersprechen,
  wähle die plausibelste (Audio-Transkripte sind oft fehlerhaft bei Zahlen!).

• `unit`:
    - "parcels"  → Pakete / Päckchen / colis / pacchi / paquet / box / Karton.
    - "pallets"  → Paletten / palettes / pallet / EUR-Palette.
  Wenn unklar, bevorzuge die Einheit, die zur Mengenangabe passt.

• `has_damage`: prüfe DREI Quellen unabhängig, dann ODER-verknüpft:

    1. **Foto**  → true bei sichtbarem:
       - aufgerissenen / eingedrückten / zerquetschten Kartons,
       - aufgeklappten oder fehlenden Klappen mit freiliegendem Inhalt,
       - sichtbaren Löchern, Rissen, Beulen, Stauchungen,
       - Flüssigkeitsaustritt, Verfärbungen durch Nässe.
       NICHT als Schaden zählen: Kratzer auf Klebeband, leichte Verformungen
       der Außenfolie, abgegriffene Labels, normale Gebrauchsspuren, eingedrückte
       Versandetiketten. Wenn unsicher zwischen "leichte Spur" und "Schaden":
       wähle FALSE.

    2. **E-Mail** → true bei expliziter Schadens-Sprache:
       - DE: beschädigt, kaputt, zerbrochen, gerissen, durchnässt, undicht,
             erhebliche Transportschäden, "cargo damage", Bruchware.
       - FR: endommagé, déchiré, cassé, fuite, dommages de transport.
       - IT: danneggiato, rotto, strappato, perdita.
       - EN: damaged, broken, leaking, torn, "major cargo damage".
       Reine Vorsichts-Hinweise wie "bitte vorsichtig handhaben" oder
       "fragile" sind KEIN Schaden.

    3. **Audio-Transkript** → gleiche Wörter wie E-Mail, beliebige Sprache.

    Wenn EINE Quelle klar Schaden meldet: `true`. Wenn keine: `false`.
    Bei Konflikt zwischen Foto und Text dem klarer Aussagenden folgen
    (Foto > expliziter Text > vager Hinweis).

    Sei konservativ: Falsch-positive Schäden kosten Punkte. Wenn du wirklich
    unsicher bist, wähle FALSE.

• `goods_type`:
    - "perishable" → Kühlware, frisch, gefroren, frais, frigorifique, fresco,
                     refrigerated, cold chain, +2°C, Trockeneis. Hat Vorrang!
    - "oversized"  → übergroß, sperrig, oversize, hors gabarit, voluminoso,
                     Maschinen / Anlagen / Maschinenteile, > 2.5m, schwere
                     Einzelteile.
    - "standard"   → alles andere.

• `reasoning`: ein kurzer Satz, warum du dich für diese Werte entschieden hast.
  Nutze ihn, um Unsicherheit oder Quellen-Konflikte transparent zu machen.

WICHTIG — niemals Platzhalter zurückgeben:
  • Auch wenn Quellen fehlen oder unklar sind, MUSST du jedes Feld mit einem
    konkreten, plausiblen Wert befüllen. KEINE Strings wie "UNKNOWN", "?",
    "N/A" — das Tool-Schema lässt sie nicht zu.
  • Wenn keine verwertbaren Quellen vorliegen: rate konservativ (z. B.
    parcel_count=1, unit="parcels", goods_type="standard", has_damage=false,
    supplier_name_raw="Unbekannt") und vermerke das im reasoning.

Gib KEINE freie Prosa zurück — rufe ausschließlich das Tool auf.
"""


EXTRACTION_TOOL = {
    "name": "extract_truck_data",
    "description": "Strukturierte Wareneingangs-Daten aus den Quellen extrahieren.",
    "input_schema": {
        "type": "object",
        "properties": {
            "supplier_name_raw": {
                "type": "string",
                "description": "Name des Lieferanten, wie aus den Quellen erkannt.",
            },
            "parcel_count": {
                "type": "integer",
                "minimum": 1,
                "description": "Anzahl der Pakete oder Paletten.",
            },
            "unit": {
                "type": "string",
                "enum": ["parcels", "pallets"],
            },
            "has_damage": {
                "type": "boolean",
                "description": "True nur bei klar erkennbarem Transportschaden.",
            },
            "goods_type": {
                "type": "string",
                "enum": ["standard", "oversized", "perishable"],
            },
            "reasoning": {
                "type": "string",
                "description": "Kurze Begründung der Extraktion.",
            },
        },
        "required": [
            "supplier_name_raw", "parcel_count", "unit",
            "has_damage", "goods_type", "reasoning",
        ],
    },
}


_MEDIA_BY_EXT = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".gif":  "image/gif",
    ".webp": "image/webp",
}


def _media_type_for(url: str) -> str:
    ext = os.path.splitext(url.split("?")[0])[1].lower()
    return _MEDIA_BY_EXT.get(ext, "image/jpeg")


class ClaudeExtractor:
    def __init__(self, model: str = ANTHROPIC_MODEL):
        self.model = model
        self.client = anthropic.AsyncAnthropic()

    async def fetch_photo(self, url: str) -> tuple[bytes, str]:
        normalized = normalize_url(url) or url
        log.debug("Photo download URL: raw=%r normalized=%r", url, normalized)
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.get(normalized)
            r.raise_for_status()
            return r.content, _media_type_for(normalized)

    async def extract(
        self,
        photo_bytes: bytes | None,
        photo_media_type: str | None,
        email_text: str,
        audio_transcript: str,
        model: str | None = None,
    ) -> dict:
        content: list[dict] = []

        if photo_bytes is not None:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": photo_media_type or "image/jpeg",
                    "data": base64.b64encode(photo_bytes).decode("ascii"),
                },
            })

        prompt_parts = []
        if email_text:
            prompt_parts.append(f"--- E-Mail vom Lieferanten ---\n{email_text.strip()}")
        else:
            prompt_parts.append("--- E-Mail vom Lieferanten ---\n(keine E-Mail vorhanden)")

        if audio_transcript:
            prompt_parts.append(f"--- Audio-Transkription ---\n{audio_transcript.strip()}")
        else:
            prompt_parts.append("--- Audio-Transkription ---\n(keine Audio-Nachricht)")

        content.append({"type": "text", "text": "\n\n".join(prompt_parts)})

        chosen_model = model or self.model
        response = await self.client.messages.create(
            model=chosen_model,
            max_tokens=ANTHROPIC_MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=[EXTRACTION_TOOL],
            tool_choice={"type": "tool", "name": "extract_truck_data"},
            messages=[{"role": "user", "content": content}],
        )

        for block in response.content:
            if block.type == "tool_use" and block.name == "extract_truck_data":
                data = dict(block.input)
                log.debug("Claude extraction: %s", data)
                return data

        raise RuntimeError(
            f"Claude did not emit expected tool_use block. Stop reason={response.stop_reason}, "
            f"content={response.content!r}"
        )

    async def pick_supplier(
        self,
        raw_name: str,
        candidates: list[tuple[int, str, float]],
        email_text: str,
        audio_transcript: str,
        model: str | None = None,
    ) -> str | None:
        """
        Given the extracted raw supplier name and a fuzzy-pre-filtered short list of
        candidates, ask Claude to pick the best match — uses general company knowledge
        + the original sources for context. Returns the chosen canonical supplier name
        (must be one of the provided candidates) or None if Claude declines.
        """
        if not candidates:
            return None

        candidate_names = [name for _sid, name, _score in candidates]
        candidate_lines = "\n".join(
            f"  {i+1}. {name} (Fuzzy-Score {score:.0f})"
            for i, (_sid, name, score) in enumerate(candidates)
        )
        user_text = (
            f"Aus dem Foto, der E-Mail und der Audio-Transkription wurde der Lieferantenname\n"
            f"  \"{raw_name}\"\n"
            f"extrahiert (vermutlich phonetisch oder grob aus einer verrauschten Quelle).\n\n"
            f"Die Lieferantendatenbank enthält folgende ähnliche Einträge:\n{candidate_lines}\n\n"
            f"--- Original E-Mail (Auszug) ---\n{(email_text or '').strip()[:800]}\n\n"
            f"--- Original Audio-Transkription (Auszug) ---\n{(audio_transcript or '').strip()[:800]}\n\n"
            f"Aufgabe:\n"
            f"  • Falls EINER der Kandidaten klar oder zumindest plausibel zu \"{raw_name}\" "
            f"    passt (phonetisch, semantisch oder aus deinem Weltwissen), wähle ihn aus.\n"
            f"  • Falls KEINER der Kandidaten überzeugend passt (alle nur zufällige "
            f"    Buchstaben-Überlappung, kein bekanntes Unternehmen erkennbar, oder die\n"
            f"    Quellen so verrauscht sind, dass jeder Pick reines Raten wäre), wähle "
            f"    `NO_MATCH`.\n\n"
            f"Falsches confidence-Pick kostet das System Punkte — wenn unsicher, lieber "
            f"`NO_MATCH` als ein Glücksspiel."
        )

        pick_tool = {
            "name": "pick_supplier",
            "description": ("Wähle den passendsten Lieferanten aus der Kandidatenliste — "
                            "oder NO_MATCH wenn kein Kandidat überzeugt."),
            "input_schema": {
                "type": "object",
                "properties": {
                    "supplier_name": {
                        "type": "string",
                        "enum": candidate_names + ["NO_MATCH"],
                    },
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "reasoning": {"type": "string"},
                },
                "required": ["supplier_name", "confidence", "reasoning"],
            },
        }

        chosen_model = model or self.model
        try:
            response = await self.client.messages.create(
                model=chosen_model,
                max_tokens=512,
                system="Du bist ein Wareneingangs-Assistent. Wähle aus einer kurzen "
                       "Kandidatenliste den passendsten Lieferanten.",
                tools=[pick_tool],
                tool_choice={"type": "tool", "name": "pick_supplier"},
                messages=[{"role": "user", "content": user_text}],
            )
        except Exception:
            log.exception("Supplier disambiguation call failed")
            return None

        for block in response.content:
            if block.type == "tool_use" and block.name == "pick_supplier":
                data = dict(block.input)
                pick = data.get("supplier_name")
                log.info("Supplier disambiguation → %r (confidence=%s, reasoning=%r)",
                         pick, data.get("confidence"), data.get("reasoning"))
                if pick == "NO_MATCH":
                    return None
                if pick in candidate_names:
                    return pick
                return None
        return None
