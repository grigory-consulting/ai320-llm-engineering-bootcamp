"""Gesetzeskorpus: Bundesgesetze von gesetze-im-internet.de (amtliche Werke, § 5 UrhG).

normen.parquet hat eine Zeile je Einzelnorm (Paragraph oder Artikel) mit den Spalten
gesetz, rechtsgebiet, enbez, titel, text, builddate.
Die Datei liegt dem Kurs bei; prefetch.py baut sie nur neu, wenn sie fehlt.
"""
from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
import zipfile

import pandas as pd

from common.paths import LAWS_DIR, NORMS_PARQUET

BASE_URL = "https://www.gesetze-im-internet.de/{slug}/xml.zip"

# Slug auf gesetze-im-internet.de -> Rechtsgebiet (Zielgröße der Klassifikation in Lab 4)
LAWS = {
    "bgb": "Zivilrecht", "hgb": "Zivilrecht", "zpo": "Zivilrecht",
    "gmbhg": "Zivilrecht", "aktg": "Zivilrecht", "inso": "Zivilrecht", "urhg": "Zivilrecht",
    "stgb": "Strafrecht", "stpo": "Strafrecht", "owig_1968": "Strafrecht", "jgg": "Strafrecht",
    "gg": "Öffentliches Recht", "vwvfg": "Öffentliches Recht", "vwgo": "Öffentliches Recht",
    "bdsg_2018": "Öffentliches Recht", "bimschg": "Öffentliches Recht", "bbaug": "Öffentliches Recht",
    "ao_1977": "Steuerrecht", "estg": "Steuerrecht", "ustg_1980": "Steuerrecht",
    "kstg_1977": "Steuerrecht", "gewstg": "Steuerrecht",
    "sgb_1": "Arbeits- und Sozialrecht", "sgb_5": "Arbeits- und Sozialrecht",
    "sgb_6": "Arbeits- und Sozialrecht", "arbzg": "Arbeits- und Sozialrecht",
    "kschg": "Arbeits- und Sozialrecht", "betrvg": "Arbeits- und Sozialrecht",
}


# Elemente, an deren Grenzen ein Leerzeichen stehen muss (Aufzählungen, Tabellen, Umbrüche).
# itertext() allein klebt sonst "1.die Approbation,2.den Abschluss" zusammen.
_BOUNDARY = {"DL", "DT", "DD", "LA", "BR", "table", "tgroup", "thead", "tbody", "row", "entry"}


def _flatten(el) -> str:
    pieces = []

    def walk(e):
        if e.tag in _BOUNDARY:
            pieces.append(" ")
        if e.text:
            pieces.append(e.text)
        for child in e:
            walk(child)
            if child.tail:
                pieces.append(child.tail)
        if e.tag in _BOUNDARY:
            pieces.append(" ")

    walk(el)
    return re.sub(r"\s+", " ", "".join(pieces)).strip()


def _text(el) -> str:
    if el is None:
        return ""
    paragraphs = [t for t in (_flatten(p) for p in el.iter("P")) if t]
    return "\n".join(paragraphs)


def parse_xml(raw: bytes, law_slug: str, area_of_law: str) -> list[dict]:
    root = ET.fromstring(raw)
    rows = []
    for norm in root.iter("norm"):
        meta = norm.find("metadaten")
        designation = (meta.findtext("enbez") or "").strip()
        if not designation.startswith(("§", "Art")):
            continue
        text = _text(norm.find("textdaten/text/Content"))
        if not text or text in ("-", "(weggefallen)"):
            continue
        rows.append({
            "gesetz": (meta.findtext("jurabk") or law_slug).strip(),
            "rechtsgebiet": area_of_law,
            "enbez": designation,
            "titel": (meta.findtext("titel") or "").strip(),
            "text": text,
            "builddate": norm.get("builddate", ""),
        })
    return rows


def download(slug: str) -> bytes:
    import httpx

    r = httpx.get(BASE_URL.format(slug=slug), timeout=60, follow_redirects=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xml"))
        return z.read(name)


def build() -> pd.DataFrame:
    LAWS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for slug, area in LAWS.items():
        rows += parse_xml(download(slug), slug, area)
    df = pd.DataFrame(rows)
    df.to_parquet(NORMS_PARQUET, index=False)
    return df


def load() -> pd.DataFrame:
    """Lädt den Gesetzeskorpus (eine Zeile je Paragraph/Artikel)."""
    if not NORMS_PARQUET.exists():
        raise FileNotFoundError(f"{NORMS_PARQUET} fehlt: einmal `python prefetch.py` ausführen")
    return pd.read_parquet(NORMS_PARQUET)
