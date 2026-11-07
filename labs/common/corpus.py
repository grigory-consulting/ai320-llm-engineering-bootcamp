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

from common.paths import GESETZE_DIR, NORMEN_PARQUET

BASE_URL = "https://www.gesetze-im-internet.de/{slug}/xml.zip"

# Slug auf gesetze-im-internet.de -> Rechtsgebiet (Zielgröße der Klassifikation in Lab 4)
GESETZE = {
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
_GRENZE = {"DL", "DT", "DD", "LA", "BR", "table", "tgroup", "thead", "tbody", "row", "entry"}


def _flach(el) -> str:
    teile = []

    def lauf(e):
        if e.tag in _GRENZE:
            teile.append(" ")
        if e.text:
            teile.append(e.text)
        for kind in e:
            lauf(kind)
            if kind.tail:
                teile.append(kind.tail)
        if e.tag in _GRENZE:
            teile.append(" ")

    lauf(el)
    return re.sub(r"\s+", " ", "".join(teile)).strip()


def _text(el) -> str:
    if el is None:
        return ""
    absaetze = [t for t in (_flach(p) for p in el.iter("P")) if t]
    return "\n".join(absaetze)


def parse_xml(raw: bytes, gesetz: str, rechtsgebiet: str) -> list[dict]:
    root = ET.fromstring(raw)
    zeilen = []
    for norm in root.iter("norm"):
        meta = norm.find("metadaten")
        enbez = (meta.findtext("enbez") or "").strip()
        if not enbez.startswith(("§", "Art")):
            continue
        text = _text(norm.find("textdaten/text/Content"))
        if not text or text in ("-", "(weggefallen)"):
            continue
        zeilen.append({
            "gesetz": (meta.findtext("jurabk") or gesetz).strip(),
            "rechtsgebiet": rechtsgebiet,
            "enbez": enbez,
            "titel": (meta.findtext("titel") or "").strip(),
            "text": text,
            "builddate": norm.get("builddate", ""),
        })
    return zeilen


def download(slug: str) -> bytes:
    import httpx

    r = httpx.get(BASE_URL.format(slug=slug), timeout=60, follow_redirects=True)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        name = next(n for n in z.namelist() if n.endswith(".xml"))
        return z.read(name)


def build() -> pd.DataFrame:
    GESETZE_DIR.mkdir(parents=True, exist_ok=True)
    zeilen = []
    for slug, gebiet in GESETZE.items():
        zeilen += parse_xml(download(slug), slug, gebiet)
    df = pd.DataFrame(zeilen)
    df.to_parquet(NORMEN_PARQUET, index=False)
    return df


def load() -> pd.DataFrame:
    """Lädt den Gesetzeskorpus (eine Zeile je Paragraph/Artikel)."""
    if not NORMEN_PARQUET.exists():
        raise FileNotFoundError(f"{NORMEN_PARQUET} fehlt: einmal `python prefetch.py` ausführen")
    return pd.read_parquet(NORMEN_PARQUET)
