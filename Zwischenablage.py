
import os
import sys
from pathlib import Path


def find_lab_dir() -> Path:
    starts = [Path.cwd()] + ([Path(__file__).resolve().parent] if "__file__" in globals() else [])
    for start in starts:
        for p in [start, *start.parents]:
            for candidate in (p, p / "labs"):
                if (candidate / "common" / "paths.py").exists():
                    return candidate
    raise FileNotFoundError("Ordner labs/ mit common/ nicht gefunden")


os.chdir(find_lab_dir())
sys.path.insert(0, str(Path.cwd()))

import hashlib
import re
import unicodedata

import pandas as pd

from common.lab1_rohkorpus import build_raw_corpus
from common.paths import ARTIFACTS_DIR

pd.set_option("display.max_colwidth", 90)



raw = build_raw_corpus()
print(f"{len(raw):,} Dokumente".replace(",", "."))
print(raw["quelle"].value_counts().to_string())
print("\n--- eine HTML-Seite, wie sie aus der Textextraktion kommt ---")
print(raw.loc[raw["quelle"] == "gii-html", "text"].iloc[0])



log = {d: {"navigation_zeilen": 0, "sprache": 0, "pii": None, "dublette": None, "sha256": None,
           "status": "behalten", "grund": None} for d in raw["doc_id"]}


def drop(df: pd.DataFrame, reasons: dict[str, str], status: str = "verworfen") -> pd.DataFrame:
    """Nimmt die Dokumente aus reasons (doc_id -> Grund) aus df und hält Status und Grund im Protokoll fest."""
    for d, reason in reasons.items():
        log[d].update(status=status, grund=reason)
    return df[~df["doc_id"].isin(reasons)].reset_index(drop=True)


df = raw.copy()
df["norm"] = df["gesetz"].str.cat(df["enbez"], sep=" ") 



NAVIGATION = {"zurück", "weiter", "Nichtamtliches Inhaltsverzeichnis", "zum Seitenanfang", "Impressum",
              "Datenschutz", "Barrierefreiheitserklärung", "Feedback-Formular"}
SINGLE_NORM = re.compile(r"^.+ - Einzelnorm$")             # Titelzeile der Seite: "§ 535 BGB - Einzelnorm"


def remove_navigation(text: str) -> tuple[str, int]:
    rows = text.split("\n")
    rest = [z for z in rows if z.strip() not in NAVIGATION and not SINGLE_NORM.match(z.strip())]
    return "\n".join(rest), len(rows) - len(rest)


def is_placeholder(text: str) -> bool:                    # ganzer Text in Klammern, z. B. "(weggefallen)"
    t = text.strip()
    return t == "-" or ("\n" not in t and t.startswith("(") and t.endswith(")") and len(t.split()) <= 8)


fresh = df["text"].map(remove_navigation)                 # je Dokument: (Text ohne Navigation, Zahl der Zeilen)
df["text"] = fresh.str[0]
for d, n in zip(df["doc_id"], fresh.str[1]):
    log[d]["navigation_zeilen"] = int(n)
placeholders = df["text"].map(is_placeholder)
print("Dokumente mit Navigation:", (fresh.str[1] > 0).sum())
print("Platzhalter verworfen:   ", placeholders.sum())
print(df.loc[placeholders, "text"].value_counts().head(6).to_string())
df = drop(df, dict.fromkeys(df.loc[placeholders, "doc_id"], "platzhalter"))
print("\n--- dieselbe HTML-Seite danach ---")
print(df.loc[df["quelle"] == "gii-html", "text"].iloc[0])





