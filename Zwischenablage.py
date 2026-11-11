
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
