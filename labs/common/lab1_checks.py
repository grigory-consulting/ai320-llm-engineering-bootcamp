"""Prüfcode für die Lab-1-Übung. Er misst und zeigt, was eine Lösung bewirkt; lesen müssen Sie ihn nicht."""
from __future__ import annotations

import re
from collections import Counter
from typing import Callable

import pandas as pd
from datasketch import MinHash

from common.lab1_kette import five_grams, pairs_above
from common.lab4_data import load, split

# Aufgabe 1 · Rufnummern


def check_phone(pattern: re.Pattern, df: pd.DataFrame, annotation: dict) -> int:
    """Recall der Regex in den internen Texten, Treffer in Normen; gibt die Zahl der Normen mit Treffer zurück."""
    intern = df[df["quelle"] == "intern-synthetisch"]
    hits = dict(zip(intern["doc_id"], intern["text"].map(pattern.findall)))
    expected = [(d, nr) for d in hits for nr in annotation[d]["TELEFON"]]
    found = sum(any(nr in h for h in hits[d]) for d, nr in expected)
    others = [h for d, hs in hits.items() for h in hs if not any(h in nr or nr in h for nr in annotation[d]["TELEFON"])]
    in_norms = df.loc[df["quelle"].str.startswith("gii"), "text"].map(pattern.findall)
    print(f"Recall in den internen Texten: {found} von {len(expected)} Rufnummern; weitere Treffer dort: {len(others)} {others[:4]}")
    print(f"Normen mit Treffer: {(in_norms.str.len() > 0).sum()}; häufigste Treffer:")
    print(pd.Series([h for hs in in_norms for h in hs], dtype=str).value_counts().head(6).to_string())
    return int((in_norms.str.len() > 0).sum())


# Aufgabe 2 · Beinahe-Dubletten


def prepare_near_dups(df: pd.DataFrame) -> tuple[dict, dict, Callable[[str, str], str]]:
    """MinHash-Signaturen (112 Permutationen, Seed 1), Referenzpaare ab Jaccard 0,4 und pair_kind(a, b)."""
    grams = {d: g for d, g in zip(df["doc_id"], df["text"].map(five_grams)) if g}
    signatures = dict(zip(grams, MinHash.bulk([[x.encode("utf-8") for x in g] for g in grams.values()],
                                               num_perm=112, seed=1)))
    reference = pairs_above(grams, 0.4)                      # {(doc_a, doc_b): Jaccard}
    info = df.set_index("doc_id")

    def pair_kind(a: str, b: str) -> str:
        x, y = info.loc[a], info.loc[b]
        if not (x["quelle"].startswith("gii") and y["quelle"].startswith("gii")):
            return "interne Schreiben" if x["quelle"] == y["quelle"] == "intern-synthetisch" else "sonstige"
        return "HTML-Kopie" if (x["gesetz"], x["enbez"]) == (y["gesetz"], y["enbez"]) else "verschiedene Normen"

    return signatures, reference, pair_kind


_candidates: dict = {}


def candidate_pairs(candidates: Callable, params: str) -> set[tuple[str, str]]:
    """candidates(params) für params wie "14x8" (14 Buckets zu je 8 Werten); merkt sich das Ergebnis."""
    key = (candidates, params)
    if key not in _candidates:
        _candidates[key] = candidates(tuple(int(x) for x in params.split("x")))
    return _candidates[key]


def show_lsh(candidates: Callable, threshold: float, params: str, reference: dict,
             pair_kind: Callable[[str, str], str], df: pd.DataFrame) -> None:
    """Welche Referenzpaare ab threshold schlägt LSH mit params vor, nach Art, und was bleibt von den HTML-Kopien."""
    k = candidate_pairs(candidates, params)
    info = df.set_index("doc_id")
    expected = {p: j for p, j in reference.items() if j >= threshold}
    found = [p for p in expected if p in k]
    kinds = Counter(pair_kind(*p) for p in found)
    html_removed = {d for p in found if pair_kind(*p) == "HTML-Kopie" for d in p if info.loc[d, "quelle"] == "gii-html"}
    print(f"params={tuple(int(x) for x in params.split('x'))}: {len(k)} Kandidatenpaare; "
          f"Referenzpaare ab {threshold:.2f}: {len(expected)}, "
          f"davon vorgeschlagen {len(found)} ({len(found) / max(1, len(expected)):.1%})")
    print("gefundene Paare nach Art:", dict(kinds), f"| HTML-Kopien ohne gefundenes Paar: {400 - len(html_removed)}")
    tight = sorted((j, p) for p, j in expected.items() if p in k and pair_kind(*p) == "verschiedene Normen")[:3]
    for j, (a, b) in tight:
        print(f"\n  {j:.2f}  {info.loc[a, 'gesetz']} {info.loc[a, 'enbez']}: {info.loc[a, 'text'][:110]}"
              f"\n        {info.loc[b, 'gesetz']} {info.loc[b, 'enbez']}: {info.loc[b, 'text'][:110]}")


# Aufgabe 3 · Manifest-Einträge

ENTRY_FIELDS = ["doc_id", "quelle", "quelle_url", "abruf", "rechte", "sha256", "verarbeitung", "status", "grund"]
RIGHTS_FIELDS = ["dataset_lizenz", "dokument", "nutzungsbedingungen", "tdm_signal"]
STATUS = {"behalten", "verworfen", "zurückgestellt"}
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")


def check_entry(entry: dict) -> None:
    """Sind alle Felder da, die Rechte ausgefüllt, das Abrufdatum genannt, Status und Grund gesetzt?"""
    rights = entry.get("rechte") or {}
    missing = [k for k in ENTRY_FIELDS if k not in entry] + [f"rechte.{k}" for k in RIGHTS_FIELDS if k not in rights]
    empty = [k for k in ("dokument", "nutzungsbedingungen", "tdm_signal")
             if k in rights and not (isinstance(rights[k], str) and rights[k].strip())]
    without_date = [] if DATE.search(rights.get("tdm_signal") or "") else ["tdm_signal"]
    status_ok = entry.get("status") in STATUS and bool(entry.get("grund"))
    ok = not (missing or empty or without_date) and status_ok
    print(f"{entry.get('doc_id')}: {'vollständig' if ok else 'unvollständig'}",
          *[f"| {t}: {v}" for t, v in [("fehlt", missing), ("leer", empty), ("ohne Abrufdatum", without_date)] if v],
          "" if status_ok else "| status oder grund fehlt")


# Aufgabe 4 · Split von Lab 4


def split_pairs(raw: pd.DataFrame, manifest: dict, reference: dict,
                pair_kind: Callable[[str, str], str]) -> tuple[dict[str, str], set[tuple[str, str]]]:
    """Seite jeder XML-Norm im Split von Lab 4 und die Zwillingspaare aus dem XML-Abzug (exakt und ab 0,75)."""
    norms4, _ = load()
    part = split(norms4)                                      # {"train": Zeilen, "val": ..., "test": ...}
    xml_ids = raw.loc[raw["quelle"] == "gii-xml", "doc_id"].iloc[:len(norms4)]    # Zeile i von load() = xml_ids[i]
    page = {d: k for k, rows in part.items() for d in xml_ids.iloc[rows]}
    source = raw.set_index("doc_id")["quelle"]
    pairs = {tuple(sorted((d, z["verarbeitung"]["dublette"]["von"]))) for d, z in manifest.items()
             if z["quelle"] == "gii-xml" and (z["verarbeitung"].get("dublette") or {}).get("art") == "exakt"}
    pairs |= {p for p, j in reference.items() if j >= 0.75 and pair_kind(*p) == "verschiedene Normen"
              and source[p[0]] == source[p[1]] == "gii-xml"}
    print(len(pairs), "Paare;", {k: len(v) for k, v in part.items()})
    return page, pairs
