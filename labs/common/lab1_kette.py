"""Filterkette aus der Lab-1-Demo als Modul: für die Übung und als Ersatzweg für Lab 2.

`clean(raw, until=...)` führt die Stufen der Demo in derselben Reihenfolge aus und liefert
(behaltene Dokumente, Manifest-Zeilen, Zählung je Stufe). Mit `until="exakt"` endet die Kette
vor den Beinahe-Dubletten; ohne `until` stellt sie am Ende Dokumente mit unbekannten Rechten zurück.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata

import pandas as pd
from datasketch import MinHash, MinHashLSH

from common.paths import ARTIFACTS_DIR, SEED

NAVIGATION = {"zurück", "weiter", "Nichtamtliches Inhaltsverzeichnis", "zum Seitenanfang", "Impressum",
              "Datenschutz", "Barrierefreiheitserklärung", "Feedback-Formular"}
SINGLE_NORM = re.compile(r"^.+ - Einzelnorm$")
FUNCTION_WORDS = set("der die das und ist nicht mit von den dem des sich wird werden oder auf für eine einer "
                       "einem zu nach bei durch als auch wenn zur zum im ein sind bis nur wie über aus kann hat".split())
LINK = re.compile(r"https?://\S+")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
IBAN = re.compile(r"\bDE\d{2}(?: ?\d{4}){4} ?\d{2}\b")
NUM_PERM, PARAMS, THRESHOLD = 112, (14, 8), 0.75
RIGHTS = {
    "gii-xml": {"dokument": "amtliches Werk, § 5 UrhG", "tdm_signal": "robots.txt 2026-10-27: User-agent: *, Disallow: leer"},
    "gii-xml-nachlauf": {"dokument": "amtliches Werk, § 5 UrhG", "tdm_signal": "robots.txt 2026-10-27: User-agent: *, Disallow: leer"},
    "gii-html": {"dokument": "amtliches Werk, § 5 UrhG", "tdm_signal": "robots.txt 2026-10-27: User-agent: *, Disallow: leer"},
    "intern-synthetisch": {"dokument": "interne Akten (synthetisch)", "tdm_signal": "entfällt"},
    "web-forum": {"dokument": "unbekannt", "tdm_signal": "unbekannt"},
}


def remove_navigation(text: str) -> tuple[str, int]:
    rows = text.split("\n")
    rest = [z for z in rows if z.strip() not in NAVIGATION and not SINGLE_NORM.match(z.strip())]
    return "\n".join(rest), len(rows) - len(rest)


def is_placeholder(text: str) -> bool:
    t = text.strip()
    return t == "-" or ("\n" not in t and t.startswith("(") and t.endswith(")") and len(t.split()) <= 8)


def language_score(text: str) -> float:
    words = re.findall(r"[a-zäöüß]+", text.lower())
    return sum(w in FUNCTION_WORDS for w in words) / max(1, len(words))


def duplicate_line_share(text: str) -> float:
    """FineWeb-Regel: Anteil der Zeichen in Zeilen, die im Dokument mehrfach vorkommen."""
    rows = [z.strip() for z in text.split("\n") if z.strip()]
    duplicate = {z for z in rows if rows.count(z) > 1}
    return sum(len(z) for z in rows if z in duplicate) / max(1, sum(len(z) for z in rows))


def foreign_or_ad(text: str, score: float) -> str | None:
    if duplicate_line_share(text) >= 0.1 or len(LINK.findall(text)) >= 2:
        return "werbung"
    if len(text.split()) >= 10 and score < 0.1:
        return "sprache"
    return None


def mask(text: str) -> tuple[str, dict]:
    n_mail = n_iban = 0
    if "@" in text:
        text, n_mail = EMAIL.subn("<EMAIL>", text)
    if "DE" in text:
        text, n_iban = IBAN.subn("<IBAN>", text)
    return text, {"EMAIL": n_mail, "IBAN": n_iban}


def dedup_key(text: str) -> str:
    norm = unicodedata.normalize("NFC", text).lower()
    norm = " ".join(norm.split())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def five_grams(text: str) -> set[str]:
    w = text.split()
    return {" ".join(w[i:i + 5]) for i in range(len(w) - 4)}


def signature(grams: set[str], num_perm: int = NUM_PERM) -> MinHash:
    m = MinHash(num_perm=num_perm, seed=SEED)
    m.update_batch([g.encode("utf-8") for g in grams])
    return m


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b)


def clean(raw: pd.DataFrame, until: str | None = None):
    """Filterkette der Demo. Liefert (df, manifest, counts); until="exakt" endet vor den Beinahe-Dubletten."""
    df = raw.copy()
    manifest = {r.doc_id: {"doc_id": r.doc_id, "quelle": r.quelle, "quelle_url": r.url, "abruf": r.abruf,
                           "rechte": {"dataset_lizenz": None, **RIGHTS[r.quelle], "nutzungsbedingungen": "unbekannt"},
                           "sha256": {"roh": hashlib.sha256(r.text.encode("utf-8")).hexdigest(), "text": None},
                           "verarbeitung": {}, "status": "behalten", "grund": None}
                for r in df.itertuples()}
    counts = []

    def discard(df, reasons, stage, changed=0):
        for d, g in zip(df["doc_id"], reasons):
            if g:
                manifest[d].update(status="verworfen", grund=g)
        mask_arr = reasons.notna().to_numpy()
        counts.append({"stufe": stage, "rein": len(df), "verworfen": int(mask_arr.sum()), "geaendert": changed})
        return df[~mask_arr].reset_index(drop=True)

    # 1 Boilerplate: Navigationszeilen weg, Platzhalter verwerfen
    fresh = df["text"].map(remove_navigation)
    df["text"] = fresh.str[0]
    for d, n in zip(df["doc_id"], fresh.str[1]):
        manifest[d]["verarbeitung"]["navigation_zeilen"] = int(n)
    df = discard(df, df["text"].map(lambda t: "platzhalter" if is_placeholder(t) else None),
                 "Boilerplate", changed=int((fresh.str[1] > 0).sum()))

    # 2 Sprache und Werbung
    scores = df["text"].map(language_score)
    for d, sc in zip(df["doc_id"], scores):
        manifest[d]["verarbeitung"]["sprache"] = round(sc, 3)
    df = discard(df, pd.Series([foreign_or_ad(t, sc) for t, sc in zip(df["text"], scores)]),
                 "Sprache und Werbung")

    # 3 PII maskieren
    fresh = df["text"].map(mask)
    df["text"] = fresh.str[0]
    for d, n in zip(df["doc_id"], fresh.str[1]):
        manifest[d]["verarbeitung"]["pii"] = n
    counts.append({"stufe": "PII", "rein": len(df), "verworfen": 0,
                     "geaendert": int(fresh.map(lambda x: sum(x[1].values()) > 0).sum())})

    # 4 Exakte Dubletten nach Normalisierung
    keys = df["text"].map(dedup_key)
    first_hit = {}
    reasons = []
    for d, k in zip(df["doc_id"], keys):
        manifest[d]["sha256"]["text"] = k
        if k in first_hit:
            manifest[d]["verarbeitung"]["dublette"] = {"art": "exakt", "von": first_hit[k]}
            reasons.append("dublette-exakt")
        else:
            first_hit[k] = d
            reasons.append(None)
    df = discard(df, pd.Series(reasons), "Exakte Dubletten")
    if until == "exakt":
        return df, manifest, counts

    # 5 Beinahe-Dubletten: MinHash + LSH, Kandidaten mit exaktem Jaccard bestätigen
    lsh = MinHashLSH(num_perm=NUM_PERM, params=PARAMS)
    grams = {}
    reasons = []
    for d, t in zip(df["doc_id"], df["text"]):
        g = five_grams(t)
        if not g:                                  # unter fünf Wörtern: nur exakt geprüft
            reasons.append(None)
            continue
        m = signature(g)
        hits = [(jaccard(g, grams[c]), c) for c in lsh.query(m)]
        hits = [x for x in hits if x[0] >= THRESHOLD]
        if hits:
            j, c = max(hits)
            manifest[d]["verarbeitung"]["dublette"] = {"art": "beinahe", "von": c, "jaccard": round(j, 3)}
            reasons.append("dublette-beinahe")
        else:
            lsh.insert(d, m)
            grams[d] = g
            reasons.append(None)
    df = discard(df, pd.Series(reasons), "Beinahe-Dubletten")

    # 6 Rechte: Dokumente mit unbekannten Rechten zurückstellen (nicht in den Trainingskorpus)
    unresolved = df["quelle"].map(lambda q: RIGHTS[q]["dokument"] == "unbekannt").to_numpy()
    for d in df.loc[unresolved, "doc_id"]:
        manifest[d].update(status="zurückgestellt", grund="rechte-ungeklaert")
    counts.append({"stufe": "Rechte", "rein": len(df), "verworfen": int(unresolved.sum()), "geaendert": 0})
    return df[~unresolved].reset_index(drop=True), manifest, counts


COLUMNS = ["doc_id", "quelle", "gesetz", "rechtsgebiet", "enbez", "titel", "text"]


def save(df: pd.DataFrame, manifest: dict, target=ARTIFACTS_DIR):
    target.mkdir(parents=True, exist_ok=True)
    df[COLUMNS].to_parquet(target / "lab1_korpus.parquet", index=False)
    with open(target / "lab1_manifest.jsonl", "w", encoding="utf-8") as f:
        for line in manifest.values():
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
    return target / "lab1_korpus.parquet", target / "lab1_manifest.jsonl"


def pairs_above(grams: dict[str, set], threshold: float) -> dict[tuple[str, str], float]:
    """Alle Paare mit exaktem Jaccard >= threshold (Prefix-Filter, ohne Näherung). Referenz für die LSH-Messung."""
    frequent = {}
    for g in grams.values():
        for x in g:
            frequent[x] = frequent.get(x, 0) + 1
    ordered = {d: sorted(g, key=lambda x: (frequent[x], x)) for d, g in grams.items() if g}
    index, pairs = {}, {}
    for d in sorted(ordered, key=lambda d: len(ordered[d])):
        g = ordered[d]
        prefix = len(g) - math.ceil(threshold * len(g)) + 1
        candidate_ids = set()
        for x in g[:prefix]:
            candidate_ids.update(index.get(x, ()))
            index.setdefault(x, []).append(d)
        for c in candidate_ids:
            if len(ordered[c]) >= threshold * len(g):
                j = jaccard(grams[d], grams[c])
                if j >= threshold:
                    pairs[tuple(sorted((c, d)))] = j
    return pairs

