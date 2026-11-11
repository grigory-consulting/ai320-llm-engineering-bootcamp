"""Bausteine für Lab 5 (Mini-RAG): Chunks, Index, Bewertung, Cache für LLM-Antworten.

Die Demo zeigt jeden Schritt einmal sichtbar im Notebook. Dieses Modul enthält
dieselben Schritte noch einmal, damit die Übung sie für andere Chunkings, andere
k oder mit Re-Ranking aufrufen kann, ohne die Demo zu kopieren.

Vorbereitete Ablagen in labs/reference/ (mitgeliefert):
    lab5_fragen.json          Fragenset mit Gold-Normen und Herkunft
    lab5_emb_<chunking>.npy   e5-small-Vektoren des Index (float16)
    lab5_emb_meta.json        Anzahl und Prüfsumme der Chunk-IDs je Ablage
    lab5_antworten.json       Antworten und Judge-Urteile des LLM (Cache)
    lab5_kontrollurteile.json Kontrollurteile von Hand zu den inhaltlichen Antworten der Demo
Fehlt eine Ablage oder passt sie nicht, rechnet das Modul neu und schreibt nach
labs/artifacts/ (Ersatzweg).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd

from common.paths import ARTIFACTS_DIR, LAB_DIR, MODELS_DIR, SEED

REFERENCE_DIR = LAB_DIR / "reference"
QUESTIONS_JSON = REFERENCE_DIR / "lab5_fragen.json"
CONTROL_JSON = REFERENCE_DIR / "lab5_kontrollurteile.json"
INDEX_LAWS = ["BGB", "StGB", "GG", "UrhG", "BetrVG", "KSchG", "ArbZG", "BDSG 2018"]
MAX_CHARS = 1500    # ab dieser Länge wird ein Paragraph in Absätze geteilt (Chunking "gemischt")


def _write_target() -> Path:
    """Neue Ablagen gehen nach artifacts/, beim Trainerbau mit LAB5_WRITE_REFERENCE=1 nach reference/."""
    target = REFERENCE_DIR if os.environ.get("LAB5_WRITE_REFERENCE") == "1" else ARTIFACTS_DIR
    target.mkdir(parents=True, exist_ok=True)
    return target


# --------------------------------------------------------------------------- Chunks

def law_short(law_slug: str) -> str:
    """'BGB' -> 'bgb', 'BDSG 2018' -> 'bdsg', 'SGB 5' -> 'sgb5'."""
    return re.sub(r"\s+", "", re.sub(r"\s+\d{4}$", "", law_slug)).lower()


def norm_id(law_slug: str, designation: str) -> str:
    """Stabile ID einer Norm: 'BGB', '§ 437' -> 'bgb#437'; 'GG', 'Art 10' -> 'gg#10'."""
    return law_short(law_slug) + "#" + re.sub(r"^(§|Art)\s*", "", designation)


def norm_of(chunk_id: str) -> str:
    """Chunk-ID auf die Norm zurückführen: 'bgb#573-2' -> 'bgb#573'."""
    return chunk_id.split("-")[0]


def index_norms(df: pd.DataFrame) -> pd.DataFrame:
    """Teilmenge des Gesetzeskorpus, die indexiert wird."""
    return df[df.gesetz.isin(INDEX_LAWS)].reset_index(drop=True)


def chunks(df: pd.DataFrame, mode: str = "gemischt", max_chars: int = MAX_CHARS) -> pd.DataFrame:
    """Chunks mit ID, Norm-ID und Text (Kopfzeile mit Gesetz, Paragraph und Titel vorangestellt).

    mode="paragraph": ein Chunk je Norm.
    mode="gemischt":  ein Chunk je Norm, Normen über max_chars je Absatz.
    mode="absatz":    jeder Absatz ein eigener Chunk.
    """
    rows = []
    for r in df.itertuples():
        nid = norm_id(r.gesetz, r.enbez)
        head = f"{r.gesetz} {r.enbez} {r.titel}".strip()
        paragraphs = r.text.split("\n")
        whole = mode == "paragraph" or len(paragraphs) == 1 or (mode == "gemischt" and len(r.text) <= max_chars)
        pieces = [(nid, r.text)] if whole else [(f"{nid}-{i}", a) for i, a in enumerate(paragraphs, 1)]
        for cid, text in pieces:
            rows.append({"id": cid, "norm": nid, "text": f"{head}\n{text}", "builddate": r.builddate})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- Embeddings

def e5_model(device: str = "cpu"):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(str(MODELS_DIR / "e5-small"), device=device)


def _chunk_hash(ch: pd.DataFrame) -> str:
    return hashlib.sha1("\n".join(ch.id + "\t" + ch.text).encode()).hexdigest()[:16]


def embeddings(name: str, ch: pd.DataFrame, compute: Callable[[], np.ndarray]) -> np.ndarray:
    """Lädt die Chunk-Vektoren aus reference/ oder artifacts/; sonst compute() und ablegen.

    Die Ablage gilt nur, wenn die Prüfsumme über Chunk-IDs und Chunk-Texte passt
    (eine neue Fassung des Korpus erzwingt also neues Einbetten).
    """
    h = _chunk_hash(ch)
    for folder in (REFERENCE_DIR, ARTIFACTS_DIR):
        npy, meta = folder / f"lab5_emb_{name}.npy", folder / "lab5_emb_meta.json"
        if npy.exists() and meta.exists():
            m = json.loads(meta.read_text()).get(name, {})
            if m.get("chunks") == h:
                print(f"Vektoren für '{name}' geladen (vorbereitet, {folder.name}/{npy.name}).")
                return np.load(npy).astype(np.float32)
    print(f"Keine passende Ablage für '{name}': rechne {len(ch)} Chunks neu (auf der CPU einige Minuten).")
    t = time.perf_counter()
    E = np.asarray(compute(), dtype=np.float32)
    target = _write_target()
    np.save(target / f"lab5_emb_{name}.npy", E.astype(np.float16))
    meta = target / "lab5_emb_meta.json"
    m = json.loads(meta.read_text()) if meta.exists() else {}
    m[name] = {"chunks": h, "n": len(ch), "modell": "intfloat/multilingual-e5-small", "praefix": "passage: "}
    meta.write_text(json.dumps(m, indent=1))
    print(f"  fertig in {time.perf_counter() - t:.0f} s, abgelegt in {target.name}/")
    return E


# --------------------------------------------------------------------------- Retrieval

def tokens(text: str) -> list[str]:
    """Tokenisierung für BM25: Kleinschreibung, Wortzeichen (Satzzeichen fallen weg)."""
    return re.findall(r"\w+", text.lower())


def rrf(*lists: list[str], k: int = 60) -> list[str]:
    score: dict[str, float] = {}
    for ranking in lists:
        for position, doc_id in enumerate(ranking, start=1):
            score[doc_id] = score.get(doc_id, 0.0) + 1 / (k + position)
    return sorted(score, key=score.get, reverse=True)


class Index:
    """Dichte Suche, BM25 und hybride Suche über einer Chunk-Tabelle."""

    def __init__(self, ch: pd.DataFrame, E: np.ndarray, model):
        from rank_bm25 import BM25Okapi

        self.ch, self.E, self.model = ch, E, model
        self.ids = ch.id.tolist()
        self.text_of = dict(zip(ch.id, ch.text))
        self.bm25 = BM25Okapi([tokens(t) for t in ch.text])
        self._q: dict[str, np.ndarray] = {}

    def qvec(self, question: str) -> np.ndarray:
        if question not in self._q:
            self._q[question] = self.model.encode(["query: " + question], normalize_embeddings=True)[0]
        return self._q[question]

    def dense(self, question: str, k: int = 50) -> list[str]:
        scores = self.E @ self.qvec(question)
        return [self.ids[i] for i in np.argsort(-scores, kind="stable")[:k]]

    def bm25_search(self, question: str, k: int = 50) -> list[str]:
        scores = self.bm25.get_scores(tokens(question))
        return [self.ids[i] for i in np.argsort(-scores, kind="stable")[:k]]

    def hybrid(self, question: str, k: int = 50, rrf_k: int = 60) -> list[str]:
        return rrf(self.dense(question, 50), self.bm25_search(question, 50), k=rrf_k)[:k]


def build_index(mode: str, df_index: pd.DataFrame, model) -> Index:
    """Chunks im gewünschten Modus, Vektoren aus der Ablage (oder neu), BM25 dazu."""
    ch = chunks(df_index, mode)
    E = embeddings(mode, ch,
                   lambda: model.encode(["passage: " + t for t in ch.text], batch_size=32,
                                         normalize_embeddings=True, show_progress_bar=True))
    return Index(ch, E, model)


# --------------------------------------------------------------------------- Bewertung

def load_questions() -> list[dict]:
    return json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))["fragen"]


def rank(ranking: list[str], gold: list[str]) -> int | None:
    """Position (ab 1) des ersten Chunks, der zu einer Gold-Norm gehört; None, wenn keiner dabei ist."""
    for i, cid in enumerate(ranking, start=1):
        if norm_of(cid) in gold:
            return i
    return None


def recall_at_k(ranking: list[str], gold: list[str], k: int) -> float:
    found = {norm_of(c) for c in ranking[:k]} & set(gold)
    return len(found) / len(gold)


def evaluate(lists: dict[str, list[str]], questions: list[dict], ks=(1, 5, 10, 50)) -> pd.DataFrame:
    """Je Frage: Recall@k und Reciprocal Rank. lists: Frage-ID -> Trefferliste."""
    rows = []
    for f in questions:
        ranking = lists[f["id"]]
        r = rank(ranking, f["gold"])
        line = {"id": f["id"], "typ": f["typ"], "rr": 1 / r if r else 0.0}
        line.update({f"R@{k}": recall_at_k(ranking, f["gold"], k) for k in ks})
        rows.append(line)
    return pd.DataFrame(rows).set_index("id")


def paired(a: pd.Series, b: pd.Series, n: int = 2000) -> str:
    """Gepaarter Vergleich zweier Varianten auf denselben Fragen: Mittelwerte, besser/schlechter, Bootstrap-Intervall."""
    d = (b - a).to_numpy()
    rng = np.random.default_rng(SEED)
    boot = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return (f"{a.mean():.3f} -> {b.mean():.3f} (Differenz {d.mean():+.3f}, 95-%-Intervall [{lo:+.3f}, {hi:+.3f}]); "
            f"besser bei {(d > 0).sum()}, schlechter bei {(d < 0).sum()} von {len(d)} Fragen")


# --------------------------------------------------------------------------- Generator-LLM

def llm_client():
    """(client, model). Ohne Schlüssel: client=None und das Modell, von dem der Cache stammt.

    Dann kommen alle Antworten aus dem Cache; was dort fehlt, ersetzt ein Platzhalter mit Meldung
    (kein Absturz). Fehlt nur der Modellname in labs/.env, gilt ebenfalls das Modell des Caches.
    """
    global _CACHE_ONLY
    from common import llm

    try:
        cl, model = llm.client()
    except RuntimeError as e:
        _CACHE_ONLY = True
        model = cache_model()
        print(f"Hinweis: {e}. Ohne Schlüssel kommen alle Antworten aus dem Cache (Modell {model or 'unbekannt'}); "
              "was dort fehlt, ersetzt ein Platzhalter.")
        return None, model
    _CACHE_ONLY = False
    if not model:
        model = cache_model()
        print(f"Hinweis: LLM_MODEL fehlt in labs/.env; verwendet wird das Modell des Caches ({model}).")
    return cl, model


NOT_IN_SOURCES = "Nicht in den Quellen"
_CACHE: dict | None = None
_CACHE_ONLY = False                                  # True, wenn llm_client() keinen Schlüssel gefunden hat
_BALANCE = {"cache": 0, "neu": 0, "platzhalter": 0}  # Antworten dieser Sitzung nach Herkunft


def _cache() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = {}
        for folder in (REFERENCE_DIR, ARTIFACTS_DIR):
            file = folder / "lab5_antworten.json"
            if file.exists():
                _CACHE.update(json.loads(file.read_text(encoding="utf-8")))
    return _CACHE


def cache_model() -> str:
    """Modell, von dem die vorbereiteten Antworten stammen (das häufigste im Cache)."""
    models = [e.get("modell", "") for e in _cache().values() if e.get("modell")]
    return max(set(models), key=models.count) if models else ""


def llm_skipped() -> bool:
    """True mit LAB5_SKIP_LLM=1: fehlende Antworten werden nicht geholt, sondern durch einen Platzhalter ersetzt."""
    return os.environ.get("LAB5_SKIP_LLM") == "1"


def cached(model: str, messages: list[dict], call: Callable[[], str], **param) -> str:
    """Antwort aus dem Cache oder über call() holen und ablegen (Schlüssel: Modell, Parameter, Nachrichten).

    Ohne Schlüssel (siehe llm_client) oder mit LAB5_SKIP_LLM=1 wird call() nie aufgerufen: Was nicht im Cache
    steht, ersetzt ein Platzhalter, und die Zelle meldet das.
    """
    key = hashlib.sha1(json.dumps([model, param, messages], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    c = _cache()
    if key in c:
        _BALANCE["cache"] += 1
        return c[key]["antwort"]
    if llm_skipped() or _CACHE_ONLY:
        _BALANCE["platzhalter"] += 1
        reason = "LAB5_SKIP_LLM=1" if llm_skipped() else "kein Schlüssel"
        question = messages[-1]["content"].rsplit("Frage: ", 1)[-1].split("\n")[0][:70]
        print(f"Nicht im Cache ({reason}): Platzhalter statt Antwort zu {question!r}")
        return '{"aussagen": []}' if param.get("json") else f"{NOT_IN_SOURCES} (Platzhalter, nicht im Cache)"
    _BALANCE["neu"] += 1
    c[key] = {"modell": model, "frage": messages[-1]["content"][-300:], "antwort": call()}
    file = _write_target() / "lab5_antworten.json"
    old = json.loads(file.read_text(encoding="utf-8")) if file.exists() else {}
    old[key] = c[key]
    file.write_text(json.dumps(old, ensure_ascii=False, indent=1), encoding="utf-8")
    return c[key]["antwort"]


def cache_stats() -> str:
    """Herkunft der LLM-Antworten in dieser Sitzung (Cache, neue Aufrufe, Platzhalter)."""
    b = _BALANCE
    return f"LLM-Antworten in dieser Sitzung: {b['cache']} aus dem Cache, {b['neu']} neue Aufrufe, {b['platzhalter']} Platzhalter"


def cited_ids(answer: str) -> list[str]:
    """IDs in eckigen Klammern, mehrere in einer Klammer an Komma oder Semikolon getrennt."""
    ids = []
    for group in re.findall(r"\[([^\]]+)\]", answer):
        ids += [t.strip() for t in re.split(r"[,;]", group) if t.strip()]
    return ids


def is_non_answer(answer: str) -> bool:
    """Nicht-Antwort: der Satz "Nicht in den Quellen" steht in der Antwort, und sie zitiert keine Quelle."""
    return NOT_IN_SOURCES.lower() in answer.lower() and not cited_ids(answer)


VERDICTS = ("gestützt", "nicht gestützt", "widersprochen")


def fully_supported(claims: list[dict]) -> bool:
    """True, wenn der LLM-Judge Aussagen geliefert hat und jede davon 'gestützt' ist (fremde Etiketten zählen als nicht gestützt)."""
    return bool(claims) and all(a.get("urteil") == "gestützt" for a in claims)


def load_control_verdicts() -> dict:
    """Kontrollurteile zu den inhaltlichen Antworten der Demo (reference/lab5_kontrollurteile.json)."""
    return json.loads(CONTROL_JSON.read_text(encoding="utf-8"))


def share_supported(claims: list[dict]) -> float:
    """Anteil 'gestützt' unter den Aussagen mit einem der drei Urteile (andere Etiketten zählen nicht)."""
    u = [a["urteil"] for a in claims if a.get("urteil") in VERDICTS]
    return sum(x == "gestützt" for x in u) / len(u) if u else float("nan")


# --------------------------------------------------------------------------- Compose, Generate, Judge
# Wortgleich mit der Demo, damit die Übung dieselben Cache-Einträge trifft.

SYSTEM = ("Beantworten Sie die Frage nur aus den Quellen. "
          "Belegen Sie jede Aussage mit der id der Quelle in eckigen Klammern. "
          "Steht die Antwort nicht in den Quellen, antworten Sie: Nicht in den Quellen.")
JUDGE = ("Sie prüfen eine Antwort gegen Quellen. Zerlegen Sie die Antwort in einzelne Aussagen. "
           "Urteilen Sie je Aussage nur anhand der Quellen, nicht nach Ihrem eigenen Wissen: "
           "'gestützt', 'nicht gestützt' oder 'widersprochen'. Die Aussage 'Nicht in den Quellen' werten Sie nicht. "
           'Antworten Sie als JSON: {"aussagen": [{"aussage": "...", "urteil": "..."}]}')
MAX_PROMPT_CHARS = 2500


def sources_block(hits: list[str], text_of: dict[str, str]) -> str:
    return "\n\n".join(f'<quelle id="{d}">\n{text_of[d][:MAX_PROMPT_CHARS]}\n</quelle>' for d in hits)


def compose(question: str, hits: list[str], text_of: dict[str, str], system: str = SYSTEM) -> list[dict]:
    return [{"role": "system", "content": system},
            {"role": "user", "content": f"{sources_block(hits, text_of)}\n\nFrage: {question}"}]


def generate(cl, model: str, messages: list[dict]) -> str:
    return cached(model, messages, lambda: cl.chat.completions.create(
        model=model, messages=messages, temperature=0).choices[0].message.content, temperature=0)


def judge(cl, model: str, question: str, context: list[str], answer: str, text_of: dict[str, str]) -> list[dict]:
    """Aussagen der Antwort mit Urteil ('gestützt', 'nicht gestützt', 'widersprochen') gegen den Kontext."""
    messages = [{"role": "system", "content": JUDGE},
                {"role": "user", "content": f"{sources_block(context, text_of)}\n\nFrage: {question}\n\nAntwort:\n{answer}"}]
    raw = cached(model, messages, lambda: cl.chat.completions.create(
        model=model, messages=messages, temperature=0, response_format={"type": "json_object"}
    ).choices[0].message.content, temperature=0, json=True)
    return json.loads(raw)["aussagen"]
