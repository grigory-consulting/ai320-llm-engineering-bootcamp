"""Lab 4: Split, Merkmale und Metriken für die Zuordnung Paragraph -> Rechtsgebiet.

Plumbing für Demo und Übung: fester Gruppensplit nach Gesetz, vorbereitete
Embeddings, Metriken mit Intervall. Die Modelle selbst stehen sichtbar im Notebook.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from common import corpus
from common.paths import ARTIFACTS_DIR, LAB_DIR, MODELS_DIR, SEED

REFERENCE_DIR = LAB_DIR / "reference"
CACHE_DIR = ARTIFACTS_DIR / "lab4"
DBERT = MODELS_DIR / "distilbert-german"
E5 = MODELS_DIR / "e5-small"


def load() -> tuple[pd.DataFrame, list[str]]:
    """Gesetzeskorpus und die fünf Rechtsgebiete (alphabetisch, Index = Label-Nummer)."""
    df = corpus.load().reset_index(drop=True)
    classes = sorted(df.rechtsgebiet.unique())
    df["y"] = df.rechtsgebiet.map({k: i for i, k in enumerate(classes)})
    return df, classes


def group_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = SEED) -> list[np.ndarray]:
    """Fünf Folds aus ganzen Gesetzen, Klassenanteile möglichst ähnlich (Indizes der Test-Zeilen)."""
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [te for _, te in sgkf.split(df, df.rechtsgebiet, df.gesetz)]


def split(df: pd.DataFrame, seed: int = SEED) -> dict[str, np.ndarray]:
    """Fester Split nach Gesetz: je Rechtsgebiet ein ganzes Gesetz im Test, eines in der Validierung.

    Test ist Fold 1 von group_folds(df), Validierung Fold 2 derselben Prozedur auf dem Rest.
    Beide Folds enthalten alle fünf Rechtsgebiete.
    """
    test = group_folds(df, seed=seed)[1]
    rest = np.setdiff1d(np.arange(len(df)), test)
    val = rest[group_folds(df.iloc[rest], seed=seed)[2]]
    train = np.setdiff1d(rest, val)
    return {"train": train, "val": val, "test": test}


def metrics(y_true, y_pred) -> dict[str, float]:
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro"))}


def e5_embeddings(texts: list[str], batch_size: int = 32, device: str = "cpu") -> np.ndarray:
    """Normierte e5-small-Embeddings ("query: "-Präfix, max. 256 Tokens) für alle Paragraphen.

    Liegt die vorbereitete Datei reference/lab4_e5_small.npy vor und passt zur Zahl der Texte,
    wird sie geladen; sonst wird gerechnet (auf der CPU einige Minuten) und unter artifacts/lab4/ abgelegt.
    """
    for file in (REFERENCE_DIR / "lab4_e5_small.npy", CACHE_DIR / "lab4_e5_small.npy"):
        if file.exists():
            E = np.load(file)
            if len(E) == len(texts):
                return E.astype(np.float32)
    from sentence_transformers import SentenceTransformer

    st = SentenceTransformer(str(E5), device=device)
    st.max_seq_length = 256
    t = time.time()
    E = st.encode(["query: " + x for x in texts], batch_size=batch_size,
                  normalize_embeddings=True, show_progress_bar=False)
    print(f"e5-small: {len(texts)} Paragraphen in {time.time() - t:.0f} s eingebettet")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_DIR / "lab4_e5_small.npy", E.astype(np.float16))
    return E.astype(np.float16).astype(np.float32)   # gleiche Genauigkeit wie die Ablage


def prepared(name: str):
    """Lädt eine vorbereitete Ablage aus labs/reference/ (json oder npz)."""
    path = REFERENCE_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"{path} fehlt. Neu erzeugen: im Ordner labs/ `python -m common.lab4_prepare` (CPU, rund 3 h)")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return dict(np.load(path, allow_pickle=False))


def file_size_mb(path: Path) -> float:
    path = Path(path)
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) / 1e6
    return path.stat().st_size / 1e6


def truncate(text: str, n_chars: int = 200) -> str:
    """Die ersten `n_chars` Zeichen, am letzten Leerzeichen davor abgeschnitten (wie eine Textvorschau)."""
    if len(text) <= n_chars:
        return text
    head = text[:n_chars]
    return head[: head.rfind(" ")] if " " in head else head


def macro_f1_fast(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 5) -> float:
    """Macro-F1 über Zählungen (gleiches Ergebnis wie sklearn, schnell genug für 10 000 Bootstrap-Ziehungen)."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(n_classes), where=denominator > 0)
    return float(f1.mean())


def tfidf_vectorizer(analyzer: str = "char_wb"):
    """TF-IDF wie in Demo und Übung: Zeichen-3- bis 5-Gramme (char_wb) oder Wort-1- bis 2-Gramme (word)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    if analyzer == "char_wb":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
    return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def tfidf_tuned(X_tr, y_tr, X_va, y_va):
    """TF-IDF + LR, auf der Validierung eingestellt wie in der Demo: 12 Varianten aus Analyzer (char_wb, word),
    Klassengewichten (keine, balanced) und C (1, 10, 100), gewählt nach Macro-F1 auf der Validierung.

    Der Vektorisierer wird je Analyzer einmal angepasst. Rückgabe: (beste Variante als dict mit trainierter
    Pipeline unter "modell", alle Varianten als Liste von dicts ohne Modell).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline

    param_grid = []
    for analyzer in ("char_wb", "word"):
        vec = tfidf_vectorizer(analyzer).fit(X_tr)
        V_tr, V_va = vec.transform(X_tr), vec.transform(X_va)
        for cw in (None, "balanced"):
            for C in (1.0, 10.0, 100.0):
                lr = LogisticRegression(C=C, class_weight=cw, max_iter=3000).fit(V_tr, y_tr)
                param_grid.append({"analyzer": analyzer, "C": C, "class_weight": cw or "keine",
                               "val_macro_f1": metrics(y_va, lr.predict(V_va))["macro_f1"],
                               "modell": make_pipeline(vec, lr)})
    top = max(param_grid, key=lambda g: g["val_macro_f1"])
    return top, [{k: v for k, v in g.items() if k != "modell"} for g in param_grid]


def stratified_subset(df: pd.DataFrame, idx: np.ndarray, n: int, seed: int = SEED) -> np.ndarray:
    """Genau n Zeilen aus idx, geschichtet nach Rechtsgebiet gezogen (Klassenanteile wie in idx)."""
    rng = np.random.default_rng(seed)
    classes, count = np.unique(df.y.values[idx], return_counts=True)
    expected = n * count / len(idx)
    per_class_counts = np.floor(expected).astype(int)
    per_class_counts[np.argsort(per_class_counts - expected)[: n - per_class_counts.sum()]] += 1  # Rest nach größtem Nachkommaanteil
    choice = [rng.choice(idx[df.y.values[idx] == k], size=m, replace=False) for k, m in zip(classes, per_class_counts)]
    return np.sort(np.concatenate(choice))
