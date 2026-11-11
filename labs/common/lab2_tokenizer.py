"""Bausteine für Lab 2 (Tokenizer trainieren und messen).

Korpus laden (Ergebnis von Lab 1 oder Ersatzweg), Holdout nach Gesetz, Kennzahlen, Komposita-Liste.
Die Tokenizer bauen die Notebooks selbst mit der Bibliothek `tokenizers`; `train_bpe` enthält
dieselben Zeilen wie die Demo, damit die Übung sie wiederverwenden kann.

Feste Ablage für Lab 3 (falls Lab 2 nicht fertig wird) neu bauen, im Ordner labs/:
    python -m common.lab2_tokenizer
Sie wird immer aus dem Ersatzweg trainiert, damit sie nicht vom Stand von Lab 1 abhängt.

Referenzmessung der Schrittzeiten für Übung, Aufgabe 2 (Rechner des Trainers, CPU), neu bauen:
    python -m common.lab2_tokenizer schrittzeit
"""
from __future__ import annotations

import collections
import json
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common.paths import ARTIFACTS_DIR, LAB_DIR, SEED

LAB1_CORPUS = ARTIFACTS_DIR / "lab1_korpus.parquet"
REFERENCE_DIR = LAB_DIR / "reference"
REFERENCE_TOKENIZER = REFERENCE_DIR / "lab2_tokenizer_v8000.json"
STEP_TIME_JSON = REFERENCE_DIR / "lab2_schrittzeit.json"
TOKENIZER_JSON = ARTIFACTS_DIR / "lab2_tokenizer.json"
TRAIN_BIN = ARTIFACTS_DIR / "lab2_train.bin"
VAL_BIN = ARTIFACTS_DIR / "lab2_val.bin"

# Ganze Gesetze im Holdout, mindestens eines je Rechtsgebiet, zusammen rund 9 % der Zeichen
HOLDOUT_LAWS = ("InsO", "OWiG 1968", "VwGO", "UStG 1980", "BetrVG", "KSchG")
SPECIAL_TOKENS = ["<pad>", "<s>", "</s>"]

_PLACEHOLDER = re.compile(r"^\([^()]*\)$")           # "(Inkrafttreten)", "(gegenstandslos)" ...
_WORD = re.compile(r"[A-Za-zÄÖÜäöüß]+")
_LINKING_ELEMENTS = ("", "s", "es", "n", "en", "e")


def format_de(x, decimals: int | None = None) -> str:
    """Zahl in deutscher Schreibweise: 12.345 oder 5,11."""
    if decimals is None and isinstance(x, int):
        return f"{x:,}".replace(",", ".")
    s = f"{x:,.{2 if decimals is None else decimals}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Kleine Bereinigung für den Ersatzweg: Platzhalter-Normen und exakte Dubletten entfernen."""
    text = df["text"].str.strip()
    df = df.assign(text=text)[~text.str.match(_PLACEHOLDER)]
    return df.drop_duplicates("text").reset_index(drop=True)


def load_corpus(verbose: bool = True) -> tuple[pd.DataFrame, str]:
    """Bereinigter Korpus aus Lab 1; fehlt er, der Gesetzeskorpus mit `clean`."""
    if LAB1_CORPUS.exists():
        df = pd.read_parquet(LAB1_CORPUS)
        if {"gesetz", "text"} <= set(df.columns):
            source = f"Lab 1 ({LAB1_CORPUS.name})"
        else:
            df, source = None, f"Ersatzweg ({LAB1_CORPUS.name} ohne Spalten gesetz/text)"
    else:
        df, source = None, "Ersatzweg (Lab 1 fehlt): Gesetzeskorpus, Platzhalter und Dubletten entfernt"
    if df is None:
        from common import corpus

        df = clean(corpus.load())
    df = df[df["text"].notna()].reset_index(drop=True)
    if verbose:
        print(f"Quelle: {source}, {format_de(len(df))} Normen")
    return df, source


def train_holdout_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training und Holdout; die Gesetze in HOLDOUT_LAWS liegen vollständig im Holdout."""
    in_holdout = df["gesetz"].isin(HOLDOUT_LAWS)
    return df[~in_holdout], df[in_holdout]


def as_text(df: pd.DataFrame) -> list[str]:
    """Eine Norm je Text: Bezeichnung und Überschrift in der ersten Zeile, darunter der Wortlaut
    (wie lab3_data.norm_text). Ohne die Spalten enbez/titel nur der Wortlaut."""
    if {"enbez", "titel"} <= set(df.columns):
        head = (df["enbez"].fillna("") + " " + df["titel"].fillna("")).str.strip()
        return (head + "\n" + df["text"]).tolist()
    return df["text"].tolist()


def train_bpe(texts, vocab_size: int, pre_tokenizer=None):
    """BPE wie in der Demo (NFC, Byte-Level, drei Spezialtokens). Liefert (Tokenizer, Sekunden).

    pre_tokenizer ersetzt den Pre-Tokenizer ByteLevel(add_prefix_space=False); Decoder bleibt ByteLevel.
    """
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    t0 = time.perf_counter()
    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizer or pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=SPECIAL_TOKENS,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, trainer)
    return tok, time.perf_counter() - t0


def ids_of(tok, texts) -> list[list[int]]:
    """Token-IDs ohne Spezialtokens, für tokenizers.Tokenizer und transformers-Tokenizer gleich."""
    texts = list(texts)
    if hasattr(tok, "encode_batch"):
        return [e.ids for e in tok.encode_batch(texts)]
    return tok(texts, add_special_tokens=False)["input_ids"]


def tokenizer_metrics(tok, texts) -> dict:
    """Tokens gesamt, Zeichen je Token und Tokens je Wort (Wort = durch Leerraum getrennt)."""
    texts = list(texts)
    n = sum(map(len, ids_of(tok, texts)))
    n_chars = sum(map(len, texts))
    words = sum(len(t.split()) for t in texts)
    return {"tokens": n, "zeichen_je_token": n_chars / n, "tokens_je_wort": n / words}


def compounds(train_texts, holdout_texts, min_length: int = 12, min_frequency: int = 2,
              with_split: bool = False):
    """Komposita aus dem Holdout, abgeleitet aus dem Korpus statt von Hand gewählt.

    Kandidaten: großgeschriebene Wörter des Holdouts mit mindestens `min_length` Buchstaben, die dort
    mindestens `min_frequency`-mal vorkommen. Ein Kandidat gilt als Kompositum, wenn er sich in zwei
    Nomen des Trainingsteils zerlegen lässt (je mindestens 4 Buchstaben, optional mit Fugenelement
    s, es, n, en, e), angelehnt an das häufigkeitsbasierte Zerlegen von Koehn und Knight (EACL 2003).
    Nomen: großgeschrieben mindestens fünfmal im Training und häufiger groß als klein.
    """
    count_train = collections.Counter(w for t in train_texts for w in _WORD.findall(t))
    count_holdout = collections.Counter(w for t in holdout_texts for w in _WORD.findall(t))

    def is_noun(w: str) -> bool:
        return len(w) >= 4 and w[0].isupper() and count_train[w] >= 5 and count_train[w.lower()] < count_train[w]

    def split_compound(w: str):
        for i in range(4, len(w) - 3):
            left, right_part = w[:i], w[i:].capitalize()
            for linker in _LINKING_ELEMENTS:
                if left.endswith(linker):
                    stem = left[: len(left) - len(linker)] if linker else left
                    if is_noun(stem) and is_noun(right_part):
                        return stem, linker, right_part
        return None

    result = []
    for w, n in sorted(count_holdout.items()):
        if len(w) >= min_length and w[0].isupper() and n >= min_frequency:
            z = split_compound(w)
            if z:
                result.append((w, z) if with_split else w)
    return result


def share_single_token(tok, words) -> float:
    """Anteil der Wörter, die mit Leerzeichen davor (wie mitten im Satz) genau ein Token ergeben."""
    ids = ids_of(tok, [" " + w for w in words])
    return sum(len(i) == 1 for i in ids) / len(ids)


def count_entries(tok, texts) -> np.ndarray:
    """Wie oft jede ID des Vokabulars in den Texten vorkommt (Index = Token-ID)."""
    ids = np.concatenate([np.asarray(i, dtype=np.int64) for i in ids_of(tok, texts)])
    return np.bincount(ids, minlength=tok.get_vocab_size())


def load_tokenizer_8000(train_texts=None):
    """Der 8.000er-Tokenizer aus der Demo; fehlt er, die feste Ablage; fehlt auch die, neu trainiert.

    Hat artifacts/lab2_tokenizer.json eine andere Größe (etwa nach einem Versuch mit vocab_size=16000 in der
    Demo), meldet die Funktion das und nimmt die feste Ablage.
    """
    from tokenizers import Tokenizer

    for path, source in [(TOKENIZER_JSON, "Demo (artifacts)"), (REFERENCE_TOKENIZER, "feste Ablage (reference)")]:
        if path.exists():
            tok = Tokenizer.from_file(str(path))
            if tok.get_vocab_size() == 8000:
                return tok, source
            print(f"{path.name} hat {format_de(tok.get_vocab_size())} statt 8.000 Einträge, wird übergangen")
    if train_texts is None:
        raise FileNotFoundError("Kein 8.000er-Tokenizer gefunden und keine Trainingstexte übergeben")
    tok, _ = train_bpe(train_texts, 8000)
    return tok, "neu trainiert"


def build_reference() -> Path:
    """Feste Ablage für Lab 3: BPE mit V = 8.000 auf dem Trainingsteil des Ersatzwegs."""
    from common import corpus

    train_df, _ = train_holdout_split(clean(corpus.load()))
    tok, sec = train_bpe(as_text(train_df), 8000)
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    tok.save(str(REFERENCE_TOKENIZER))
    print(f"{REFERENCE_TOKENIZER} geschrieben ({tok.get_vocab_size()} Einträge, {sec:.1f} s)")
    return REFERENCE_TOKENIZER


VOCAB_SIZES = (2000, 4000, 8000, 16000, 32000)


def step_time(vocab: int, batch: int = 16, seq_len: int = 256, steps: int = 6, warmup_steps: int = 2) -> float:
    """Sekunden je Trainingsschritt des Mini-GPT aus Lab 3 (d = 256, vier Blöcke, Weight Tying) auf der CPU.

    Ein Schritt wie in Lab 3: vorwärts, Cross-Entropy, rückwärts, Gradient Clipping, AdamW. Die Token-IDs sind
    zufällig; die Rechenzeit hängt nur von den Formen ab. Gemessen wird nach `warmup_steps` Schritten.
    """
    import torch
    import torch.nn.functional as F

    from common.lab3_model import MiniGPT

    torch.manual_seed(SEED)
    model = MiniGPT(vocab=vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    x, y = torch.randint(0, vocab, (2, batch, seq_len))
    for i in range(warmup_steps + steps):
        if i == warmup_steps:
            t0 = time.perf_counter()
        loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return (time.perf_counter() - t0) / steps


def load_step_time_reference() -> dict:
    """Referenzmessung des Trainers: {"sekunden": {V: s je Schritt}, "rechner": ..., ...}."""
    ref = json.loads(STEP_TIME_JSON.read_text())
    ref["sekunden"] = {int(v): s for v, s in ref["sekunden"].items()}
    return ref


def build_step_time_reference(repetitions: int = 5, steps: int = 8) -> Path:
    """Referenzmessung: je V der Median über `repetitions` Durchgänge, die Größen abwechselnd gemessen."""
    import torch

    t0 = time.perf_counter()
    run_logs = {v: [] for v in VOCAB_SIZES}
    for _ in range(repetitions):
        for v in VOCAB_SIZES:
            run_logs[v].append(step_time(v, steps=steps))
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        import subprocess

        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    ref = {
        "rechner": f"{cpu}, {platform.system()} {platform.machine()}, CPU, {torch.get_num_threads()} Threads",
        "torch": torch.__version__,
        "aufbau": f"MiniGPT d=256, 4 Blöcke, Batch 16 x 256 Tokens, AdamW; Median über {repetitions} Durchgänge "
                  f"mit je {steps} Schritten nach 2 zum Aufwärmen",
        "datum": time.strftime("%Y-%m-%d"),
        "sekunden": {str(v): float(np.median(s)) for v, s in run_logs.items()},
        "spanne": {str(v): [min(s), max(s)] for v, s in run_logs.items()},
    }
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    STEP_TIME_JSON.write_text(json.dumps(ref, indent=1, ensure_ascii=False) + "\n")
    print(f"{STEP_TIME_JSON} geschrieben ({time.perf_counter() - t0:.0f} s)")
    for v in VOCAB_SIZES:
        print(f"  V = {v:>6}: {1000 * ref['sekunden'][str(v)]:6.1f} ms je Schritt "
              f"(Spanne {1000 * min(run_logs[v]):.1f} bis {1000 * max(run_logs[v]):.1f})")
    return STEP_TIME_JSON


if __name__ == "__main__":
    if sys.argv[1:] == ["schrittzeit"]:
        build_step_time_reference()
    else:
        build_reference()
