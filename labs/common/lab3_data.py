"""Token-Strom für Lab 3: Train und Holdout als uint16-Dateien, dazu der passende Tokenizer.

Reihenfolge der Quellen:
1. Ergebnis von Lab 2: artifacts/lab2_train.bin, artifacts/lab2_val.bin, artifacts/lab2_tokenizer.json
   (nur mit genau 8.000 Einträgen im Tokenizer).
2. Ersatzweg (fehlt eine der drei Dateien oder passt die Größe nicht): feste Tokenizer-Ablage reference/lab2_tokenizer_v8000.json
   (fehlt auch die, ein eigener BPE-Tokenizer nach dem Rezept aus Teil III), Gesetzeskorpus selbst kodiert
   nach artifacts/lab3_train.bin und artifacts/lab3_val.bin.
Split wie in Lab 2: ganze Gesetze im Holdout (HOLDOUT_LAWS), eine Norm je Text mit Bezeichnung und Überschrift
in der ersten Zeile, Normen getrennt durch </s>. Die vorbereiteten Läufe in reference/lab3_* nutzen immer den Ersatzweg
mit einer eingefrorenen Kopie des Tokenizers (reference/lab3_tokenizer.json).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common.paths import ARTIFACTS_DIR, LAB_DIR, NORMS_PARQUET

REFERENCE_DIR = LAB_DIR / "reference"
LAB2_TRAIN = ARTIFACTS_DIR / "lab2_train.bin"
LAB2_VAL = ARTIFACTS_DIR / "lab2_val.bin"
LAB2_TOKENIZER = ARTIFACTS_DIR / "lab2_tokenizer.json"
LAB2_REFERENCE = REFERENCE_DIR / "lab2_tokenizer_v8000.json"
FALLBACK_TOKENIZER = ARTIFACTS_DIR / "lab3_tokenizer_ersatz.json"
OWN_TRAIN = ARTIFACTS_DIR / "lab3_train.bin"
OWN_VAL = ARTIFACTS_DIR / "lab3_val.bin"
PREPARED_TOKENIZER = REFERENCE_DIR / "lab3_tokenizer.json"
VOCAB = 8000                                         # Zeilen im Embedding des Mini-GPT

# wie common/lab2_tokenizer.py: ganze Gesetze im Holdout, mindestens eines je Rechtsgebiet
HOLDOUT_LAWS = ("InsO", "OWiG 1968", "VwGO", "UStG 1980", "BetrVG", "KSchG")
_PLACEHOLDER = re.compile(r"^\([^()]*\)$")           # "(Inkrafttreten)", "(gegenstandslos)" ...


@dataclass
class Data:
    train: np.ndarray          # np.memmap, uint16
    val: np.ndarray
    tokenizer_path: Path
    tokenizer_sha: str
    source: str
    train_path: Path = None
    val_path: Path = None


def format_de(n: int) -> str:
    """Ganzzahl mit Tausenderpunkt."""
    return f"{n:,}".replace(",", ".")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norms():
    """Gesetzeskorpus ohne Platzhalter-Normen und exakte Dubletten (wie der Ersatzweg von Lab 2)."""
    from common import corpus

    df = corpus.load()
    text = df["text"].str.strip()
    df = df.assign(text=text)[~text.str.match(_PLACEHOLDER)]
    return df.drop_duplicates("text").reset_index(drop=True)


def split_norms(df):
    in_holdout = df["gesetz"].isin(HOLDOUT_LAWS)
    return df[~in_holdout], df[in_holdout]


def as_text(df) -> list[str]:
    head = (df["enbez"].fillna("") + " " + df["titel"].fillna("")).str.strip()
    return (head + "\n" + df["text"]).tolist()


def train_fallback_tokenizer(texts, path: Path) -> Path:
    """BPE mit 8.000 Einträgen nach der Folie in Teil III (NFC, Byte-Level, drei Spezialtokens)."""
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE())
    tok.normalizer = normalizers.NFC()
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=8000, min_frequency=2, show_progress=False,
                                  special_tokens=["<pad>", "<s>", "</s>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(str(path))
    return path


def encode(tok, texts) -> np.ndarray:
    eos = tok.token_to_id("</s>")
    ids = []
    for enc in tok.encode_batch(list(texts)):
        ids += enc.ids + [eos]
    return np.array(ids, dtype=np.uint16)


def _memmaps(train_path, val_path, tok_path, source) -> Data:
    return Data(train=np.memmap(train_path, dtype=np.uint16, mode="r"),
                 val=np.memmap(val_path, dtype=np.uint16, mode="r"),
                 tokenizer_path=tok_path, tokenizer_sha=sha256(tok_path), source=source,
                 train_path=train_path, val_path=val_path)


def fallback_tokenizer() -> Path:
    if LAB2_REFERENCE.exists():
        return LAB2_REFERENCE
    if not FALLBACK_TOKENIZER.exists():
        train_fallback_tokenizer(as_text(split_norms(norms())[0]), FALLBACK_TOKENIZER)
    return FALLBACK_TOKENIZER


def vocab_size(tok_path: Path) -> int:
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(tok_path)).get_vocab_size()


def load(verbose: bool = True) -> Data:
    """Train- und Holdout-Tokens als memmap plus Tokenizer-Pfad und -Hash.

    Die Dateien aus Lab 2 gelten nur, wenn ihr Tokenizer genau VOCAB Einträge hat (das Mini-GPT hat ein Embedding
    mit VOCAB Zeilen). Sonst nimmt die Funktion den Ersatzweg und sagt das.
    """
    d = None
    if LAB2_TRAIN.exists() and LAB2_VAL.exists() and LAB2_TOKENIZER.exists():
        v = vocab_size(LAB2_TOKENIZER)
        if v == VOCAB:
            d = _memmaps(LAB2_TRAIN, LAB2_VAL, LAB2_TOKENIZER, "Lab 2 (artifacts/lab2_*)")
        else:
            print(f"Hinweis: artifacts/lab2_tokenizer.json hat {format_de(v)} Einträge, das Modell erwartet {format_de(VOCAB)}. "
                  f"Lab 3 nimmt deshalb den Ersatzweg mit der festen Tokenizer-Ablage.")
    if d is None:
        tok_path = fallback_tokenizer()
        name = "feste Ablage aus Lab 2" if tok_path == LAB2_REFERENCE else "eigener BPE-Tokenizer (Rezept Teil III)"
        d = load_own(tok_path, f"Ersatzweg: {name}, Gesetzeskorpus selbst kodiert")
    if verbose:
        print(f"Daten: {d.source}")
        print(f"  Train {format_de(len(d.train))} Tokens, Holdout {format_de(len(d.val))} Tokens, Vokabular {format_de(vocab_size(d.tokenizer_path))}")
    return d


def _stamp(tok_path: Path) -> str:
    return f"tokenizer={sha256(tok_path)} korpus={sha256(NORMS_PARQUET)} holdout={','.join(HOLDOUT_LAWS)}"


def load_own(tok_path: Path, source: str = "Ersatzweg") -> Data:
    """Ersatzweg mit einem bestimmten Tokenizer; kodiert neu, wenn sich Tokenizer oder Korpus geändert haben."""
    stamp = ARTIFACTS_DIR / "lab3_data.sha"
    if not (OWN_TRAIN.exists() and OWN_VAL.exists() and stamp.exists() and stamp.read_text() == _stamp(tok_path)):
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tok_path))
        train_df, val_df = split_norms(norms())
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        encode(tok, as_text(train_df)).tofile(OWN_TRAIN)
        encode(tok, as_text(val_df)).tofile(OWN_VAL)
        stamp.write_text(_stamp(tok_path))
    return _memmaps(OWN_TRAIN, OWN_VAL, tok_path, source)


def holdout_for(tok_path: Path) -> np.ndarray:
    """Holdout-Gesetze des Ersatzwegs, mit einem bestimmten Tokenizer kodiert."""
    from tokenizers import Tokenizer

    return encode(Tokenizer.from_file(str(tok_path)), as_text(split_norms(norms())[1]))


def prepared_tokenizer_and_holdout(prep: dict, data: Data):
    """Tokenizer und Holdout für ein vorbereitetes Modell.

    Die vorbereiteten Läufe trainieren auf dem Ersatzweg mit dem Tokenizer reference/lab3_tokenizer.json.
    Ist der Holdout des Kurses genau diese Datei, gilt er; sonst kodiert die Funktion die Holdout-Gesetze
    mit dem Tokenizer des Laufs neu: Ein Modell versteht nur die IDs seines eigenen Tokenizers.
    """
    from tokenizers import Tokenizer

    if data.tokenizer_sha == prep["tokenizer_sha256"] and sha256(data.val_path) == prep["val_sha256"]:
        return Tokenizer.from_file(str(data.tokenizer_path)), data.val
    if sha256(PREPARED_TOKENIZER) != prep["tokenizer_sha256"]:
        raise ValueError(f"{PREPARED_TOKENIZER.name} passt nicht zum vorbereiteten Lauf")
    val = holdout_for(PREPARED_TOKENIZER)
    print(f"Holdout für das vorbereitete Modell mit {PREPARED_TOKENIZER.name} kodiert ({format_de(len(val))} Tokens)")
    return Tokenizer.from_file(str(PREPARED_TOKENIZER)), val


def subset(law_slugs: tuple[str, ...], holdout: tuple[str, ...], tok_path: Path, name: str) -> Data:
    """Nur einige Gesetze: Training auf `law_slugs`, Holdout auf `holdout` (für einen vorbereiteten Lauf)."""
    from tokenizers import Tokenizer

    df = norms()
    tok = Tokenizer.from_file(str(tok_path))
    paths = {}
    for part, selection in (("train", law_slugs), ("val", holdout)):
        paths[part] = ARTIFACTS_DIR / f"lab3_{name}_{part}.bin"
        encode(tok, as_text(df[df["gesetz"].isin(selection)])).tofile(paths[part])
    return _memmaps(paths["train"], paths["val"], tok_path, name)
