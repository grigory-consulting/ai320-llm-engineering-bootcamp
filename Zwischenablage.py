import os
import sys
from pathlib import Path

for candidate in [Path.cwd(), Path.cwd() / "labs", *Path.cwd().parents, *(p / "labs" for p in Path.cwd().parents)]:
    if (candidate / "common" / "paths.py").exists():
        os.chdir(candidate)
        sys.path.insert(0, str(candidate))
        break
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # nur lokale Modelle aus labs/models

import random
import time

import numpy as np
import transformers
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer

from common import lab2_tokenizer as l2
from common.lab2_tokenizer import format_de
from common.paths import ARTIFACTS_DIR, MODELS_DIR, SEED

transformers.logging.set_verbosity_error()            # keine Warnung zur Maximallänge: hier wird nur gezählt
print("Lab-Ordner:", Path.cwd())



df, source = l2.load_corpus()
train_df, holdout_df = l2.train_holdout_split(df)
train, holdout = l2.as_text(train_df), l2.as_text(holdout_df)

print("Holdout-Gesetze:", ", ".join(l2.HOLDOUT_LAWS))
for name, texts in [("Training", train), ("Holdout", holdout)]:
    print(f"{name:9s} {format_de(len(texts)):>6} Normen {format_de(sum(map(len, texts))):>12} Zeichen")
print()
print(holdout[0][:300])



t0 = time.perf_counter()
tok = Tokenizer(models.BPE())
tok.normalizer = normalizers.NFC()
tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tok.decoder = decoders.ByteLevel()
trainer = trainers.BpeTrainer(
    vocab_size=8000, min_frequency=2,
    special_tokens=["<pad>", "<s>", "</s>"],
    initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),   # alle 256 Bytes
    show_progress=False)
tok.train_from_iterator(train, trainer)
seconds = time.perf_counter() - t0

print(f"Vokabular {format_de(tok.get_vocab_size())} Einträge, Training {format_de(seconds, 1)} s")
print("Spezialtokens:", {t: tok.token_to_id(t) for t in ["<pad>", "<s>", "</s>"]})

