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
