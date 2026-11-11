import os
import sys
from pathlib import Path

for candidate in [Path.cwd(), Path.cwd() / "labs", *Path.cwd().parents, *(p / "labs" for p in Path.cwd().parents)]:
    if (candidate / "common" / "paths.py").exists():
        os.chdir(candidate)
        sys.path.insert(0, str(candidate))
        break

import math
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from common import lab3_data, lab3_train
from common.lab3_model import MiniGPT
from common.paths import SEED

device = "cpu"
print("torch", torch.__version__, "| device", device, "| Threads", torch.get_num_threads())


data = lab3_data.load()
train_data, val_data = data.train, data.val

from tokenizers import Tokenizer
tok = Tokenizer.from_file(str(data.tokenizer_path))
print(tok.decode(train_data[:60].tolist(), skip_special_tokens=False))



torch.manual_seed(SEED)
model = MiniGPT(vocab=8000, d=256, n_layer=4, n_head=4, ctx=256).to(device)
print(f"Parameter: {sum(p.numel() for p in model.parameters()):,}".replace(",", "."))
