"""Lab 4: Fine-Tuning von distilbert-german als Klassifikator (kurzer eigener Trainingsloop).

Derselbe Loop steht sichtbar in der Demo; die Übung und lab4_prepare.py rufen ihn von hier auf.
"""
from __future__ import annotations

import os
import time

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from common.lab4_data import DBERT

# Im Kurs immer CPU, damit Laufzeiten denen der Kursrechner entsprechen.
# Nur die vorbereiteten Läufe (lab4_prepare.py) dürfen mit LAB4_DEVICE=mps schneller rechnen.
device = os.environ.get("LAB4_DEVICE", "cpu")


def batches(texts, y, batch_size: int = 16, seed: int = 1, bucket: int = 20):
    """Zufällige Batches, innerhalb von Blöcken aus bucket Batches nach Länge sortiert.

    Ähnlich lange Paragraphen landen im selben Batch; das spart Padding und damit Rechenzeit.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    block = batch_size * bucket
    out = []
    for i in range(0, len(order), block):
        chunk = sorted(order[i:i + block], key=lambda j: len(texts[j]))
        out += [chunk[k:k + batch_size] for k in range(0, len(chunk), batch_size)]
    for b in rng.permutation(len(out)):
        idx = out[b]
        yield [texts[j] for j in idx], torch.tensor([int(y[j]) for j in idx])


def load_model(classes: list[str], seed: int = 1):
    """Tokenizer und Modell mit frisch initialisiertem Klassifikationskopf (Seed bestimmt den Kopf)."""
    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(DBERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        DBERT, num_labels=len(classes),
        id2label=dict(enumerate(classes)), label2id={k: i for i, k in enumerate(classes)},
    ).to(device)
    return tok, model


@torch.no_grad()
def predict_proba(model, tok, texts, max_length: int = 256, batch_size: int = 64) -> np.ndarray:
    """Klassenwahrscheinlichkeiten, nach Länge sortiert gerechnet (weniger Padding)."""
    model.eval()
    order = np.argsort([len(t) for t in texts])
    out = np.zeros((len(texts), model.config.num_labels), dtype=np.float32)
    for i in range(0, len(texts), batch_size):
        idx = order[i:i + batch_size]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=max_length,
                  padding=True, return_tensors="pt").to(device)
        out[idx] = model(**enc).logits.float().softmax(-1).cpu().numpy()
    return out


def train(tok, model, texts, y, epochs: int = 1, lr: float = 5e-5, batch_size: int = 16,
          max_length: int = 256, seed: int = 1, on_epoch=None, log_every: int = 0,
          plan_epochs: int | None = None) -> dict:
    """AdamW, linearer Warmup über 10 % der Plan-Schritte, dann linear auf null; Gradient Clipping 1,0.

    Trainiert wird `epochs` Epochen. Der Lernratenplan (Warmup und Abfall) ist über `plan_epochs` Epochen
    ausgelegt, Standard `epochs`. Mit plan_epochs > epochs endet das Training, bevor die Lernrate auf null ist.
    on_epoch(epoch) wird nach jeder Epoche aufgerufen (z. B. für Validierung).
    """
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.01)
    per_epoch = int(np.ceil(len(texts) / batch_size))
    n_steps = epochs * per_epoch
    n_plan = (plan_epochs or epochs) * per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * n_plan), n_plan)
    t0, step, losses = time.time(), 0, []
    for ep in range(epochs):
        model.train()
        for bt, by in batches(texts, y, batch_size, seed=seed * 1000 + ep):
            enc = tok(bt, truncation=True, max_length=max_length, padding=True, return_tensors="pt").to(device)
            loss = model(**enc, labels=by.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            losses.append(loss.item()); step += 1
            if log_every and step % log_every == 0:
                print(f"Schritt {step}/{n_steps}  Loss {np.mean(losses[-log_every:]):.3f}  {time.time() - t0:.0f} s", flush=True)
        if on_epoch is not None:
            on_epoch(ep + 1)
    return {"schritte": step, "sekunden": time.time() - t0, "losses": losses}
