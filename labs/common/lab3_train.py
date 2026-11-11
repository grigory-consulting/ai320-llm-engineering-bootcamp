"""Trainingsbausteine für Lab 3: dieselben Schritte wie in der Demo, als Funktionen für Übung und Vorbereitung.

Die Demo schreibt die Schleife sichtbar aus; die Übung und lab3_prepare.py rufen `run` auf.
Ein Lauf ist durch `config` vollständig bestimmt (Seed, Lernratenplan, Batch, Sequenzlänge).

Lernratenplan wie auf der Folie in Teil V: lineares Warmup über `warmup` Schritte, danach Cosine bis `total_steps`.
Mit `decay_start` beginnt der Cosine erst bei diesem Schritt; zwischen Warmup und `decay_start` steht der Faktor auf 1.
Ohne Angabe (None) ist `decay_start = warmup`, also genau die Folienformel.
"""
from __future__ import annotations

import math
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from common.lab3_model import MiniGPT

DEFAULTS = {
    "seed": 1, "vocab": 8000, "d": 256, "n_layer": 4, "n_head": 4, "ctx": 256,
    "batch": 16, "seq_len": 256, "lr": 1e-3, "warmup": 100, "decay_start": None, "total_steps": 2000, "min_ratio": 0.1,
    "weight_decay": 0.1, "betas": (0.9, 0.95), "clip": 1.0,
    "eval_every": 50, "eval_batches": 8, "device": "cpu", "ffn": "gelu",
}


def make_config(**kw) -> dict:
    unknown = set(kw) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"unbekannte Schlüssel: {sorted(unknown)}")
    return {**DEFAULTS, **kw}


def get_batch(data, batch, seq_len):
    """Zufällige Fenster aus dem Token-Strom (wie nanoGPT); die Positionen zieht der globale Torch-Zufall."""
    i = torch.randint(len(data) - seq_len - 1, (batch,))
    w = torch.stack([torch.from_numpy(data[j : j + seq_len + 1].astype(np.int64)) for j in i.tolist()])
    return w[:, :-1], w[:, 1:]


def lr_factor_fn(warmup, total, min_ratio, decay_start=None):
    start = warmup if decay_start is None else decay_start
    if start < warmup:
        raise ValueError("decay_start muss mindestens warmup sein")

    def lr_factor(step):
        if step < warmup:
            return (step + 1) / (warmup + 1)
        if step < start:
            return 1.0
        progress = min(1.0, (step - start) / max(1, total - start))
        return min_ratio + 0.5 * (1 - min_ratio) * (1 + math.cos(math.pi * progress))
    return lr_factor


def make_model(cfg) -> MiniGPT:
    torch.manual_seed(cfg["seed"])
    return MiniGPT(cfg["vocab"], cfg["d"], cfg["n_layer"], cfg["n_head"], cfg["ctx"],
                   cfg.get("ffn", "gelu")).to(cfg["device"])


def make_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in params if p.dim() >= 2]
    no_decay = [p for p in params if p.dim() < 2]
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg["lr"], betas=tuple(cfg["betas"]), eps=1e-8)


def make_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_factor_fn(cfg["warmup"], cfg["total_steps"], cfg["min_ratio"], cfg.get("decay_start")))


@torch.no_grad()
def evaluate(model, val, seq_len, batches, batch=16):
    """Mittlerer Loss auf festen, gleichmäßig verteilten Holdout-Fenstern (ohne Zufall)."""
    model.eval()
    n = batches * batch
    starts = np.linspace(0, len(val) - seq_len - 2, n).astype(np.int64)
    losses = []
    for k in range(0, n, batch):
        w = torch.stack([torch.from_numpy(val[j : j + seq_len + 1].astype(np.int64)) for j in starts[k:k + batch]])
        logits = model(w[:, :-1])
        losses.append(F.cross_entropy(logits.flatten(0, 1), w[:, 1:].flatten()).item())
    model.train()
    return float(np.mean(losses))


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}


def set_rng_state(s):
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])


class Run:
    """Modell, Optimizer und Scheduler eines Laufs samt Protokoll; Seed setzt Initialisierung und Batchfolge."""

    def __init__(self, cfg, data):
        self.cfg, self.data = cfg, data
        self.model = make_model(cfg)
        self.optimizer = make_optimizer(self.model, cfg)
        self.scheduler = make_scheduler(self.optimizer, cfg)
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
        self.step = 0
        self.log = []          # je Schritt: step, loss, grad_norm, lr (+ val_loss alle eval_every Schritte)

    def train_step(self):
        cfg = self.cfg
        x, y = get_batch(self.data.train, cfg["batch"], cfg["seq_len"])
        logits = self.model(x.to(cfg["device"]))
        loss = F.cross_entropy(logits.flatten(0, 1), y.to(cfg["device"]).flatten())
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg["clip"])
        lr = self.scheduler.get_last_lr()[0]
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        row = {"step": self.step, "loss": loss.item(), "grad_norm": grad_norm.item(), "lr": lr}
        self.step += 1
        if cfg["eval_every"] and (self.step % cfg["eval_every"] == 0 or self.step == cfg["total_steps"]):
            row["val_loss"] = evaluate(self.model, self.data.val, cfg["seq_len"], cfg["eval_batches"])
        self.log.append(row)
        return row

    def train(self, steps=None, until=None, mlflow=None, log_every=10, verbose_every=0):
        """Trainiert bis `until` (oder `steps` weitere Schritte, sonst bis total_steps).

        Mit `mlflow` (dem Modul) gehen Loss, Grad-Norm und Lernrate alle `log_every` Schritte
        und jede Holdout-Messung als Metriken in den aktiven Run.
        """
        until = until if until is not None else (self.step + steps if steps else self.cfg["total_steps"])
        t0 = time.perf_counter()
        while self.step < until:
            row = self.train_step()
            if mlflow is not None and (row["step"] % log_every == 0 or "val_loss" in row):
                mlflow.log_metrics({k: v for k, v in row.items() if k != "step"}, step=row["step"] + 1)
            if verbose_every and self.step % verbose_every == 0:
                print(f"Schritt {self.step:5d}  Loss {row['loss']:.3f}  Grad-Norm {row['grad_norm']:.2f}  "
                      f"LR {row['lr']:.2e}" + (f"  Holdout {row['val_loss']:.3f}" if "val_loss" in row else ""))
        return time.perf_counter() - t0

    def state(self):
        return {"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(), "step": self.step, "config": self.cfg,
                "rng": rng_state(),
                "tokenizer_sha": self.data.tokenizer_sha}

    def save(self, path):
        torch.save(self.state(), path)

    def load(self, path_or_state, parts=("model", "optimizer", "scheduler", "rng")):
        ck = torch.load(path_or_state, weights_only=False) if not isinstance(path_or_state, dict) else path_or_state
        if "model" in parts:
            self.model.load_state_dict(ck["model"])
        if "optimizer" in parts:
            self.optimizer.load_state_dict(ck["optimizer"])
        if "scheduler" in parts:
            self.scheduler.load_state_dict(ck["scheduler"])
        if "rng" in parts:
            set_rng_state(ck["rng"])
        self.step = ck["step"]
        return ck


def curves(log):
    """Protokoll als Spalten: step, loss, grad_norm, lr; val_step, val_loss."""
    out = {k: np.array([r[k] for r in log]) for k in ("step", "loss", "grad_norm", "lr")}
    out["val_step"] = np.array([r["step"] + 1 for r in log if "val_loss" in r])
    out["val_loss"] = np.array([r["val_loss"] for r in log if "val_loss" in r])
    return out
