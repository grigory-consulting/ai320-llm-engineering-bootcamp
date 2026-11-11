"""Vorbereitete Läufe für Lab 3 (Trainer, einmalig vor dem Kurs; Ablage in labs/reference/lab3_*).

Aufruf im Ordner labs/ (mit wenigen Threads, wenn der Rechner nebenbei rechnet):
    OMP_NUM_THREADS=2 python -m common.lab3_prepare langer_lauf   # 2.000 Schritte, Modell + Protokoll
    OMP_NUM_THREADS=2 python -m common.lab3_prepare warmup        # Warmup 0/10/100 (Cosine ab Schritt 100) und Folienformel mit 0, Seeds 1-3, je 600 Schritte
    OMP_NUM_THREADS=2 python -m common.lab3_prepare steuerrecht   # nur Steuerrecht, Seeds 1-3, Plan 4.000 und 2.000 Schritte

Alle Läufe trainieren auf dem Ersatzweg von lab3_data (Holdout: ganze Gesetze wie in Lab 2) mit dem eingefrorenen
Tokenizer reference/lab3_tokenizer.json. Der erste Aufruf kopiert ihn aus reference/lab2_tokenizer_v8000.json;
mit --neuer-tokenizer wird er neu kopiert, danach alle drei Teile neu rechnen (die Zahlen in den Notebooks ändern sich).
"""
from __future__ import annotations

import json
import platform
import shutil
import sys
import time

import numpy as np
import torch

from common import lab3_data
from common import lab3_train as L

REF = lab3_data.REFERENCE_DIR

LONG_RUN = {"batch": 16, "seq_len": 256, "lr": 1e-3, "warmup": 100, "total_steps": 2000,
               "eval_every": 100, "eval_batches": 16}
# Warmup-Vergleich: Der Cosine beginnt in allen Läufen bei Schritt 100 (decay_start), damit sich die Läufe nur in den
# ersten 100 Schritten unterscheiden. Dazu je Seed ein Lauf mit warmup=0 in der Folienformel (Cosine über alle 600 Schritte).
WARMUP = {"batch": 8, "seq_len": 128, "lr": 1e-3, "total_steps": 600, "decay_start": 100,
          "eval_every": 50, "eval_batches": 16}
WARMUPS = (0, 10, 100)
TAX = {"batch": 8, "seq_len": 128, "lr": 1e-3, "warmup": 100, "total_steps": 4000,
          "eval_every": 100, "eval_batches": 8}
TAX_TRAIN = ("AO 1977", "EStG", "KStG 1977", "GewStG")
TAX_HOLDOUT = ("UStG 1980",)
SEEDS = (1, 2, 3)


def _data():
    """Ersatzweg mit eingefrorenem Tokenizer: reference/lab3_tokenizer.json (Kopie der festen Ablage aus Lab 2)."""
    if "--neuer-tokenizer" in sys.argv or not lab3_data.PREPARED_TOKENIZER.exists():
        shutil.copyfile(lab3_data.fallback_tokenizer(), lab3_data.PREPARED_TOKENIZER)
    return lab3_data.load_own(lab3_data.PREPARED_TOKENIZER)


def _meta(data, seconds):
    return {"tokenizer_sha256": data.tokenizer_sha, "val_sha256": lab3_data.sha256(data.val_path),
            "train_sha256": lab3_data.sha256(data.train_path), "seconds": round(seconds),
            "platform": f"{platform.machine()} {platform.processor() or platform.system()}",
            "torch": torch.__version__, "threads": torch.get_num_threads()}


def _rounded(log):
    return [{k: (v if k == "step" else round(v, 5)) for k, v in r.items()} for r in log]


def long_run():
    data = _data()
    cfg = L.make_config(**LONG_RUN)
    run = L.Run(cfg, data)
    seconds = run.train(verbose_every=100)
    sd = {k: v.half() for k, v in run.model.state_dict().items() if k != "head.weight"}
    torch.save(sd, REF / "lab3_langer_lauf.pt")
    out = {"config": cfg, **_meta(data, seconds), "log": _rounded(run.log)}
    (REF / "lab3_langer_lauf.json").write_text(json.dumps(out), encoding="utf-8")


def warmup():
    data = _data()
    runs, t0 = [], time.perf_counter()
    variants = [(w, WARMUP["decay_start"]) for w in WARMUPS] + [(0, None)]
    for seed in SEEDS:
        for w, decay_start in variants:
            cfg = L.make_config(seed=seed, **{**WARMUP, "warmup": w, "decay_start": decay_start})
            run = L.Run(cfg, data)
            t = run.train()
            runs.append({"seed": seed, "warmup": w, "decay_start": decay_start, "seconds": round(t, 1),
                         "log": _rounded(run.log)})
            print(seed, w, decay_start, run.log[-1]["val_loss"], f"{t:.0f} s", flush=True)
    out = {"config": {k: v for k, v in L.make_config(**WARMUP).items() if k not in ("seed", "warmup", "decay_start")},
           **_meta(data, time.perf_counter() - t0), "runs": runs}
    (REF / "lab3_warmup_vergleich.json").write_text(json.dumps(out), encoding="utf-8")


def _tax_plan(data, total_steps):
    runs = []
    for seed in SEEDS:
        cfg = L.make_config(seed=seed, **{**TAX, "total_steps": total_steps})
        run = L.Run(cfg, data)
        run.train()
        runs.append({"seed": seed, "log": _rounded(run.log)})
        print(total_steps, seed, min(r.get("val_loss", 99) for r in run.log), flush=True)
    return {"config": {k: v for k, v in L.make_config(**{**TAX, "total_steps": total_steps}).items() if k != "seed"},
            "runs": runs}


def tax_law():
    """Plan über 4.000 Schritte (Aufgabe) und zum Vergleich derselbe Lauf mit Plan über 2.000 Schritte."""
    _data()
    data = lab3_data.subset(TAX_TRAIN, TAX_HOLDOUT, lab3_data.PREPARED_TOKENIZER, "steuerrecht")
    t0 = time.perf_counter()
    long = _tax_plan(data, TAX["total_steps"])
    short = _tax_plan(data, 2000)
    out = {**long, "train_gesetze": TAX_TRAIN, "holdout_gesetze": TAX_HOLDOUT,
           "train_tokens": len(data.train), "val_tokens": len(data.val),
           **_meta(data, time.perf_counter() - t0), "plan_2000": short}
    (REF / "lab3_lauf_steuerrecht.json").write_text(json.dumps(out), encoding="utf-8")


if __name__ == "__main__":
    REF.mkdir(exist_ok=True)
    torch.set_num_threads(int(__import__("os").environ.get("OMP_NUM_THREADS", torch.get_num_threads())))
    for part in sys.argv[1:]:
        {"langer_lauf": long_run, "warmup": warmup, "steuerrecht": tax_law}[part]()
