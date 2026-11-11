"""Lab 6, Demo Abschnitte 1 bis 3 und Übung Aufgabe 1: Daten, Messung und Training für LoRA gegen Full-Fine-Tuning.

Aufgabe: SmolLM2-135M wird auf Gesetzestexten weitertrainiert (Causal LM).
Die Gesetze sind in drei disjunkte Gruppen geteilt: Training, Validierung
(Lernrate wählen) und Test (berichtete Qualität). Gemessen wird Bits per Byte
(BPB) auf Blöcken zu 256 Token aus den Testgesetzen.

Aufruf als Skript erzeugt die vorbereiteten Läufe in labs/reference/:
    python -m common.lab6_lora --prepare [--device mps]       (rund 40 Läufe, auf MPS etwa eine Stunde)
    python -m common.lab6_lora --budget [--device mps]        (Full und lora-qv16 mit 300 Schritten, Seed 1)
    python -m common.lab6_lora --rss-ablage                   (nur reference/lab6_lora_rss.json neu, CPU)
    python -m common.lab6_lora --rss full|lora-qv16|lora-qv16-d0|... [batch]  (ein Messlauf, druckt Spitzen-RSS)
Die Endung "-d0" heißt: dieselbe LoRA-Konfiguration mit lora_dropout=0 statt 0.05.
Die Quantisierungsreferenz (Demo Abschnitte 4 und 5) legt `python -m common.lab6_gguf` ab (reference/lab6_quant.json).
"""
from __future__ import annotations

import json
import math
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from common.paths import LAB_DIR, MODELS_DIR, SEED

MODEL_DIR = MODELS_DIR / "smollm2-135m"
REFERENCE = LAB_DIR / "reference"
RUNS_JSON = REFERENCE / "lab6_lora_runs.json"
RSS_JSON = REFERENCE / "lab6_lora_rss.json"
ADAPTER_DIR = REFERENCE / "lab6_adapter_qv16"

TEST_LAWS = ["UrhG", "VwGO", "BetrVG", "UStG 1980", "JGG"]
VAL_LAWS = ["GmbHG", "VwVfG", "SGB 1", "KStG 1977", "OWiG 1968"]
BLOCK = 256
EVAL_PER_LAW = 16          # 5 Gesetze x 16 Blöcke = 80 Blöcke, rund 20.000 vorhergesagte Token
STEPS, BATCH, WARMUP = 150, 8, 15

# Varianten: Name -> LoRA-Einstellung (None = Full-Fine-Tuning)
VARIANTS = {
    "full": None,
    "lora-qv16": {"r": 16, "target_modules": ["q_proj", "v_proj"]},
}
GRID = {f"lora-{t}{r}": {"r": r, "target_modules": mods}
        for t, mods in [("qv", ["q_proj", "v_proj"]), ("all", "all-linear")] for r in (4, 16, 64)}


def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_DIR)


def law_text(df, law_slug: str) -> str:
    return "\n\n".join(df[df.gesetz == law_slug].text)


def blocks(tok, df, laws, per_law: int | None = None, block: int = BLOCK) -> torch.Tensor:
    """Schneidet jedes Gesetz in Blöcke zu `block` Token; per_law wählt gleichmäßig verteilte Blöcke."""
    out = []
    for g in laws:
        ids = tok(law_text(df, g), verbose=False)["input_ids"]
        n = len(ids) // block
        x = torch.tensor(ids[: n * block]).view(n, block)
        if per_law is not None and per_law < n:
            x = x[np.linspace(0, n - 1, per_law).round().astype(int)]
        out.append(x)
    return torch.cat(out)


def train_laws(df) -> list[str]:
    return sorted(set(df.gesetz) - set(TEST_LAWS) - set(VAL_LAWS))


def byte_count(tok, x: torch.Tensor) -> int:
    """Bytes der vorhergesagten Token (Position 1 bis Ende je Block).

    Byte-Level-BPE: jedes Zeichen eines Tokens steht für genau ein Byte.
    """
    lengths = {}
    total = 0
    for i in x[:, 1:].flatten().tolist():
        if i not in lengths:
            lengths[i] = len(tok.convert_ids_to_tokens(i))
        total += lengths[i]
    return total


@torch.no_grad()
def nll_bits(model, x: torch.Tensor, batch: int = 8) -> float:
    """Summe der negativen Log-Likelihood in Bit über alle vorhergesagten Token."""
    model.eval()
    dev = next(model.parameters()).device
    total = 0.0
    for i in range(0, len(x), batch):
        xb = x[i : i + batch].to(dev)
        logits = model(input_ids=xb).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, xb[:, 1:, None]).squeeze(-1)
        total -= lp.sum().item()
    return total / math.log(2)


def bits_per_byte(model, x: torch.Tensor, n_bytes: int, batch: int = 8) -> float:
    return nll_bits(model, x, batch) / n_bytes


def load_base(device: str = "cpu"):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to(device)


def add_lora(model, r: int, target_modules, alpha: int | None = None, dropout: float = 0.05):
    from peft import LoraConfig, TaskType, get_peft_model

    cfg = LoraConfig(r=r, lora_alpha=alpha or 2 * r, lora_dropout=dropout,
                     target_modules=target_modules, task_type=TaskType.CAUSAL_LM)
    return get_peft_model(model, cfg)


def make_optimizer(model, lr: float):
    return torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=0.0)


def train_step(model, opt, xb: torch.Tensor) -> float:
    model.train()
    loss = model(input_ids=xb, labels=xb).loss
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return loss.item()


def state_bytes(model, opt) -> dict:
    """Bytes in Gewichten, Gradienten und AdamW-Zustand (ohne Aktivierungen)."""
    params = list(model.parameters())
    opt_bytes = sum(t.numel() * t.element_size() for s in opt.state.values() for t in s.values()
                    if torch.is_tensor(t) and t.dim() > 0)
    return {
        "gewichte_MB": sum(p.numel() * p.element_size() for p in params) / 1e6,
        "trainierbar": sum(p.numel() for p in params if p.requires_grad),
        "gradienten_MB": sum(p.numel() * p.element_size() for p in params if p.requires_grad) / 1e6,
        "adamw_MB": opt_bytes / 1e6,
    }


def time_steps(model, x: torch.Tensor, lr: float, batch: int, steps: int = 2, seed: int = SEED) -> dict:
    """Ein Aufwärmschritt, dann `steps` gemessene Schritte; liefert s/Schritt, die Zeit je Phase
    (Vorwärtslauf, Rückwärtslauf, AdamW-Update) und die Speicherbilanz."""
    g = torch.Generator().manual_seed(seed)
    opt = make_optimizer(model, lr)
    idx = torch.randint(len(x), (steps + 1, batch), generator=g)
    train_step(model, opt, x[idx[0]])
    model.train()
    phases = []
    for i in range(1, steps + 1):
        xb = x[idx[i]]
        t0 = time.perf_counter()
        loss = model(input_ids=xb, labels=xb).loss
        t1 = time.perf_counter()
        loss.backward()
        t2 = time.perf_counter()
        opt.step()
        opt.zero_grad(set_to_none=True)
        phases.append((t1 - t0, t2 - t1, time.perf_counter() - t2))
    fwd, bwd, upd = (float(v) for v in np.mean(phases, axis=0))
    return {"s_je_schritt": fwd + bwd + upd, "s_vorwaerts": fwd, "s_rueckwaerts": bwd, "s_update": upd,
            **state_bytes(model, opt)}


def train(model, x: torch.Tensor, lr: float, steps: int = STEPS, batch: int = BATCH, seed: int = SEED,
          eval_every: int = 25, eval_fn=None) -> dict:
    """Weitertraining mit AdamW, Warmup und Cosine; eval_fn(model) -> BPB an den Messpunkten."""
    from transformers import get_cosine_schedule_with_warmup

    dev = next(model.parameters()).device
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = make_optimizer(model, lr)
    sched = get_cosine_schedule_with_warmup(opt, WARMUP, steps)
    curve = [(0, eval_fn(model))] if eval_fn else []
    losses = []
    for step in range(1, steps + 1):
        xb = x[torch.randint(len(x), (batch,), generator=g)].to(dev)
        losses.append(train_step(model, opt, xb))
        sched.step()
        if eval_fn and (step % eval_every == 0 or step == steps):
            curve.append((step, eval_fn(model)))
    return {"curve": curve, "train_loss": losses}


# --- vorbereitete Läufe -------------------------------------------------------------------------

RSS_MODES = ("full", "lora-qv16", "lora-qv16-d0", "lora-all64", "lora-all64-d0")


def _peak_rss(mode: str, batch: int = BATCH, steps: int = 3) -> None:
    """Kindprozess: Modell laden, `steps` Trainingsschritte auf CPU, Spitzen-RSS in MB ausgeben.

    Endet `mode` auf "-d0", läuft dieselbe LoRA-Konfiguration mit lora_dropout=0.
    """
    from common import corpus

    torch.manual_seed(SEED)
    tok = tokenizer()
    df = corpus.load()
    x = blocks(tok, df, ["BGB"], per_law=64)
    model = load_base("cpu")
    name, dropout = (mode[:-3], 0.0) if mode.endswith("-d0") else (mode, 0.05)
    cfg = {**VARIANTS, **GRID}[name]
    if cfg:
        model = add_lora(model, **cfg, dropout=dropout)
    res = time_steps(model, x, 1e-4, batch, steps)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / 1e6 if sys.platform == "darwin" else peak / 1e3   # macOS: Bytes, Linux: KiB
    print(json.dumps({"mode": mode, "lora_dropout": dropout if cfg else None, "peak_rss_MB": peak_mb, **res}))


def measure_rss(modes=RSS_MODES, batch: int = BATCH) -> dict:
    out = {}
    for mode in modes:
        r = subprocess.run([sys.executable, "-m", "common.lab6_lora", "--rss", mode, str(batch)],
                           cwd=LAB_DIR, capture_output=True, text=True, check=True)
        out[mode] = json.loads(r.stdout.strip().splitlines()[-1])
    return out


def prepare(device: str = "cpu", seeds=(1, 2, 3)) -> None:
    """Lernraten-Wahl auf den Validierungsgesetzen, dann alle Varianten mit drei Seeds auf Test."""
    from common import corpus

    REFERENCE.mkdir(exist_ok=True)
    tok = tokenizer()
    df = corpus.load()
    x_train = blocks(tok, df, train_laws(df))
    x_val = blocks(tok, df, VAL_LAWS, EVAL_PER_LAW)
    x_test = blocks(tok, df, TEST_LAWS, EVAL_PER_LAW)
    b_val, b_test = byte_count(tok, x_val), byte_count(tok, x_test)
    res = json.loads(RUNS_JSON.read_text()) if RUNS_JSON.exists() else {}
    res.update({"device": device, "steps": STEPS, "batch": BATCH, "block": BLOCK, "warmup": WARMUP,
                "train_blocks": len(x_train), "test_blocks": len(x_test), "test_bytes": b_test,
                "test_laws": TEST_LAWS, "val_laws": VAL_LAWS})
    res.setdefault("lr_sweep", {})
    res.setdefault("runs", {})

    def save():
        RUNS_JSON.write_text(json.dumps(res, indent=1))

    def run(name, cfg, lr, seed, xe, be, eval_every):
        torch.manual_seed(seed)
        model = load_base(device)
        if cfg:
            model = add_lora(model, **cfg)
        t0 = time.time()
        out = train(model, x_train, lr, seed=seed, eval_every=eval_every,
                    eval_fn=lambda m: bits_per_byte(m, xe, be))
        out["minutes"] = (time.time() - t0) / 60
        out["trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(name, lr, seed, out["curve"][-1], f"{out['minutes']:.1f} min", flush=True)
        return model, out

    base = load_base(device)
    res["base_test_bpb"] = bits_per_byte(base, x_test, b_test)
    res["base_val_bpb"] = bits_per_byte(base, x_val, b_val)
    del base
    save()

    lrs = {"full": [1e-4, 2e-4, 4e-4, 8e-4, 1.6e-3], "lora-qv": [1e-3, 3e-3, 6e-3, 1.2e-2],
           "lora-all": [1e-3, 3e-3, 6e-3]}
    best = {}
    for family, grid in lrs.items():
        cfg = None if family == "full" else {"r": 16, "target_modules": ["q_proj", "v_proj"] if family == "lora-qv" else "all-linear"}
        grid = list(grid)
        while True:
            for lr in grid:
                key = f"{family}@{lr:g}"
                if key not in res["lr_sweep"]:
                    _, out = run(key, cfg, lr, SEED, x_val, b_val, STEPS)
                    res["lr_sweep"][key] = out["curve"][-1][1]
                    save()
            best[family] = min(grid, key=lambda lr: res["lr_sweep"][f"{family}@{lr:g}"])
            if best[family] != grid[-1] or len(grid) >= 8:
                break
            grid.append(grid[-1] * 2)          # bestes Ergebnis am Rand: Raster nach oben erweitern
    res["best_lr"] = best

    for name, cfg in {**VARIANTS, **GRID}.items():
        family = "full" if cfg is None else ("lora-qv" if cfg["target_modules"] != "all-linear" else "lora-all")
        for seed in seeds:
            key = f"{name}/s{seed}"
            if key in res["runs"]:
                continue
            model, out = run(key, cfg, best[family], seed, x_test, b_test, 25)
            res["runs"][key] = {"lr": best[family], "curve": out["curve"], "trainable": out["trainable"],
                                "val_bpb": bits_per_byte(model, x_val, b_val),
                                "minutes": out["minutes"],
                                "train_loss_last20": float(np.mean(out["train_loss"][-20:]))}
            if name == "lora-qv16" and seed == SEED:
                model.save_pretrained(ADAPTER_DIR)
            del model
            save()

    # Kontrolle: r=64 lief mit der Lernrate von r=16; je ein Lauf mit kleinerer Lernrate (Seed 1, Validierung)
    res.setdefault("kontrolle_r64", {})
    for name, lrs_k in [("lora-qv64", [3e-3, 1.5e-3]), ("lora-all64", [1.5e-3, 7.5e-4])]:
        for lr in lrs_k:
            key = f"{name}@{lr:g}"
            if key not in res["kontrolle_r64"]:
                _, out = run(key, GRID[name], lr, SEED, x_val, b_val, STEPS)
                res["kontrolle_r64"][key] = out["curve"][-1][1]
                save()


def prepare_budget(device: str = "cpu", steps: int = 2 * STEPS, seed: int = SEED, names=("full", "lora-qv16")) -> None:
    """Kontrolle zum Budget: Full und lora-qv16 mit doppelt so vielen Schritten (Cosine auf das neue Ende),
    gleiche Lernraten wie in den 150-Schritte-Läufen, ein Seed. Ablage unter res["budget300"]."""
    from common import corpus

    tok = tokenizer()
    df = corpus.load()
    x_train = blocks(tok, df, train_laws(df))
    x_val = blocks(tok, df, VAL_LAWS, EVAL_PER_LAW)
    x_test = blocks(tok, df, TEST_LAWS, EVAL_PER_LAW)
    b_val, b_test = byte_count(tok, x_val), byte_count(tok, x_test)
    res = json.loads(RUNS_JSON.read_text())
    store = res.setdefault("budget300", {"device": device, "steps": steps, "seed": seed, "runs": {}})
    store.update({"device": device, "steps": steps, "seed": seed})
    for name in names:
        cfg = VARIANTS[name]
        lr = res["best_lr"]["full" if cfg is None else "lora-qv"]
        torch.manual_seed(seed)
        model = load_base(device)
        if cfg:
            model = add_lora(model, **cfg)
        t0 = time.time()
        out = train(model, x_train, lr, steps=steps, seed=seed, eval_every=50,
                    eval_fn=lambda m: bits_per_byte(m, x_test, b_test))
        store["runs"][name] = {"lr": lr, "curve": out["curve"], "val_bpb": bits_per_byte(model, x_val, b_val),
                                "minutes": (time.time() - t0) / 60, "train_loss_max": float(max(out["train_loss"])),
                                "train_loss_last20": float(np.mean(out["train_loss"][-20:]))}
        print(name, lr, out["curve"][-1], f"max loss {max(out['train_loss']):.2f}",
              f"{store['runs'][name]['minutes']:.1f} min", flush=True)
        del model
        RUNS_JSON.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    dev = sys.argv[sys.argv.index("--device") + 1] if "--device" in sys.argv else "cpu"
    if "--rss-ablage" in sys.argv:
        RSS_JSON.write_text(json.dumps({"batch8": measure_rss(batch=8), "batch2": measure_rss(batch=2)}, indent=1))
    elif "--budget" in sys.argv:
        only = [a for a in sys.argv[1:] if a in VARIANTS]
        prepare_budget(dev, names=only or ("full", "lora-qv16"))
    elif "--rss" in sys.argv:
        i = sys.argv.index("--rss")
        _peak_rss(sys.argv[i + 1], int(sys.argv[i + 2]) if len(sys.argv) > i + 2 else BATCH)
    elif "--prepare" in sys.argv:
        prepare(dev)
        RSS_JSON.write_text(json.dumps({"batch8": measure_rss(batch=8), "batch2": measure_rss(batch=2)}, indent=1))
