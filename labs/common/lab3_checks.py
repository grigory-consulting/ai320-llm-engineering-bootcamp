"""Prüfcode für die Lab-3-Übung. Er vergleicht Ihre Läufe und Tipps mit den vorbereiteten Läufen über drei Seeds;
lesen müssen Sie ihn nicht."""
from __future__ import annotations

import json

import numpy as np

from common.interactive import check_guess
from common.paths import LAB_DIR

SEEDS = (1, 2, 3)


def holdout(log: list[dict]) -> dict[int, float]:
    """Holdout-Loss je Messpunkt eines Protokolls: {Schritt: val_loss}."""
    return {x["step"] + 1: x["val_loss"] for x in log if "val_loss" in x}


def best(run: dict) -> tuple[int, float]:
    """Schritt und Holdout-Loss des besten Checkpoints eines Laufs."""
    v = [x for x in run["log"] if "val_loss" in x]
    b = min(v, key=lambda x: x["val_loss"])
    return b["step"] + 1, b["val_loss"]


def warmup_runs() -> tuple[dict, dict]:
    """Vorbereitete Läufe zu Aufgabe 1, 2 und 5: {(seed, warmup): log} mit Cosine ab Schritt 100,
    dazu {seed: log} für warmup=0 in der Folienformel."""
    comparison = json.loads((LAB_DIR / "reference" / "lab3_warmup_vergleich.json").read_text(encoding="utf-8"))
    logs = {(r["seed"], r["warmup"]): r["log"] for r in comparison["runs"] if r["decay_start"] == 100}
    slide = {r["seed"]: r["log"] for r in comparison["runs"] if r["decay_start"] is None}
    return logs, slide


def peak(log: list[dict], n: int = 60) -> int:
    """Index des Schritts mit der größten Grad-Norm unter den ersten n Schritten."""
    return int(np.argmax([x["grad_norm"] for x in log[:n]]))


# Aufgabe 1 · Warmup weglassen?


def check_warmup(run_logs: dict, guess) -> None:
    """Ihre Holdout-Differenz und Ihr Tipp gegen die vorbereiteten Läufe; Grad-Norm-Spitzen je Seed."""
    logs, slide = warmup_runs()
    final = {k: holdout(log)[600] for k, log in logs.items()}
    diffs = [final[(s, 0)] - final[(s, 100)] for s in SEEDS]
    diff_own = run_logs[0]["val_loss"][-1] - run_logs[100]["val_loss"][-1]
    print("vorbereitet, Holdout-Differenz je Seed:", " ".join(f"{d:+.3f}" for d in diffs))
    print(f"Ihr Lauf (Seed 1): {diff_own:+.3f}")
    check_guess("Holdout ohne minus mit Warmup", guess, float(np.mean(diffs)), tolerance=0.01)

    print("\nvorbereitete Läufe: größte Grad-Norm der ersten 60 Schritte, Holdout nach 50 und 600 Schritten")
    for (s, w), log in sorted(logs.items()):
        if w == 10:
            continue                                             # warmup=10 gehört zu Aufgabe 5
        k = peak(log)
        print(f"  Seed {s}, warmup={w:3d}: {log[k]['grad_norm']:6.1f} in Schritt {k + 1:2d} | "
              f"Holdout {holdout(log)[50]:.3f} / {holdout(log)[600]:.3f}")
    spread = max(final[(s, 100)] for s in SEEDS) - min(final[(s, 100)] for s in SEEDS)
    print("Holdout-Werte mit warmup=100 über die Seeds: Spanne", f"{spread:.3f}")
    naive = [holdout(slide[s])[600] - final[(s, 100)] for s in SEEDS]
    print("zum Vergleich warmup=0 in der Folienformel (Cosine über 600 Schritte) minus warmup=100:",
          " ".join(f"{d:+.3f}" for d in naive))


# Aufgabe 2 · Die ganze Laufzeit in MLflow


def crossover_point(log_a: list[dict], log_b: list[dict]) -> int:
    """Erster Messpunkt, ab dem sich die Reihenfolge der beiden Läufe im Holdout nicht mehr ändert."""
    a, b = holdout(log_a), holdout(log_b)
    measured = sorted(a)
    front = [np.sign(a[s] - b[s]) for s in measured]
    return next(s for i, s in enumerate(measured) if len(set(front[i:])) == 1)


def check_crossover(guess) -> None:
    """Messpunkte mit Vorsprung ohne Warmup und Kreuzungspunkt je Seed; Ihr Tipp gegen das Mittel."""
    logs, _ = warmup_runs()
    counts = {}
    for s in SEEDS:
        a, b = holdout(logs[(s, 0)]), holdout(logs[(s, 100)])
        counts[s] = sum(a[k] < b[k] for k in a)
        print(f"vorbereitet, Seed {s}: ohne Warmup vorn an {counts[s]} von {len(a)} Messpunkten, "
              f"Reihenfolge fest ab Schritt {crossover_point(logs[(s, 0)], logs[(s, 100)])}, "
              f"Differenz nach 100 Schritten {a[100] - b[100]:+.3f}, nach 600 {a[600] - b[600]:+.3f}")
    check_guess("Messpunkte mit Vorsprung ohne Warmup", guess, float(np.mean(list(counts.values()))),
                tolerance=2, fmt="{:.1f}")


# Aufgabe 3 · Welchen Checkpoint liefern Sie aus?


def check_plan(tax: dict, guess) -> None:
    """Plan 2.000 am Ende gegen den besten Checkpoint des Plans 4.000, je Seed; Ihr Tipp gegen die Richtung."""
    gap = [[x["val_loss"] for x in r2["log"] if "val_loss" in x][-1] - best(r4)[1]
           for r4, r2 in zip(tax["runs"], tax["plan_2000"]["runs"])]
    direction = "besser" if max(gap) < 0 else "schlechter" if min(gap) > 0 else "gleich"
    print("vorbereitet, Plan 2.000 am Ende minus bester Checkpoint Plan 4.000, je Seed:", " ".join(f"{a:+.3f}" for a in gap))
    check_guess("Plan 2.000 gegen besten Checkpoint", guess, direction)


# Aufgabe 5 · Zehn Warmup-Schritte


def check_warmup_10(guess) -> None:
    """warmup=10 gegen warmup=100 und warmup=0 je Seed; Ihr Tipp gegen das Mittel."""
    logs, _ = warmup_runs()
    final = {k: holdout(log)[600] for k, log in logs.items()}
    diffs_10 = [final[(s, 10)] - final[(s, 100)] for s in SEEDS]
    print("vorbereitet, warmup=10 minus warmup=100 je Seed:", " ".join(f"{d:+.3f}" for d in diffs_10))
    print("vorbereitet, warmup=0 minus warmup=10 je Seed:  ", " ".join(f"{final[(s, 0)] - final[(s, 10)]:+.3f}" for s in SEEDS))
    for s in SEEDS:
        log = logs[(s, 10)]
        k = peak(log)
        print(f"  Seed {s}, warmup=10: größte Grad-Norm {log[k]['grad_norm']:.1f} in Schritt {k + 1}")
    check_guess("Holdout warmup=10 minus warmup=100", guess, float(np.mean(diffs_10)), tolerance=0.008)
