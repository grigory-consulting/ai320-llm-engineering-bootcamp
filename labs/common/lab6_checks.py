"""Prüf- und Vergleichsfunktionen für die Lab-6-Übung. Sie stellen Ihre Tabellen neben die vorbereiteten Läufe aus
labs/reference/ und vergleichen Tipps mit Messwerten; lesen müssen Sie sie nicht."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.interactive import check_guess

SEEDS = (1, 2, 3)


def _test_bpb(ref_lora: dict, name: str, seed: int) -> float:
    """Test-BPB am Ende eines vorbereiteten Laufs."""
    return ref_lora["runs"][f"{name}/s{seed}"]["curve"][-1][1]


# Aufgabe 1 · Zehn Adapter auf einer Basis

def check_lora_choice(tab: pd.DataFrame, choice: str, ref_lora: dict, guess) -> None:
    """Ihre Tabelle und Wahl neben Basis und Full-Fine-Tuning auf dem Test; Tipp gegen den Anteil an der Verbesserung."""
    baseline_test = ref_lora["base_test_bpb"]
    full_test = np.mean([_test_bpb(ref_lora, "full", s) for s in SEEDS])
    print(tab.round({"Anteil %": 3, "Val-BPB": 4, "Val ± (Seeds)": 4, "Test-BPB": 4}).to_string())
    print("\nim Budget:", list(tab.index[tab["Anteil %"] <= 1.0]), "| Wahl nach Validierung:", choice)
    share = 100 * (baseline_test - float(tab.loc[choice, "Test-BPB"])) / (baseline_test - full_test)
    per_seed = [100 * (baseline_test - _test_bpb(ref_lora, choice, s)) / (baseline_test - _test_bpb(ref_lora, "full", s))
                for s in SEEDS]
    print(f"Test-BPB: Basis {baseline_test:.4f}, Full-Fine-Tuning {full_test:.4f}, Wahl {float(tab.loc[choice, 'Test-BPB']):.4f}"
          f" | Anteil je Seed: {', '.join(f'{a:.0f} %' for a in per_seed)}")
    check_guess("Anteil der Wahl an der Verbesserung von Full-Fine-Tuning in %", guess, share, tolerance=8, fmt="{:.0f}")


def lora_context(ref_lora: dict) -> None:
    """Full-Fine-Tuning auf der Validierung, die gewählten Lernraten und die Kontrolle für r=64."""
    full_val = [ref_lora["runs"][f"full/s{s}"]["val_bpb"] for s in SEEDS]
    print(f"Basis: Val {ref_lora['base_val_bpb']:.4f} | Full-Fine-Tuning: Val {np.mean(full_val):.4f} ± {np.std(full_val, ddof=1):.4f}")
    print("Lernrate im Raster (gewählt bei r=16, für r=4 und r=64 übernommen):", ref_lora["best_lr"])
    print("Kontrolle r=64, Val-BPB:", {k: round(v, 4) for k, v in ref_lora["kontrolle_r64"].items()})


# Aufgabe 2 · Speicherformel gegen Datei

def check_types(formula_MB: float, file_MB: float, n_baseline: int, type_table: pd.DataFrame, guess) -> None:
    """Formel gegen Datei und Ihre Tabelle je Tensortyp; Tipp gegen den Anteil der Gewichte in Q4_K."""
    print(f"Formel {formula_MB:.1f} MB, Datei {file_MB:.1f} MB, Differenz {file_MB - formula_MB:+.1f} MB "
          f"({100 * (file_MB / formula_MB - 1):+.0f} %), Datei {8 * file_MB * 1e6 / n_baseline:.2f} Bit je Gewicht")
    print(type_table.round(2).to_string())
    share_q4k = float(type_table["Anteil Parameter %"].get("Q4_K", 0.0))
    check_guess("Anteil der Gewichte in Q4_K in %", guess, share_q4k, tolerance=10, fmt="{:.0f}")


# Aufgabe 3 · Höchstens 100 MB für die Gewichte

def check_quant_choice(k: pd.DataFrame, top: str, guess) -> None:
    """Ihre Qualitätstabelle und die gewählte Stufe; Tipp gegen deren PPL-Aufschlag."""
    print(k[["MB", "Bit je Gewicht", "ppl", "Aufschlag %", "± %", "kld", "same_top"]].astype(float).round(3).to_string())
    print("beste Stufe unter 100 MB:", top)
    check_guess("PPL-Aufschlag der besten Stufe unter 100 MB in %", guess, float(k.loc[top, "Aufschlag %"]),
                tolerance=5, fmt="{:.1f}")
