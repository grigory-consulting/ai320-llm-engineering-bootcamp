"""Prüf- und Vergleichsfunktionen für die Lab-4-Übung. Sie stellen Ihre Läufe neben die vorbereiteten Läufe aus
labs/reference/ und vergleichen Tipps mit Messwerten; lesen müssen Sie sie nicht."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from common import lab4_data as d
from common.interactive import check_guess
from common.paths import SEED

SEEDS = (1, 2, 3)


def chosen_tfidf(ref: dict):
    """TF-IDF + LR (untrainiert) in der Einstellung, die die Validierung in der Demo gewählt hat."""
    choice = ref["tfidf_beste"]
    class_weight = None if choice["class_weight"] == "keine" else choice["class_weight"]
    return make_pipeline(d.tfidf_vectorizer(choice["analyzer"]),
                         LogisticRegression(C=choice["C"], class_weight=class_weight, max_iter=3000))


def demo_ft_test(ref: dict, proba: dict) -> list[np.ndarray]:
    """Testvorhersagen des Fine-Tunings aus der Demo, je Seed aus der Epoche mit der besten Macro-F1 auf der Validierung."""
    best_epoch = {sd: max((v for v in ref[f"ft_s{sd}"]["verlauf"] if v["teil"] == "val"),
                          key=lambda v: v["macro_f1"])["epoche"] for sd in SEEDS}
    return [proba[f"ft_s{sd}_e{best_epoch[sd]}_test"].argmax(1) for sd in SEEDS]


# Aufgabe 1 · Nur 800 gelabelte Paragraphen

SHORT = "FT kurzes Rezept (128 Tokens, Plan 2 Ep.)"
TOK = "FT Token-Variante (256 Tokens, Plan 2 Ep.)"
PLAN = "FT Plan-Variante (128 Tokens, Plan 8 Ep.)"


def check_tfidf_800(X, y, s: dict, sub: np.ndarray, ref: dict, guess) -> list[float]:
    """TF-IDF + LR auf den 800 Paragraphen, auf der Validierung eingestellt; Seed 1 hier gerechnet, Seeds 2 und 3
    vorbereitet (gleiches Verfahren). Gibt die Macro-F1 auf dem Test je Seed zurück."""
    t = time.time()
    tf800, _ = d.tfidf_tuned(X[sub], y[sub], X[s["val"]], y[s["val"]])
    seed_1 = d.metrics(y[s["test"]], tf800["modell"].predict(X[s["test"]]))["macro_f1"]
    print(f"TF-IDF + LR 800, Seed 1: gewählt {tf800['analyzer']}, C = {tf800['C']:g}, Klassengewichte {tf800['class_weight']}; "
          f"Validierung {tf800['val_macro_f1']:.3f}, Test {seed_1:.3f} ({time.time() - t:.0f} s)")
    seeds = [seed_1] + [ref[f"klein_tfidf_s{sd}"]["macro_f1"] for sd in (2, 3)]
    check_guess("Macro-F1 TF-IDF + LR mit 800 (Mittel über 3 Seeds)", guess, float(np.mean(seeds)), tolerance=0.05)
    return seeds


def compare_800(p_short, p_plan, tf800_seeds: list[float], ref: dict, y_te, guess) -> None:
    """Alle Rezepte mit 800 Paragraphen nebeneinander, je drei Seeds (Seed 1 der Live-Läufe von Ihnen)."""
    def f1_test(p):
        return d.metrics(y_te, p)["macro_f1"]

    tab800 = pd.DataFrame({
        "TF-IDF + LR, eingestellt": tf800_seeds,
        SHORT: [f1_test(p_short)] + [ref[f"klein_kurz_s{sd}"]["macro_f1"] for sd in (2, 3)],
        TOK: [ref[f"klein_tok_s{sd}"]["macro_f1"] for sd in SEEDS],
        PLAN: [f1_test(p_plan)] + [ref[f"klein_plan_s{sd}"]["macro_f1"] for sd in (2, 3)],
        "FT 8-Epochen-Plan mit Validierung (256 Tokens)": [ref[f"klein_s{sd}"]["macro_f1"] for sd in SEEDS],
    }, index=["Seed 1", "Seed 2", "Seed 3"]).T
    tab800["Mittel"] = tab800.mean(axis=1)
    print("Macro-F1 auf dem Test, 800 Trainingsparagraphen\n", tab800.round(3).to_string())
    print("gewählte Epoche im 8-Epochen-Plan mit Validierung, je Seed:", {sd: ref[f"klein_s{sd}"]["beste_epoche"] for sd in SEEDS})
    gain = {"256 Tokens": tab800.loc[TOK] - tab800.loc[SHORT], "längerer Lernratenplan": tab800.loc[PLAN] - tab800.loc[SHORT]}
    print("\nGewinn gegenüber dem kurzen Rezept, Seeds 1, 2, 3 und Mittel:", {k: np.round(v.to_numpy(), 3).tolist() for k, v in gain.items()})
    mean = {k: float(v["Mittel"]) for k, v in gain.items()}
    lever = "etwa gleich" if abs(mean["256 Tokens"] - mean["längerer Lernratenplan"]) < 0.03 else max(mean, key=mean.get)
    check_guess("Was holt mehr bei gleicher Schrittzahl", guess, lever)


def epochs_and_times_800(ref: dict, proba: dict, ft_test, y_te, sec_short: float, sec_plan: float) -> None:
    """Testwerte je Epoche im 8-Epochen-Plan, Vergleich mit allen Trainingsparagraphen, Zeit je Schritt."""
    def f1_test(p):
        return d.metrics(y_te, p)["macro_f1"]

    test_per_epoch = pd.DataFrame({f"Seed {sd}": [v["macro_f1"] for v in ref[f"klein_s{sd}"]["verlauf"] if v["teil"] == "test"]
                                   for sd in SEEDS}, index=pd.RangeIndex(1, 9, name="Epoche"))
    print("8-Epochen-Plan: Macro-F1 auf dem Test je Epoche (nur zum Ansehen, gewählt wird auf der Validierung)\n",
          test_per_epoch.T.round(3).to_string())
    full = {"Fine-Tuning": np.mean([f1_test(p) for p in ft_test]), "TF-IDF + LR": f1_test(proba["tfidf_test"].argmax(1))}
    print(f"\nmit allen 6893 Trainingsparagraphen (Demo): Fine-Tuning {full['Fine-Tuning']:.3f} (Plan über 3 Epochen, "
          f"Epoche auf der Validierung, Mittel über 3 Seeds), TF-IDF + LR {full['TF-IDF + LR']:.3f}")
    print(f"Zeit je Schritt, hier gemessen: kurzes Rezept {sec_short / 100:.2f} s, Plan-Variante {sec_plan / 100:.2f} s (je 128 Tokens)")
    print("Zeit je Schritt in der Ablage (Trainerrechner, CPU, Seed 1):",
          ", ".join(f"{n} {ref[k]['sekunden_training'] / 100:.2f} s" for n, k in
                    (("kurzes Rezept", "klein_kurz_s1"), ("Plan-Variante", "klein_plan_s1"), ("Token-Variante (256 Tokens)", "klein_tok_s1"))))


def folds_800(df: pd.DataFrame, proba: dict) -> None:
    """Vorbereitete Läufe mit 800 Paragraphen auf den fünf Gesetzes-Folds: Macro-F1 über alle Paragraphen
    (Out-of-Fold) und Bootstrap über ganze Gesetze, je Rechtsgebiet so viele, wie es hat."""
    y = df.y.values
    rows_per = {g: np.flatnonzero(df.gesetz.values == g) for g in df.gesetz.unique()}
    per_area = [[g for g, z in rows_per.items() if y[z[0]] == k] for k in range(y.max() + 1)]
    rng = np.random.default_rng(SEED)
    folds = []
    for sd in SEEDS:
        oof = {n: np.zeros(len(df), dtype=int) for n in ("Fine-Tuning", "TF-IDF + LR")}
        for f, te in enumerate(d.group_folds(df)):
            oof["Fine-Tuning"][te] = proba[f"klein_cv_s{sd}_f{f}_test"].argmax(1)
            oof["TF-IDF + LR"][te] = proba[f"klein_cv_tfidf_s{sd}_f{f}_test"].argmax(1)
        diffs = []
        for _ in range(2000):
            draw = np.concatenate([rows_per[g] for laws in per_area for g in rng.choice(laws, len(laws))])
            diffs.append(d.macro_f1_fast(y[draw], oof["Fine-Tuning"][draw]) - d.macro_f1_fast(y[draw], oof["TF-IDF + LR"][draw]))
        f1 = {n: d.macro_f1_fast(y, p) for n, p in oof.items()}
        folds.append({"Seed": sd, **f1, "FT minus TF-IDF": f1["Fine-Tuning"] - f1["TF-IDF + LR"],
                      "95-%-Intervall über Gesetze": np.percentile(diffs, [2.5, 97.5]).round(3)})
    print("800 Paragraphen je Fold, Macro-F1 über alle Paragraphen (Out-of-Fold)")
    print(pd.DataFrame(folds).round(3).to_string(index=False))
