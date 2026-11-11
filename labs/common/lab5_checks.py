"""Vorbereitungs- und Vergleichsfunktionen für die Lab-5-Übung. Sie rechnen die Reranker-Scores vor und stellen
Ihre Messungen neben die Pipeline der Demo; lesen müssen Sie sie nicht."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from common import lab5_rag as lab5
from common.interactive import explore


def rerank_scores(reranker, ix: lab5.Index, questions: list[dict], n_max: int = 30) -> dict:
    """Aufgabe 2 und 3: Reranker-Scores für die ersten n_max dichten Treffer jeder Frage.
    Ergebnis: Frage-ID -> (Kandidaten, Scores), Scores in der Reihenfolge der Kandidaten."""
    rerank = {}
    t0 = time.perf_counter()
    for f in questions:
        candidate_ids = ix.dense(f["frage"], n_max)
        rerank[f["id"]] = (candidate_ids, reranker.predict([(f["frage"], ix.text_of[c]) for c in candidate_ids], batch_size=32))
    print(f"Scores für {len(rerank)} Fragen mit je {n_max} Kandidaten in {time.perf_counter() - t0:.0f} s")
    return rerank


def _demo_answer(f: dict, ix: lab5.Index, cl, model: str) -> tuple[list[str], str]:
    """Kontext (fünf dichte Treffer) und Antwort des Generators wie in der Demo, aus dem Cache."""
    context = ix.dense(f["frage"], 5)
    return context, lab5.generate(cl, model, lab5.compose(f["frage"], context, ix.text_of))


def explore_threshold(rerank: dict, ix: lab5.Index, gold_questions: list[dict], open_questions: list[dict],
                      cl, model: str) -> None:
    """Aufgabe 3: Regler für eine Schwelle auf den höchsten Reranker-Score. Zeigt, wie viele unbeantwortbare Fragen
    sie abfängt und welche Ausgänge der Demo-Pipeline sie bei den beantwortbaren Fragen abweist."""
    highest = pd.Series({qid: float(np.max(s)) for qid, (_, s) in rerank.items()})
    outcome = {}
    for f in gold_questions:
        _, answer = _demo_answer(f, ix, cl, model)
        outcome[f["id"]] = ("Nicht in den Quellen" if lab5.is_non_answer(answer) else
                            "zitiert Gold" if any(lab5.norm_of(c) in f["gold"] for c in lab5.cited_ids(answer)) else "ohne Gold-Beleg")
    outcome = pd.Series(outcome)
    open_ids = [f["id"] for f in open_questions]

    def show(threshold):
        removed = outcome[highest[outcome.index] < threshold]
        print(f"Schwelle {threshold:+.2f}: fängt {(highest[open_ids] < threshold).sum()} von 8 unbeantwortbaren ab, "
              f"weist {len(removed)} von 33 beantwortbaren ab")
        print(f"  davon zitierte die Pipeline die Gold-Norm: {(removed == 'zitiert Gold').sum()} von {(outcome == 'zitiert Gold').sum()}, "
              f"ohnehin 'Nicht in den Quellen': {(removed == 'Nicht in den Quellen').sum()}, ohne Gold-Beleg: {(removed == 'ohne Gold-Beleg').sum()}")

    explore(show, threshold=(-0.2, -7.0, 9.0, 0.05))


def demo_answers(ix: lab5.Index, gold_questions: list[dict], cl, model: str) -> pd.DataFrame:
    """Aufgabe 4: die inhaltlichen Antworten der Demo mit Kontext, Judge-Urteil und Kontrollurteil.
    Spalten: kontext, antwort, judge_gestuetzt, kontext_traegt, kern; Index: Frage-ID."""
    rows = []
    for f in gold_questions:
        context, answer = _demo_answer(f, ix, cl, model)
        if lab5.is_non_answer(answer):
            continue
        claims = lab5.judge(cl, model, f["frage"], context, answer, ix.text_of)
        rows.append({"id": f["id"], "kontext": context, "antwort": answer, "judge_gestuetzt": lab5.fully_supported(claims)})
    run_df = pd.DataFrame(rows, columns=["id", "kontext", "antwort", "judge_gestuetzt"]).set_index("id")
    control = pd.DataFrame(lab5.load_control_verdicts()["urteile"]).set_index("id")
    run_df = run_df.join(control[["kontext_traegt", "kern"]])
    print(f"{len(run_df)} inhaltliche Antworten, davon laut LLM-Judge voll gestützt: {run_df.judge_gestuetzt.sum()}, "
          f"laut Kontrolle vom Kontext getragen: {run_df.kontext_traegt.sum()}")
    return run_df
