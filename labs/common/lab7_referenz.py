"""Schreibt die Referenzmessung für Lab 7 nach labs/reference/lab7_referenz.json.

Aufruf im Ordner labs/ auf einem Rechner ohne andere Last (dauert etwa 2 bis 3 Minuten):
    ../.venv/bin/python -m common.lab7_referenz
Optional: AI320_LAB7_HINWEIS="..." landet als Hinweis in der Datei (z. B. "vorläufig, unter Last").

Gemessen wird dasselbe wie in Demo und Übung: Messreihe mit Concurrency 1 bis 16 gegen
8 Slots, Einzelanfragen mit kurzem und langem Prompt und mit längerer Antwort, und der
Wert des Gauges llamacpp:predicted_tokens_seconds nach einem Lauf mit 8 Clients.
Am Ende stehen die Zahlen, die in den Markdown-Zellen der Notebooks zitiert werden.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import statistics as st
import time

from openai import OpenAI

from common import lab7_serving as lab7

STAGES = (1, 2, 4, 8, 16)
COLUMNS = ["concurrency", "client", "runde", "ttft", "tpot", "e2e", "n_out", "n_prompt", "itl_max", "server_prompt_s"]


def one_by_one(client, prompts, max_tokens):
    z = [lab7.request(client, p, max_tokens=max_tokens) for p in prompts[:lab7.REPEATS]]
    return {"ttft_p50": st.median(x["ttft"] for x in z), "e2e_p50": st.median(x["e2e"] for x in z),
            "n_prompt_p50": st.median(x["n_prompt"] for x in z), "max_tokens": max_tokens}


def main() -> None:
    t0 = time.time()
    short, long = lab7.prompts(64, n_chars=400), lab7.prompts(16, n_chars=1600)
    with lab7.LlamaServer(parallel=8, ctx_per_slot=2048) as srv:
        client = OpenAI(base_url=srv.url + "/v1", api_key="sk-no-key-required", timeout=600, max_retries=0)
        lab7.request(client, short[-1], max_tokens=8)  # Aufwärmen
        df = lab7.sweep(client, short, stages=STAGES)
        tab = lab7.table(df)
        length = {"kurz": one_by_one(client, short, lab7.MAX_TOKENS),
                  "lang": one_by_one(client, long, lab7.MAX_TOKENS),
                  "kurz_4x": one_by_one(client, short, 4 * lab7.MAX_TOKENS)}
        m0 = lab7.server_metrics(srv.url)
        z8 = lab7.load_test(client, short, 8, offset=df[df.concurrency == 1].e2e.median())
        m1 = lab7.server_metrics(srv.url)
        wall = max(z["t_ende"] for z in z8) - min(z["t_send"] for z in z8)
        metric_values = {
            "gauge_predicted_tokens_seconds": m1["llamacpp:predicted_tokens_seconds"],
            "zaehler_token": m1["llamacpp:tokens_predicted_total"] - m0["llamacpp:tokens_predicted_total"],
            "zaehler_decode_s": m1["llamacpp:tokens_predicted_seconds_total"] - m0["llamacpp:tokens_predicted_seconds_total"],
            "wanduhr_s": wall,
            "durchsatz_client": lab7.throughput(z8),
            "tpot_p50_ms": 1000 * st.median(z["tpot"] for z in z8),
        }
    stages = [{"concurrency": int(c), **{k: float(v) for k, v in r.items()}} for c, r in tab.iterrows()]
    data = {
        "erstellt": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rechner": lab7.machine_info(),
        "einstellungen": {"modell": lab7.GGUF_PATH.name, "parallel": 8, "ctx_per_slot": 2048,
                          "max_tokens": lab7.MAX_TOKENS, "runden": lab7.ROUNDS, "wdh": lab7.REPEATS,
                          "threads": os.environ.get("AI320_LAB7_THREADS", "Vorgabe llama.cpp"),
                          "prompt_zeichen": 400, "prompt_zeichen_lang": 1600},
        "hinweis": os.environ.get("AI320_LAB7_HINWEIS", ""),
        "messreihe": {"stufen": stages, "anfragen": df[COLUMNS].round(5).to_dict("records")},
        "prompt_laenge": length,
        "metriken_c8": metric_values,
    }
    lab7.REFERENCE.parent.mkdir(parents=True, exist_ok=True)
    lab7.REFERENCE.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"geschrieben: {lab7.REFERENCE} ({time.time() - t0:.0f} s)\n")
    print(tab.round(3).to_string())
    print("\nZahlen für den Text der Notebooks:")
    r = {s["concurrency"]: s for s in stages}
    for c in (8, 16):
        print(f"  Concurrency 1 -> {c}: Durchsatz x{r[c]['Durchsatz [Token/s]'] / r[1]['Durchsatz [Token/s]']:.1f}, "
              f"TTFT p50 x{r[c]['TTFT p50 [s]'] / r[1]['TTFT p50 [s]']:.1f}, TPOT p50 x{r[c]['TPOT p50 [ms]'] / r[1]['TPOT p50 [ms]']:.1f}")
    print(f"  Langer Prompt: TTFT x{length['lang']['ttft_p50'] / length['kurz']['ttft_p50']:.1f}; "
          f"4x max_tokens: TTFT x{length['kurz_4x']['ttft_p50'] / length['kurz']['ttft_p50']:.2f}, "
          f"E2E x{length['kurz_4x']['e2e_p50'] / length['kurz']['e2e_p50']:.1f}")
    print(f"  Gauge {metric_values['gauge_predicted_tokens_seconds']:.1f} Token/s, Client {metric_values['durchsatz_client']:.1f} Token/s, "
          f"1/TPOT {1000 / metric_values['tpot_p50_ms']:.1f} Token/s")


if __name__ == "__main__":
    main()
