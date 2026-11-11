"""Prüf- und Vergleichsfunktionen für die Lab-7-Übung.

Die Übung ruft sie nach Ihrer Lösung auf. Sie prüfen die Tipps und stellen Ihre Werte neben die
Referenzmessung des Trainers; lesen müssen Sie sie nicht.
"""
import pandas as pd
from openai import OpenAI

from common import lab7_serving as lab7
from common.interactive import check_guess


def check_queue(ttft_factor_16, throughput_factor_16, ref, guess_ttft, guess_throughput):
    """Aufgabe 1: Faktoren 16 / 8 Clients prüfen, die Faktoren der Referenz daneben."""
    check_guess("TTFT 16 / 8 Clients", guess_ttft, ttft_factor_16, tolerance=0.3 * ttft_factor_16, fmt="{:.2f}")
    check_guess("Durchsatz 16 / 8 Clients", guess_throughput, throughput_factor_16,
                tolerance=0.15 * throughput_factor_16, fmt="{:.2f}")
    if ref:
        r = pd.DataFrame(ref["messreihe"]["stufen"]).set_index("concurrency")
        print(f"Referenz: TTFT x{r.loc[16, 'TTFT p50 [s]'] / r.loc[8, 'TTFT p50 [s]']:.2f}, "
              f"Durchsatz x{r.loc[16, 'Durchsatz [Token/s]'] / r.loc[8, 'Durchsatz [Token/s]']:.2f}")


def control_16_slots(run_df, prompts):
    """Aufgabe 1, Kontrolle: zweiter Server mit 16 Slots, misst 8 und 16 Clients, beide Server nebeneinander."""
    offset = run_df[run_df.concurrency == 1].e2e.median()
    with lab7.LlamaServer(parallel=16, ctx_per_slot=2048, port=8081) as srv16:
        c16 = OpenAI(base_url=srv16.url + "/v1", api_key="sk-no-key-required", timeout=600, max_retries=0)
        lab7.request(c16, prompts[-1], max_tokens=8)          # Aufwärmen
        run16 = pd.concat([pd.DataFrame(lab7.load_test(c16, prompts, c, offset=offset)) for c in (8, 16)])
    for name, df in [("8 Slots", run_df[run_df.concurrency >= 8]), ("16 Slots", run16)]:
        t = lab7.table(df)[["TTFT p50 [s]", "TPOT p50 [ms]", "Durchsatz [Token/s]"]]
        t["Prefill p50 [s]"] = df.groupby("concurrency").server_prompt_s.median()
        print(name, "\n", t.round(3).to_string(), "\n")


def check_capacity(gp_own, operating_point, users_chat, users_summary, guess):
    """Aufgabe 2: Tipp prüfen, Nutzerzahlen und die Stufen mit fast gleichem Goodput zeigen."""
    check_guess("Chat trägt mindestens so viele Nutzer", guess, bool(users_chat >= users_summary))
    print(f"Chat {users_chat:.0f} Nutzer am Arbeitspunkt {operating_point}, Zusammenfassungsdienst {users_summary:.0f} Nutzer")
    equivalent = [int(c) for c, g in gp_own["Goodput [Anfragen/min]"].items()
                  if g >= 0.9 * gp_own["Goodput [Anfragen/min]"].max()]
    print("Stufen mit mindestens 90 % des besten Chat-Goodputs (im Rahmen der Messgenauigkeit gleichwertig):", equivalent)


def check_prompt_length(ttft_factor_long, e2e_factor_4x, guess_ttft, guess_e2e):
    """Aufgabe 3: Faktoren für den langen Prompt und die vierfache Antwortlänge prüfen."""
    check_guess("TTFT lang / kurz", guess_ttft, ttft_factor_long, tolerance=0.3 * ttft_factor_long, fmt="{:.2f}")
    check_guess("E2E 4x Antwort / normal", guess_e2e, e2e_factor_4x, tolerance=0.2 * e2e_factor_4x, fmt="{:.2f}")


def reference_prompt_length(ref):
    """Aufgabe 3: die Faktoren der Referenzmessung."""
    if ref:
        rp = ref["prompt_laenge"]
        print(f"Referenz: TTFT x{rp['lang']['ttft_p50'] / rp['kurz']['ttft_p50']:.2f}, "
              f"E2E x{rp['kurz_4x']['e2e_p50'] / rp['kurz']['e2e_p50']:.2f}, "
              f"TTFT bei 4x Antwort x{rp['kurz_4x']['ttft_p50'] / rp['kurz']['ttft_p50']:.2f}")


def reference_gauge(ref):
    """Aufgabe 4: Gauge, Zähler, Clients und 1 / TPOT der Referenzmessung."""
    if ref:
        rm = ref["metriken_c8"]
        print(f"Referenz: Gauge nach dem Lauf {rm['gauge_predicted_tokens_seconds']:.1f}, Zähler nur über den Lauf "
              f"{rm['zaehler_token']:.0f} / {rm['zaehler_decode_s']:.1f} s = {rm['zaehler_token'] / rm['zaehler_decode_s']:.1f}, "
              f"Clients {rm['durchsatz_client']:.1f}, 1 / TPOT {1000 / rm['tpot_p50_ms']:.1f} Token/s")
