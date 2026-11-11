"""Lab 6, Demo Abschnitte 4 und 5, Übung Aufgaben 2 und 3: GGUF schreiben und die llama.cpp-Werkzeuge aufrufen.

llama.cpp liefert für Schritt 1 (Hugging Face nach GGUF in 16 Bit) das Skript
convert_hf_to_gguf.py aus dem Quellbaum. Homebrew und das pip-Paket `gguf`
bringen es nicht mit. `hf_to_gguf_f16` erledigt denselben Schritt für
Llama-artige Modelle (SmolLM2) mit dem Paket `gguf`, auf dem auch das
Original-Skript aufbaut: Hyperparameter und Tokenizer als Metadaten,
Gewichte unter GGUF-Namen, Matrizen in F16, Normen in F32.

Die übrigen Funktionen rufen llama-quantize, llama-perplexity und llama-bench
auf und lesen deren Ausgabe.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


def threads() -> list[str]:
    """`-t N` aus der Umgebungsvariable LLAMA_THREADS; ohne sie wählt llama.cpp selbst (alle Kerne)."""
    n = os.environ.get("LLAMA_THREADS")
    return ["-t", n] if n else []


def opts() -> list[str]:
    """Optionen für llama-perplexity und llama-bench: nur CPU (`-ngl 0`, auch auf Rechnern mit Metal/CUDA)
    und `-t` aus LLAMA_THREADS. LLAMA_NGL überschreibt die Zahl der GPU-Schichten."""
    return ["-ngl", os.environ.get("LLAMA_NGL", "0"), *threads()]


def llama_tools_available() -> bool:
    return all(shutil.which(t) for t in ("llama-quantize", "llama-perplexity", "llama-bench"))


def check_space(folder: Path, needs_mb: float, purpose: str) -> None:
    """Bricht mit klarer Meldung ab, wenn im Ordner weniger als `needs_mb` MB frei sind."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(folder).free / 1e6
    if free < needs_mb:
        def mb(x):
            return f"{x:,.0f}".replace(",", ".")
        raise RuntimeError(
            f"Zu wenig Plattenplatz für {purpose}: frei {mb(free)} MB, gebraucht rund {mb(needs_mb)} MB "
            f"in {folder}. Dort liegen schon {mb(mb_in_dir(folder))} MB aus diesem Lab; die Aufräumzelle am Ende "
            "der Übung löscht alles außer F16 und Q4_K_M. Ohne Platz rechnen Sie mit TOOLS = False weiter, "
            "dann lesen die Zellen die Referenzmessung.")


def mb_in_dir(folder: Path) -> float:
    return sum(p.stat().st_size for p in Path(folder).rglob("*") if p.is_file()) / 1e6


def _permute(w: np.ndarray, n_head: int) -> np.ndarray:
    # Gleiche Umordnung der q/k-Zeilen wie convert_hf_to_gguf.py (RoPE-Layout von llama.cpp)
    return w.reshape(n_head, 2, w.shape[0] // n_head // 2, *w.shape[1:]).swapaxes(1, 2).reshape(w.shape)


def hf_to_gguf_f16(model_dir: Path, out: Path) -> Path:
    """Schreibt ein Llama-artiges HF-Modell als GGUF (Matrizen F16, Normen F32)."""
    import gguf
    import torch
    from safetensors.torch import load_file

    model_dir, out = Path(model_dir), Path(out)
    cfg = json.loads((model_dir / "config.json").read_text())
    tok = json.loads((model_dir / "tokenizer.json").read_text())
    n_layer, n_head, n_kv = cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"]

    w = gguf.GGUFWriter(str(out), "llama")
    w.add_name(model_dir.name)
    w.add_context_length(cfg["max_position_embeddings"])
    w.add_embedding_length(cfg["hidden_size"])
    w.add_block_count(n_layer)
    w.add_feed_forward_length(cfg["intermediate_size"])
    w.add_head_count(n_head)
    w.add_head_count_kv(n_kv)
    w.add_rope_dimension_count(cfg["hidden_size"] // n_head)
    # transformers 5 speichert rope_theta unter rope_parameters
    w.add_rope_freq_base(float(cfg.get("rope_theta") or cfg["rope_parameters"]["rope_theta"]))
    w.add_layer_norm_rms_eps(cfg["rms_norm_eps"])
    w.add_vocab_size(cfg["vocab_size"])
    w.add_file_type(gguf.LlamaFileType.MOSTLY_F16)

    # Tokenizer: Byte-Level-BPE wie GPT-2, Vorzerlegung wie SmolLM (Ziffern einzeln)
    vocab = tok["model"]["vocab"]
    special = {a["id"] for a in tok["added_tokens"] if a.get("special")}
    tokens = [None] * cfg["vocab_size"]
    for t, i in vocab.items():
        tokens[i] = t
    types = [gguf.TokenType.CONTROL if i in special else gguf.TokenType.NORMAL for i in range(len(tokens))]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("smollm")
    w.add_token_list(tokens)
    w.add_token_types(types)
    gguf.SpecialVocab(model_dir, load_merges=True, n_vocab=len(tokens)).add_to_gguf(w)

    names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.LLAMA, n_layer)
    sd = load_file(str(model_dir / "model.safetensors"))
    for name, t in sd.items():
        if name == "lm_head.weight" and cfg.get("tie_word_embeddings"):
            continue
        a = t.to(torch.float32).numpy()
        if name.endswith("q_proj.weight"):
            a = _permute(a, n_head)
        elif name.endswith("k_proj.weight"):
            a = _permute(a, n_kv)
        new = names.get_name(name, try_suffixes=(".weight", ".bias"))
        w.add_tensor(new, a.astype(np.float16) if a.ndim > 1 else a.astype(np.float32))

    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return out


def run(cmd: list[str], timeout: int = 900) -> str:
    """Führt ein llama.cpp-Werkzeug aus und liefert stdout+stderr als Text."""
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} endete mit Code {r.returncode}:\n{r.stderr[-2000:]}")
    return r.stdout + r.stderr


def parse_ppl(log: str) -> tuple[float, float]:
    """PPL und Unsicherheit aus 'Final estimate: PPL = 12.3456 +/- 0.1234'."""
    m = re.search(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)", log)
    if not m:
        raise ValueError("Keine PPL in der Ausgabe gefunden")
    return float(m.group(1)), float(m.group(2))


def parse_kld(log: str) -> dict:
    """Kennzahlen aus llama-perplexity --kl-divergence (gepaart gegen die Referenz-Logits)."""
    def grab(label: str) -> tuple[float, float]:
        m = re.search(rf"{label}\s*:\s*([-\d.]+)\s*±\s*([\d.]+)", log)
        if not m:
            raise ValueError(f"'{label}' nicht gefunden")
        return float(m.group(1)), float(m.group(2))

    out = {}
    for key, label in [("ppl", r"Mean\s+PPL\(Q\)"), ("ppl_base", r"Mean\s+PPL\(base\)"),
                       ("ln_ratio", r"Mean\s+ln\(PPL\(Q\)/PPL\(base\)\)"), ("kld", r"Mean\s+KLD"),
                       ("same_top", r"Same top p")]:
        out[key], out[key + "_se"] = grab(label)
    return out


def parse_bench(stdout: str) -> list[dict]:
    """llama-bench -o json: eine Zeile je Test (pp = Prompt, tg = Generierung)."""
    rows, _ = json.JSONDecoder().raw_decode(stdout[stdout.index("["):])
    return [{"test": "pp" if r["n_prompt"] else "tg", "n": r["n_prompt"] or r["n_gen"],
             "t_s": r["avg_ts"], "sd": r["stddev_ts"], "threads": r["n_threads"]} for r in rows]


def tensor_bytes_by_type(path: Path) -> dict:
    """Summiert Parameter und Bytes je Tensortyp einer GGUF-Datei."""
    import gguf

    r = gguf.GGUFReader(str(path))
    out: dict[str, dict] = {}
    for t in r.tensors:
        d = out.setdefault(t.tensor_type.name, {"tensoren": 0, "parameter": 0, "bytes": 0})
        d["tensoren"] += 1
        d["parameter"] += int(t.n_elements)
        d["bytes"] += int(t.n_bytes)
    return out


def torch_ppl_like_llama(model, tok, text: str, n_ctx: int = 256, chunks: int = 20) -> float:
    """PPL nach dem Schema von llama-perplexity: Fenster zu n_ctx Token, bewertet wird die zweite Hälfte."""
    import torch

    ids = tok(text, verbose=False)["input_ids"]
    chunks = min(chunks, len(ids) // n_ctx)
    x = torch.tensor(ids[: chunks * n_ctx]).view(chunks, n_ctx)
    first = n_ctx // 2
    nll = []
    with torch.no_grad():                       # je 5 Fenster, damit die Logits klein bleiben
        for xb in x.split(5):
            lp = torch.log_softmax(model(input_ids=xb).logits.float(), -1)
            nll.append(-lp[:, first : n_ctx - 1].gather(-1, xb[:, first + 1 :, None]).squeeze(-1))
    return float(torch.exp(torch.cat(nll).mean()))


def reference(path: Path) -> dict:
    return json.loads(Path(path).read_text())


# --- Holdout und vorbereitete Referenzmessung ---------------------------------------------------

STAGES = ["F16", "Q8_0", "Q5_K_M", "Q4_K_M"]                      # Demo
MORE_TYPES = ["Q6_K", "Q4_K_S", "Q4_0", "Q3_K_M", "Q2_K"]          # Übung


def holdout_text(df, laws, chars: int = 4_000) -> str:
    """Die ersten `chars` Zeichen jedes Testgesetzes, durch Leerzeilen getrennt."""
    return "\n\n".join("\n\n".join(df[df.gesetz == g].text)[:chars] for g in laws)


def platform_name() -> str:
    """CPU-Bezeichnung und Kernzahl, für die Einordnung von Tokens/s."""
    import platform

    name = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            name = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        elif Path("/proc/cpuinfo").exists():
            name = next(l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                        if l.startswith("model name"))
    except Exception:
        pass
    return f"{name}, {os.cpu_count()} logische Kerne"


def prepare_reference(out_json: Path, workdir: Path) -> dict:
    """Ganze Messreihe einmal ablegen: Größe, PPL, KLD, Tokens/s, Tensortypen je Stufe.

    Je Stufe ein Lauf mit --kl-divergence (liefert PPL und gepaarten Vergleich zugleich);
    jede Stufe wird nach der Messung gelöscht, die Logit-Datei am Ende auch bei einem Fehler.
    """
    from common import corpus
    from common.lab6_lora import MODEL_DIR, TEST_LAWS

    check_space(workdir, 900, "die Referenzmessung")
    text = workdir / "holdout.txt"
    text.write_text(holdout_text(corpus.load(), TEST_LAWS), encoding="utf-8")
    f16 = hf_to_gguf_f16(MODEL_DIR, workdir / "smollm2-135m-F16.gguf")
    base = workdir / "base.kld"
    res = {}
    try:
        log = run(["llama-perplexity", "-m", str(f16), "-f", str(text), "-c", "256",
                   "--kl-divergence-base", str(base), *opts()])
        for q in STAGES + MORE_TYPES:
            path = workdir / f"smollm2-135m-{q}.gguf"
            if q == "F16":
                ppl, se = parse_ppl(log)
                kld = None
            else:
                run(["llama-quantize", str(f16), str(path), q, *(threads()[1:] or [])])
                kld = parse_kld(run(["llama-perplexity", "-m", str(path), "--kl-divergence-base", str(base),
                                     "--kl-divergence", "-c", "256", *opts()]))
                ppl, se = kld["ppl"], kld["ppl_se"]
            bench = parse_bench(run(["llama-bench", "-m", str(path), "-p", "512", "-n", "64", "-r", "5",
                                     "-o", "json", *opts()]))
            res[q] = {"bytes": path.stat().st_size, "ppl": ppl, "ppl_se": se, "kld": kld, "bench": bench,
                      "typen": tensor_bytes_by_type(path)}
            if q != "F16":
                path.unlink()
            print(q, res[q]["bytes"], ppl, kld and round(kld["ln_ratio"], 4), flush=True)
    finally:
        base.unlink(missing_ok=True)
        f16.unlink(missing_ok=True)
    out = {"plattform": platform_name(),
           "threads": "nur CPU (-ngl 0), Threads " + os.environ.get("LLAMA_THREADS", "nach llama.cpp-Standard"),
           "stufen": res}
    out_json.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    from common.paths import ARTIFACTS_DIR, LAB_DIR

    prepare_reference(LAB_DIR / "reference" / "lab6_quant.json", ARTIFACTS_DIR / "lab6_ref")
