# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Setup-Check
#
# Prüft die Kursumgebung und endet mit einer Ampel-Liste. Aufruf im Terminal:
#
# ```
# cd labs
# python _src/lab0_setup_check.py
# ```
#
# Alles grün: fertig. Gelb: läuft, aber langsamer oder mit abweichender Version.
# Rot: vor dem ersten Lab beheben (der Hinweis in der Zeile sagt, wie).

# %%
import os
import sys
from pathlib import Path


def find_lab_dir() -> Path:
    """Sucht den Ordner labs/ (mit common/), egal ob als Skript, aus dem Pack-Root oder als Notebook gestartet."""
    starts = [Path.cwd()]
    if "__file__" in globals():
        starts.insert(0, Path(__file__).resolve().parent)
    for start in starts:
        for p in [start, *start.parents]:
            for candidate in (p, p / "labs"):
                if (candidate / "common" / "paths.py").exists():
                    return candidate
    raise SystemExit("Ordner labs/ mit common/ nicht gefunden: bitte im Kursordner starten.")


LAB_DIR = find_lab_dir()
os.chdir(LAB_DIR)
sys.path.insert(0, str(LAB_DIR))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # nur lokale Modelle, kein Hub-Zugriff
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

RESULTS: list[tuple[str, str, str]] = []              # (Ampel, Prüfung, Befund)


def report(light: str, check_name: str, finding: str) -> None:
    RESULTS.append((light, check_name, finding))
    print(f"{light:5s} {check_name:28s} {finding}")


print("Lab-Ordner:", LAB_DIR)

# %% [markdown]
# ## Python und Pakete

# %%
import importlib.metadata as md
import platform

v = sys.version_info
report("GRÜN" if (v.major, v.minor) == (3, 11) else "GELB", "Python",
      f"{platform.python_version()} ({platform.machine()}, {platform.system()})"
      + ("" if (v.major, v.minor) == (3, 11) else ", Kurs ist auf 3.11 getestet"))

# Stand der Trainer-Umgebung (Major.Minor); andere Versionen laufen oft, sind aber nicht getestet
REFERENCE = {
    "torch": "2.14", "transformers": "5.21", "tokenizers": "0.23", "peft": "0.21",
    "sentence-transformers": "6.2", "datasets": "4.0", "mlflow": "2.22", "datasketch": "2.0",
    "openai": "3.40", "pandas": "2.3", "pyarrow": "19.0", "numpy": "2.4", "matplotlib": "3.11",
    "scikit-learn": "1.9", "huggingface-hub": "1.33", "safetensors": "0.8", "accelerate": "1.15",
    "gguf": "0.19", "jupytext": "1.19", "nbclient": "0.11", "ipywidgets": "8.1", "ipykernel": "7.4",
    "rank-bm25": "0.2", "prometheus-client": "0.26", "httpx": "0.28", "pillow": "13.0", "joblib": "1.6",
}
for package, ref in REFERENCE.items():
    try:
        actual = md.version(package)
    except md.PackageNotFoundError:
        report("ROT", package, "fehlt: pip install -r requirements.txt")
        continue
    equal = ".".join(actual.split(".")[:2]) == ref
    report("GRÜN" if equal else "GELB", package, actual if equal else f"{actual} (Kurs: {ref}.x)")
try:
    import mlflow.store.db.utils  # noqa: F401
    report("GRÜN", "mlflow-Import", "ok")
except ImportError as errors:
    report("ROT", "mlflow-Import", f"{errors}: pip install -r requirements.txt")

# %% [markdown]
# ## device
#
# Alle Labs rechnen auf der CPU (`device = "cpu"`), damit die Zahlen auf jedem Rechner vergleichbar sind.

# %%
import torch

device = torch.device("cpu")
x = torch.ones(2, 2, device=device)
report("GRÜN" if float((x @ x).sum()) == 8.0 else "ROT", "device",
      f"{device}, {torch.get_num_threads()} Threads, {os.cpu_count()} logische Kerne")
extra = [n for n, ok in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available())) if ok]
print("      (zusätzlich vorhanden, in den Labs nicht genutzt:", ", ".join(extra) or "nichts", ")")

try:
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    report("GRÜN" if ram_gb >= 15 else "GELB", "Arbeitsspeicher",
          f"{ram_gb:.0f} GB" + ("" if ram_gb >= 15 else ", Kurs plant mit 16 GB"))
except (ValueError, OSError, AttributeError):
    report("GELB", "Arbeitsspeicher", "nicht ermittelbar")

# %% [markdown]
# ## Gesetzeskorpus

# %%
from common import corpus
from common.paths import NORMS_PARQUET

try:
    df = corpus.load()
    areas = df["rechtsgebiet"].nunique()
    ok = len(df) == 10_095 and areas == 5
    report("GRÜN" if ok else "GELB", "Gesetzeskorpus",
          f"{len(df):_} Normen aus {df['gesetz'].nunique()} Gesetzen, {areas} Rechtsgebiete".replace("_", ".")
          + ("" if ok else " (Kurs: 10.095 Normen, 5 Rechtsgebiete)"))
except FileNotFoundError:
    report("ROT", "Gesetzeskorpus", f"{NORMS_PARQUET.relative_to(LAB_DIR)} fehlt: python prefetch.py")

# %% [markdown]
# ## Modelle
#
# Lädt je Modell nur Konfiguration und Tokenizer aus `labs/models/` und prüft, ob die Gewichtsdatei da ist, ohne sie zu laden.

# %%
import json
import time

from transformers import AutoConfig, AutoTokenizer, BertTokenizer

from common.models import GGUF, MODELS, TOKENIZER_ONLY
from common.paths import MODELS_DIR

TEST_SENTENCE = "Die Würde des Menschen ist unantastbar."


def load_tokenizer(path: Path):
    """AutoTokenizer; ältere Repos ohne model_type in config.json (gbert-base) brauchen die Klasse ausdrücklich."""
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True), ""
    except ValueError:
        if not (path / "vocab.txt").exists():
            raise
        return BertTokenizer.from_pretrained(path, local_files_only=True), " (nur mit BertTokenizer)"


for name in MODELS:
    path = MODELS_DIR / name
    if not path.exists():
        report("ROT", f"Modell {name}", "fehlt: python prefetch.py")
        continue
    t0 = time.perf_counter()
    try:
        tok, hint = load_tokenizer(path)
        doc_type = json.loads((path / "config.json").read_text()).get("model_type") or "ohne model_type"
        if name not in TOKENIZER_ONLY:
            doc_type = AutoConfig.from_pretrained(path, local_files_only=True).model_type
            if not (list(path.glob("*.safetensors")) or (path / "pytorch_model.bin").exists()):
                report("ROT", f"Modell {name}", "Gewichte (*.safetensors oder pytorch_model.bin) fehlen: python prefetch.py")
                continue
        n = len(tok(TEST_SENTENCE)["input_ids"])
        report("GRÜN", f"Modell {name}",
              f"{doc_type}, Tokenizer ok{hint}, {n} Tokens im Testsatz, {time.perf_counter() - t0:.1f} s")
    except Exception as e:  # noqa: BLE001  (jede Ursache soll in der Ampel landen)
        report("ROT", f"Modell {name}", f"lädt nicht: {type(e).__name__}: {str(e).splitlines()[0][:80]}")

for name, (_, file) in GGUF.items():
    path = MODELS_DIR / name / file
    if not path.exists():
        report("ROT", f"GGUF {name}", "fehlt: python prefetch.py")
        continue
    try:
        from gguf import GGUFReader

        reader = GGUFReader(path)              # liest nur den Kopf, die Gewichte bleiben auf der Platte
        field = reader.fields["general.architecture"]
        arch = bytes(field.parts[field.data[0]]).decode()
        report("GRÜN", f"GGUF {name}", f"{arch}, {len(reader.tensors)} Tensoren, {path.stat().st_size / 1e6:.0f} MB")
    except Exception as e:  # noqa: BLE001
        report("ROT", f"GGUF {name}", f"lesbar? {type(e).__name__}: {str(e)[:80]}")

# %% [markdown]
# ## Generator-LLM und llama.cpp
#
# Der Schlüssel in `labs/.env` wird nur auf "gesetzt" geprüft, nie ausgegeben.

# %%
import shutil
import subprocess

env_file = LAB_DIR / ".env"
values = {}
if env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, value = line.split("=", 1)
            values[k.strip()] = value.strip()
key_set = bool(os.environ.get("LLM_API_KEY") or values.get("LLM_API_KEY"))
if not env_file.exists():
    report("ROT", "labs/.env", "fehlt: Datei vom Trainer nach labs/.env kopieren")
else:
    report("GRÜN" if key_set else "ROT", "LLM_API_KEY", "gesetzt: ja" if key_set else "gesetzt: nein (Schlüssel vom Trainer)")
    model = os.environ.get("LLM_MODEL") or values.get("LLM_MODEL")
    report("GRÜN" if model else "GELB", "LLM_MODEL", model or "nicht gesetzt")
del values

exe = shutil.which("llama-server")
if exe is None:
    report("ROT", "llama-server", "nicht im PATH: llama.cpp installieren (siehe README)")
else:
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
        line = next((z for z in (r.stdout + r.stderr).splitlines() if z.startswith("version")), "")
        report("GRÜN" if line else "GELB", "llama-server", line or f"gefunden ({exe}), Version nicht lesbar")
    except subprocess.TimeoutExpired:
        report("GELB", "llama-server", "antwortet nicht innerhalb von 20 s")

# %% [markdown]
# ## CPU-Messung
#
# Zwei kurze Messungen, beide auf `device = "cpu"`: eine Matrixmultiplikation und ein Trainingsschritt einer
# Transformer-Schicht. Die Referenz stammt vom Trainer-Rechner. Die Labs sind so geplant, dass sie auch auf
# einem drei- bis viermal langsameren Rechner in der Zeit bleiben. Maßgeblich für die Ampel ist der Trainingsschritt.

# %%
import statistics

# Trainer-Rechner (Apple M3 Max, 16 Kerne, torch 2.14.1, device cpu), gemessen am 27.10.2026
REF_MATMUL_MS = 4.1
REF_STEP_MS = 75.0


def median_ms(fn, repetitions: int, warmup_steps: int = 3) -> float:
    for _ in range(warmup_steps):
        fn()
    times = []
    for _ in range(repetitions):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.median(times)


torch.manual_seed(1)
a, b = torch.randn(1024, 1024, device=device), torch.randn(1024, 1024, device=device)
matmul_ms = median_ms(lambda: a @ b, repetitions=20)

layer = torch.nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.0, batch_first=True).to(device)
inp = torch.randn(16, 128, 256, device=device)
opt = torch.optim.AdamW(layer.parameters())


def train_step():
    opt.zero_grad()
    layer(inp).pow(2).mean().backward()
    opt.step()


step_ms = median_ms(train_step, repetitions=10)
print(f"Matmul 1024 x 1024: {matmul_ms:.1f} ms ({2 * 1024**3 / matmul_ms / 1e6:.0f} GFLOP/s), Trainer {REF_MATMUL_MS} ms")
factor = step_ms / REF_STEP_MS
light = "GRÜN" if factor <= 4 else ("GELB" if factor <= 6 else "ROT")
report(light, "CPU-Trainingsschritt",
      f"{step_ms:.0f} ms, Trainer {REF_STEP_MS:.0f} ms, also {factor:.1f}-mal so lang"
      + ("" if light == "GRÜN" else ": längere Läufe in den Labs kürzen (Trainer fragen)"))

# %% [markdown]
# ## Ampel

# %%
print()
print("=" * 72)
for color in ("ROT", "GELB", "GRÜN"):
    rows = [e for e in RESULTS if e[0] == color]
    print(f"{color}: {len(rows)}")
    if color != "GRÜN":
        for _, check_name, finding in rows:
            print(f"   {check_name}: {finding}")
print("=" * 72)
red = sum(e[0] == "ROT" for e in RESULTS)
print("Bereit für die Labs." if red == 0 else f"{red} Punkt(e) rot: bitte vor dem ersten Lab beheben.")
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    sys.exit(1 if red else 0)
