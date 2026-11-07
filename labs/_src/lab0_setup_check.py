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


def finde_lab_ordner() -> Path:
    """Sucht den Ordner labs/ (mit common/), egal ob als Skript, aus dem Pack-Root oder als Notebook gestartet."""
    starts = [Path.cwd()]
    if "__file__" in globals():
        starts.insert(0, Path(__file__).resolve().parent)
    for start in starts:
        for p in [start, *start.parents]:
            for kandidat in (p, p / "labs"):
                if (kandidat / "common" / "paths.py").exists():
                    return kandidat
    raise SystemExit("Ordner labs/ mit common/ nicht gefunden: bitte im Kursordner starten.")


LAB_DIR = finde_lab_ordner()
os.chdir(LAB_DIR)
sys.path.insert(0, str(LAB_DIR))
os.environ.setdefault("HF_HUB_OFFLINE", "1")          # nur lokale Modelle, kein Hub-Zugriff
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

ERGEBNIS: list[tuple[str, str, str]] = []             # (Ampel, Prüfung, Befund)


def melde(ampel: str, pruefung: str, befund: str) -> None:
    ERGEBNIS.append((ampel, pruefung, befund))
    print(f"{ampel:5s} {pruefung:28s} {befund}")


print("Lab-Ordner:", LAB_DIR)

# %% [markdown]
# ## Python und Pakete

# %%
import importlib.metadata as md
import platform

v = sys.version_info
melde("GRÜN" if (v.major, v.minor) == (3, 11) else "GELB", "Python",
      f"{platform.python_version()} ({platform.machine()}, {platform.system()})"
      + ("" if (v.major, v.minor) == (3, 11) else ", Kurs ist auf 3.11 getestet"))

# Stand der Trainer-Umgebung (Major.Minor); andere Versionen laufen oft, sind aber nicht getestet
REFERENZ = {
    "torch": "2.14", "transformers": "5.21", "tokenizers": "0.23", "peft": "0.21",
    "sentence-transformers": "6.2", "datasets": "4.0", "mlflow": "2.22", "datasketch": "2.0",
    "openai": "3.40", "pandas": "2.3", "pyarrow": "19.0", "numpy": "2.4", "matplotlib": "3.11",
    "scikit-learn": "1.9", "huggingface-hub": "1.33", "safetensors": "0.8", "accelerate": "1.15",
    "gguf": "0.19", "jupytext": "1.19", "nbclient": "0.11", "ipywidgets": "8.1", "ipykernel": "7.4",
    "rank-bm25": "0.2", "prometheus-client": "0.26", "httpx": "0.28", "pillow": "13.0", "joblib": "1.6",
}
for paket, ref in REFERENZ.items():
    try:
        ist = md.version(paket)
    except md.PackageNotFoundError:
        melde("ROT", paket, "fehlt: pip install -r requirements.txt")
        continue
    gleich = ".".join(ist.split(".")[:2]) == ref
    melde("GRÜN" if gleich else "GELB", paket, ist if gleich else f"{ist} (Kurs: {ref}.x)")
try:
    import mlflow.store.db.utils  # noqa: F401
    melde("GRÜN", "mlflow-Import", "ok")
except ImportError as fehler:
    melde("ROT", "mlflow-Import", f"{fehler}: pip install -r requirements.txt")

# %% [markdown]
# ## device
#
# Alle Labs rechnen auf der CPU (`device = "cpu"`), damit die Zahlen auf jedem Rechner vergleichbar sind.

# %%
import torch

device = torch.device("cpu")
x = torch.ones(2, 2, device=device)
melde("GRÜN" if float((x @ x).sum()) == 8.0 else "ROT", "device",
      f"{device}, {torch.get_num_threads()} Threads, {os.cpu_count()} logische Kerne")
extra = [n for n, ok in (("cuda", torch.cuda.is_available()), ("mps", torch.backends.mps.is_available())) if ok]
print("      (zusätzlich vorhanden, in den Labs nicht genutzt:", ", ".join(extra) or "nichts", ")")

try:
    ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024**3
    melde("GRÜN" if ram_gb >= 15 else "GELB", "Arbeitsspeicher",
          f"{ram_gb:.0f} GB" + ("" if ram_gb >= 15 else ", Kurs plant mit 16 GB"))
except (ValueError, OSError, AttributeError):
    melde("GELB", "Arbeitsspeicher", "nicht ermittelbar")

# %% [markdown]
# ## Gesetzeskorpus

# %%
from common import corpus
from common.paths import NORMEN_PARQUET

try:
    df = corpus.load()
    gebiete = df["rechtsgebiet"].nunique()
    ok = len(df) == 10_095 and gebiete == 5
    melde("GRÜN" if ok else "GELB", "Gesetzeskorpus",
          f"{len(df):_} Normen aus {df['gesetz'].nunique()} Gesetzen, {gebiete} Rechtsgebiete".replace("_", ".")
          + ("" if ok else " (Kurs: 10.095 Normen, 5 Rechtsgebiete)"))
except FileNotFoundError:
    melde("ROT", "Gesetzeskorpus", f"{NORMEN_PARQUET.relative_to(LAB_DIR)} fehlt: python prefetch.py")

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

TESTSATZ = "Die Würde des Menschen ist unantastbar."


def lade_tokenizer(pfad: Path):
    """AutoTokenizer; ältere Repos ohne model_type in config.json (gbert-base) brauchen die Klasse ausdrücklich."""
    try:
        return AutoTokenizer.from_pretrained(pfad, local_files_only=True), ""
    except ValueError:
        if not (pfad / "vocab.txt").exists():
            raise
        return BertTokenizer.from_pretrained(pfad, local_files_only=True), " (nur mit BertTokenizer)"


for name in MODELS:
    pfad = MODELS_DIR / name
    if not pfad.exists():
        melde("ROT", f"Modell {name}", "fehlt: python prefetch.py")
        continue
    t0 = time.perf_counter()
    try:
        tok, hinweis = lade_tokenizer(pfad)
        typ = json.loads((pfad / "config.json").read_text()).get("model_type") or "ohne model_type"
        if name not in TOKENIZER_ONLY:
            typ = AutoConfig.from_pretrained(pfad, local_files_only=True).model_type
            if not (list(pfad.glob("*.safetensors")) or (pfad / "pytorch_model.bin").exists()):
                melde("ROT", f"Modell {name}", "Gewichte (*.safetensors oder pytorch_model.bin) fehlen: python prefetch.py")
                continue
        n = len(tok(TESTSATZ)["input_ids"])
        melde("GRÜN", f"Modell {name}",
              f"{typ}, Tokenizer ok{hinweis}, {n} Tokens im Testsatz, {time.perf_counter() - t0:.1f} s")
    except Exception as e:  # noqa: BLE001  (jede Ursache soll in der Ampel landen)
        melde("ROT", f"Modell {name}", f"lädt nicht: {type(e).__name__}: {str(e).splitlines()[0][:80]}")

for name, (_, datei) in GGUF.items():
    pfad = MODELS_DIR / name / datei
    if not pfad.exists():
        melde("ROT", f"GGUF {name}", "fehlt: python prefetch.py")
        continue
    try:
        from gguf import GGUFReader

        reader = GGUFReader(pfad)              # liest nur den Kopf, die Gewichte bleiben auf der Platte
        feld = reader.fields["general.architecture"]
        arch = bytes(feld.parts[feld.data[0]]).decode()
        melde("GRÜN", f"GGUF {name}", f"{arch}, {len(reader.tensors)} Tensoren, {pfad.stat().st_size / 1e6:.0f} MB")
    except Exception as e:  # noqa: BLE001
        melde("ROT", f"GGUF {name}", f"lesbar? {type(e).__name__}: {str(e)[:80]}")

# %% [markdown]
# ## Generator-LLM und llama.cpp
#
# Der Schlüssel in `labs/.env` wird nur auf "gesetzt" geprüft, nie ausgegeben.

# %%
import shutil
import subprocess

env_datei = LAB_DIR / ".env"
werte = {}
if env_datei.exists():
    for zeile in env_datei.read_text(encoding="utf-8").splitlines():
        zeile = zeile.strip()
        if zeile and not zeile.startswith("#") and "=" in zeile:
            k, wert = zeile.split("=", 1)
            werte[k.strip()] = wert.strip()
schluessel_da = bool(os.environ.get("LLM_API_KEY") or werte.get("LLM_API_KEY"))
if not env_datei.exists():
    melde("ROT", "labs/.env", "fehlt: Datei vom Trainer nach labs/.env kopieren")
else:
    melde("GRÜN" if schluessel_da else "ROT", "LLM_API_KEY", "gesetzt: ja" if schluessel_da else "gesetzt: nein (Schlüssel vom Trainer)")
    modell = os.environ.get("LLM_MODEL") or werte.get("LLM_MODEL")
    melde("GRÜN" if modell else "GELB", "LLM_MODEL", modell or "nicht gesetzt")
del werte

exe = shutil.which("llama-server")
if exe is None:
    melde("ROT", "llama-server", "nicht im PATH: llama.cpp installieren (siehe README)")
else:
    try:
        r = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=20)
        zeile = next((z for z in (r.stdout + r.stderr).splitlines() if z.startswith("version")), "")
        melde("GRÜN" if zeile else "GELB", "llama-server", zeile or f"gefunden ({exe}), Version nicht lesbar")
    except subprocess.TimeoutExpired:
        melde("GELB", "llama-server", "antwortet nicht innerhalb von 20 s")

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
REF_SCHRITT_MS = 75.0


def median_ms(fn, wiederholungen: int, aufwaermen: int = 3) -> float:
    for _ in range(aufwaermen):
        fn()
    zeiten = []
    for _ in range(wiederholungen):
        t0 = time.perf_counter()
        fn()
        zeiten.append((time.perf_counter() - t0) * 1000)
    return statistics.median(zeiten)


torch.manual_seed(1)
a, b = torch.randn(1024, 1024, device=device), torch.randn(1024, 1024, device=device)
matmul_ms = median_ms(lambda: a @ b, wiederholungen=20)

schicht = torch.nn.TransformerEncoderLayer(256, 4, 1024, dropout=0.0, batch_first=True).to(device)
eingabe = torch.randn(16, 128, 256, device=device)
opt = torch.optim.AdamW(schicht.parameters())


def schritt():
    opt.zero_grad()
    schicht(eingabe).pow(2).mean().backward()
    opt.step()


schritt_ms = median_ms(schritt, wiederholungen=10)
print(f"Matmul 1024 x 1024: {matmul_ms:.1f} ms ({2 * 1024**3 / matmul_ms / 1e6:.0f} GFLOP/s), Trainer {REF_MATMUL_MS} ms")
faktor = schritt_ms / REF_SCHRITT_MS
ampel = "GRÜN" if faktor <= 4 else ("GELB" if faktor <= 6 else "ROT")
melde(ampel, "CPU-Trainingsschritt",
      f"{schritt_ms:.0f} ms, Trainer {REF_SCHRITT_MS:.0f} ms, also {faktor:.1f}-mal so lang"
      + ("" if ampel == "GRÜN" else ": längere Läufe in den Labs kürzen (Trainer fragen)"))

# %% [markdown]
# ## Ampel

# %%
print()
print("=" * 72)
for farbe in ("ROT", "GELB", "GRÜN"):
    zeilen = [e for e in ERGEBNIS if e[0] == farbe]
    print(f"{farbe}: {len(zeilen)}")
    if farbe != "GRÜN":
        for _, pruefung, befund in zeilen:
            print(f"   {pruefung}: {befund}")
print("=" * 72)
rot = sum(e[0] == "ROT" for e in ERGEBNIS)
print("Bereit für die Labs." if rot == 0 else f"{rot} Punkt(e) rot: bitte vor dem ersten Lab beheben.")
if __name__ == "__main__" and "ipykernel" not in sys.modules:
    sys.exit(1 if rot else 0)
