#!/usr/bin/env bash
# Richtet einen Kursrechner mit Debian 13 ein: Python 3.11, Pakete, llama.cpp, Korpus und Modelle.
#
# Aufruf aus dem Pack-Root, als der Benutzer, mit dem die Teilnehmenden arbeiten:
#     bash scripts/setup_debian13.sh                 # alles (braucht Netz, belegt rund 5 GB)
#     bash scripts/setup_debian13.sh --ohne-modelle  # Korpus und Modelle liegen schon in labs/
#
# Python und .venv liegen im Pack-Ordner. Der Ordner darf danach nicht mehr umziehen, sonst
# findet .venv sein Python nicht. llama.cpp kommt nach /opt und /usr/local/bin und ist damit
# für alle Benutzer und jeden Notebook-Kernel im PATH. Dafür und für apt braucht das Skript sudo.
# Mehrfach aufrufen schadet nicht.
set -euo pipefail

LLAMA_BUILD=b9290
PYTHON_VERSION=3.11
TORCH_VERSION=2.14.1

PACK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PACK_DIR"

MODELLE=1
for arg in "$@"; do
  case "$arg" in
    --ohne-modelle) MODELLE=0 ;;
    *) echo "Unbekannte Option: $arg" >&2; exit 2 ;;
  esac
done

schritt() { printf '\n=== %s\n' "$1"; }

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
  echo "Hinweis: Das Skript läuft als root. Geben Sie den Pack-Ordner danach dem Kursbenutzer"
  echo "(chown -R), die Labs schreiben nach labs/artifacts, labs/mlruns und labs/models."
else
  SUDO="sudo"
fi

schritt "Rechner prüfen"
if [ -r /etc/os-release ]; then
  . /etc/os-release
  echo "System: ${PRETTY_NAME:-unbekannt}"
  if [ "${ID:-}" != "debian" ] || [ "${VERSION_ID:-}" != "13" ]; then
    echo "Hinweis: Das Skript ist für Debian 13 geschrieben, hier läuft etwas anderes."
  fi
fi
frei_gb=$(df -BG --output=avail "$PACK_DIR" | tail -1 | tr -dc '0-9')
echo "Frei im Pack-Ordner: ${frei_gb} GB"
if [ "$frei_gb" -lt 15 ]; then
  echo "Zu wenig Platz: Die Labs brauchen mindestens 15 GB." >&2
  exit 1
fi

case "$(uname -m)" in
  x86_64)
    LLAMA_ASSET="llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz"
    LLAMA_SHA256=4249e4d3f5b5dbb7e924b4abac6253088744a88328c6a9068cac385bb7b2a60c ;;
  aarch64)
    LLAMA_ASSET="llama-${LLAMA_BUILD}-bin-ubuntu-arm64.tar.gz"
    LLAMA_SHA256=26f1a821ad45170fca0f122feda41e2747180b620f43c859d73b6993ce8c2402 ;;
  *) echo "Nicht unterstützte Architektur: $(uname -m)" >&2; exit 1 ;;
esac

schritt "Systempakete"
$SUDO apt-get update -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq curl ca-certificates tar libgomp1

schritt "Python ${PYTHON_VERSION} (Debian 13 bringt nur 3.13 mit)"
export UV_UNMANAGED_INSTALL="$PACK_DIR/.tools/uv"
export UV_PYTHON_INSTALL_DIR="$PACK_DIR/.tools/python"
if [ ! -x "$UV_UNMANAGED_INSTALL/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV="$UV_UNMANAGED_INSTALL/uv"
"$UV" python install "$PYTHON_VERSION"
if [ ! -x .venv/bin/python ]; then
  "$UV" venv --python "$PYTHON_VERSION" --python-preference only-managed --seed .venv
fi
.venv/bin/python --version

schritt "Python-Pakete (torch als CPU-Variante)"
.venv/bin/pip install --quiet "torch==${TORCH_VERSION}" --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install --quiet -r requirements.txt

schritt "llama.cpp ${LLAMA_BUILD}"
LLAMA_DIR="/opt/llama-${LLAMA_BUILD}"
if [ ! -x "$LLAMA_DIR/llama-server" ]; then
  tmp="$(mktemp -d)"
  curl -fL --retry 3 -o "$tmp/$LLAMA_ASSET" \
    "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/${LLAMA_ASSET}"
  echo "${LLAMA_SHA256}  $tmp/$LLAMA_ASSET" | sha256sum -c -
  tar -xzf "$tmp/$LLAMA_ASSET" -C "$tmp"
  rm "$tmp/$LLAMA_ASSET"
  quelle="$(dirname "$(find "$tmp" -type f -name llama-server | head -1)")"
  $SUDO mkdir -p "$LLAMA_DIR"
  $SUDO cp -a "$quelle/." "$LLAMA_DIR/"
  rm -rf "$tmp"
fi
for prog in llama-server llama-quantize llama-bench llama-perplexity; do
  $SUDO ln -sf "$LLAMA_DIR/$prog" "/usr/local/bin/$prog"
done
llama-server --version 2>&1 | head -2

cd labs
if [ "$MODELLE" -eq 1 ]; then
  schritt "Korpus und Modelle"
  ../.venv/bin/python prefetch.py
fi
../.venv/bin/python prefetch.py --check

schritt "Setup-Check"
if ../.venv/bin/python _src/lab0_setup_check.py; then
  echo
  echo "Fertig. Notebooks öffnen: cd labs && ../.venv/bin/jupyter lab"
else
  echo
  echo "Der Setup-Check meldet ROT. Ohne labs/.env ist genau diese eine Zeile erwartet:"
  echo "Lab 5 läuft dann aus dem Cache. Alles andere bitte vor dem ersten Lab beheben."
fi
