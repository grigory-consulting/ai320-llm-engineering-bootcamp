"""Lab 7: llama-server starten und beenden, gestreamte Anfragen messen, Last erzeugen.

Der Server läuft als Kindprozess eines kleinen Wächters (dieselbe Datei mit --watch).
Der Wächter beendet llama-server, sobald das Notebook beendet wird oder abstürzt,
damit kein Server-Prozess den Port belegt hält. Zusätzlich räumt `LlamaServer.start`
einen Server aus einem früheren, abgebrochenen Lauf mit demselben Wunschport ab
(PID-Datei je Wunschport, damit ein zweiter Server mit anderem Wunschport den ersten nicht beendet).

Zustand (Log, PID-Dateien) liegt in labs/artifacts/lab7/ oder im Ordner aus AI320_LAB7_STATE.

Umgebungsvariablen nur für Bauten auf einem ausgelasteten Rechner (Standard = Kursbetrieb):
    AI320_LAB7_THREADS     Rechen-Threads von llama-server (-t), Standard: Vorgabe von llama.cpp
    AI320_LAB7_RUNDEN      Anfragen je Client und Stufe im Lasttest, Standard 2
    AI320_LAB7_WDH         Wiederholungen bei Einzelmessungen, Standard 4
    AI320_LAB7_MAX_TOKENS  Antwortlänge in Token, Standard 64
"""
from __future__ import annotations

import atexit
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from common.paths import ARTIFACTS_DIR, LAB_DIR, MODELS_DIR, SEED

GGUF_PATH = MODELS_DIR / "qwen2.5-0.5b-instruct" / "qwen2.5-0.5b-instruct-q4_k_m.gguf"
REFERENCE = LAB_DIR / "reference" / "lab7_referenz.json"
STATE_DIR = Path(os.environ.get("AI320_LAB7_STATE", ARTIFACTS_DIR / "lab7"))

ROUNDS = int(os.environ.get("AI320_LAB7_RUNDEN", 2))
REPEATS = int(os.environ.get("AI320_LAB7_WDH", 4))
MAX_TOKENS = int(os.environ.get("AI320_LAB7_MAX_TOKENS", 64))


# ---------------------------------------------------------------- Server-Prozess

def llama_server_binary() -> str:
    """Pfad zu llama-server: Umgebungsvariable LLAMA_SERVER, sonst PATH, sonst Homebrew."""
    candidate_ids = [os.environ.get("LLAMA_SERVER"), shutil.which("llama-server"),
                  "/opt/homebrew/bin/llama-server", "/usr/local/bin/llama-server"]
    for k in candidate_ids:
        if k and Path(k).exists():
            return k
    raise FileNotFoundError("llama-server nicht gefunden: llama.cpp installieren oder LLAMA_SERVER setzen")


def server_available() -> bool:
    """True, wenn llama-server und die GGUF-Datei da sind."""
    try:
        llama_server_binary()
    except FileNotFoundError:
        return False
    return GGUF_PATH.exists()


def _port_free(port: int) -> bool:
    """True, wenn auf 127.0.0.1 und ::1 niemand lauscht und sich 127.0.0.1:port binden lässt.

    Der Verbindungsversuch erkennt auch Dienste, die auf allen Adressen lauschen (0.0.0.0 oder ::,
    etwa Docker Desktop); ein bind-Test mit SO_REUSEADDR übersieht sie auf macOS.
    """
    for family, address in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
        try:
            with socket.socket(family, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((address, port)) == 0:
                    return False
        except OSError:
            pass  # z. B. kein IPv6 auf diesem Rechner
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def _pidfile(preferred_port: int) -> Path:
    return STATE_DIR / f"llama_server_{preferred_port}.pid"


def _stop_old_server(pidfile: Path) -> None:
    """Beendet einen Server aus einem früheren Lauf, falls die PID-Datei noch auf ihn zeigt."""
    if not pidfile.exists():
        return
    try:
        pid = int(pidfile.read_text().split()[0])
        command = subprocess.run(["ps", "-o", "command=", "-p", str(pid)], capture_output=True, text=True).stdout
        if "lab7_serving" not in command:
            raise ProcessLookupError
        os.killpg(pid, signal.SIGTERM)
        for _ in range(40):
            os.killpg(pid, 0)
            time.sleep(0.25)
        os.killpg(pid, signal.SIGKILL)
    except (ValueError, ProcessLookupError, PermissionError, OSError):
        pass
    pidfile.unlink(missing_ok=True)


class LlamaServer:
    """llama-server auf der CPU (-ngl 0, ohne GPU-Gerät) mit `parallel` Slots und /metrics.

    Benutzung:
        srv = LlamaServer(parallel=8).start()   # wartet, bis /health 200 meldet
        ...
        srv.stop()
    oder als Kontextmanager (`with LlamaServer(parallel=2) as srv: ...`).
    """

    def __init__(self, parallel: int = 8, ctx_per_slot: int = 2048, port: int = 8080,
                 model: Path = GGUF_PATH, extra: list[str] | None = None):
        self.parallel, self.ctx_per_slot, self.preferred_port = parallel, ctx_per_slot, port
        self.model, self.extra = Path(model), list(extra or [])
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def cmd(self) -> list[str]:
        return [llama_server_binary(), "-m", str(self.model),
                "-ngl", "0", "--device", "none",
                "--parallel", str(self.parallel), "-c", str(self.parallel * self.ctx_per_slot),
                "--metrics", "--host", "127.0.0.1", "--port", str(self.port), *self._threads(), *self.extra]

    @staticmethod
    def _threads() -> list[str]:
        # Nur für Bauten auf einem ausgelasteten Rechner: AI320_LAB7_THREADS=2 verhindert, dass die
        # Rechen-Threads von llama-server gegen andere Prozesse um Kerne kämpfen. Sonst Default.
        t = os.environ.get("AI320_LAB7_THREADS")
        return ["-t", t] if t else []

    def start(self, timeout: float = 120) -> "LlamaServer":
        import httpx

        if not self.model.exists():
            raise FileNotFoundError(f"{self.model} fehlt: einmal `python prefetch.py` ausführen")
        self.pidfile = _pidfile(self.preferred_port)
        _stop_old_server(self.pidfile)
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.port = next((p for p in range(self.preferred_port, self.preferred_port + 20) if _port_free(p)), None)
        if self.port is None:
            raise RuntimeError(f"Ports {self.preferred_port} bis {self.preferred_port + 19} sind belegt")
        if self.port != self.preferred_port:
            print(f"Port {self.preferred_port} ist von einem anderen Dienst belegt, der Server nutzt Port {self.port}.")
        self.log = STATE_DIR / f"llama_server_{self.port}.log"
        with open(self.log, "w") as logf:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "common.lab7_serving", "--watch", str(os.getpid()), "--", *self.cmd()],
                cwd=LAB_DIR, stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
        self.pidfile.write_text(f"{self.proc.pid} {self.port}\n")
        atexit.register(self.stop)
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                tail = "".join(self.log.read_text(errors="replace").splitlines(keepends=True)[-15:])
                self.proc = None
                raise RuntimeError(f"llama-server hat sich beendet:\n{tail}")
            try:
                if httpx.get(self.url + "/health", timeout=2).status_code == 200:
                    return self
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        self.stop()
        raise TimeoutError(f"llama-server meldet nach {timeout:.0f} s nicht bereit, Log: {self.log}")

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=5)
        except ProcessLookupError:
            pass
        self.proc = None
        self.pidfile.unlink(missing_ok=True)

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def _watch(parent: int, cmd: list[str]) -> None:
    """Wächter: startet cmd und beendet es, wenn der Elternprozess weg ist oder SIGTERM kommt."""
    child = subprocess.Popen(cmd)

    def shutdown(*_):
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    while child.poll() is None:
        try:
            os.kill(parent, 0)
        except ProcessLookupError:
            shutdown()
        time.sleep(0.5)
    sys.exit(child.returncode)


# ---------------------------------------------------------------- Prompts

def prompts(n: int, n_chars: int = 600, seed: int = SEED) -> list[str]:
    """n feste Prompts: je ein Paragraph des Gesetzeskorpus, auf `n_chars` Zeichen gekürzt."""
    from common import corpus

    df = corpus.load()
    df = df[df.text.str.len() >= n_chars].sample(n=n, random_state=seed)
    return [f"Fassen Sie {r.enbez} {r.gesetz} in zwei Sätzen zusammen.\n\n{r.text[:n_chars]}"
            for r in df.itertuples()]


# ---------------------------------------------------------------- Messung

def request(client, prompt: str, max_tokens: int = MAX_TOKENS, model: str = "local", llama_options: bool = True) -> dict:
    """Eine gestreamte Anfrage; alle Zeiten am Client in Sekunden ab dem Absenden.

    ttft: erster Chunk mit Text; tpot: (e2e - ttft) / (n_out - 1); itl: Abstände der Text-Chunks.
    llama_options (nur llama-server): ignore_eos hält die Antwortlänge auf max_tokens fest
    (wie `vllm bench serve --ignore-eos`), cache_prompt=False rechnet jeden Prompt neu,
    auch wenn er schon einmal kam.
    """
    extra = {"ignore_eos": True, "cache_prompt": False} if llama_options else {}
    t0 = time.perf_counter()
    stream = client.chat.completions.create(
        model=model, max_tokens=max_tokens, stream=True, temperature=0, seed=SEED,
        stream_options={"include_usage": True}, extra_body=extra,
        messages=[{"role": "user", "content": prompt}],
    )
    times, last_chunk = [], None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            times.append(time.perf_counter() - t0)
        last_chunk = chunk
    e2e = time.perf_counter() - t0
    usage = getattr(last_chunk, "usage", None)
    n_out = usage.completion_tokens if usage else len(times)
    timings = (getattr(last_chunk, "model_extra", None) or {}).get("timings") or {}
    itl = [b - a for a, b in zip(times, times[1:])]
    return {
        "ttft": times[0] if times else float("nan"),
        "tpot": (e2e - times[0]) / (n_out - 1) if times and n_out > 1 else float("nan"),
        "e2e": e2e,
        "n_out": n_out,
        "n_prompt": usage.prompt_tokens if usage else None,
        "itl_max": max(itl) if itl else float("nan"),
        "server_prompt_s": timings["prompt_ms"] / 1000 if "prompt_ms" in timings else float("nan"),
        "token_t": times,
    }


def load_test(client, prompts: list[str], concurrency: int, rounds: int = ROUNDS, max_tokens: int = MAX_TOKENS,
             offset: float = 0.0, **kw) -> list[dict]:
    """Closed Loop: `concurrency` Clients, jeder schickt `rounds` Anfragen nacheinander.

    Client i startet i * offset / concurrency Sekunden später, damit nicht alle im
    Gleichschritt ihren Prompt schicken (wie Nutzer, die nicht gleichzeitig klicken).
    Liefert eine Zeile je Anfrage; t_send und t_ende zählen ab dem Start des Tests.
    """
    jobs = [prompts[k % len(prompts)] for k in range(concurrency * rounds)]
    t_start = time.perf_counter()
    rows, lock = [], threading.Lock()

    def client_loop(i: int) -> None:
        time.sleep(i * offset / concurrency)
        for r, p in enumerate(jobs[i::concurrency]):
            t = time.perf_counter() - t_start
            z = request(client, p, max_tokens=max_tokens, **kw)
            z.update(concurrency=concurrency, client=i, runde=r, t_send=t, t_ende=t + z["e2e"])
            with lock:
                rows.append(z)

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        list(ex.map(client_loop, range(concurrency)))
    return sorted(rows, key=lambda z: z["t_send"])


def throughput(rows) -> float:
    """Output-Token/s, solange alle Clients gleichzeitig arbeiten (ohne Anlauf und Auslauf).

    Fenster: vom letzten ersten Absenden bis zum ersten letzten Ende; gezählt werden die
    Token, deren Zeitstempel in diesem Fenster liegt. Ist das Fenster leer, die ganze Dauer.
    """
    rows = list(rows.to_dict("records") if hasattr(rows, "to_dict") else rows)
    clients = {}
    for z in rows:
        a, b = clients.get(z["client"], (float("inf"), 0.0))
        clients[z["client"]] = (min(a, z["t_send"]), max(b, z["t_ende"]))
    since, until = max(a for a, _ in clients.values()), min(b for _, b in clients.values())
    if until - since < 0.5:
        since, until = min(z["t_send"] for z in rows), max(z["t_ende"] for z in rows)
    n = sum(since <= z["t_send"] + t <= until for z in rows for t in z["token_t"])
    return n / (until - since)


def sweep(client, prompts: list[str], stages=(1, 2, 4, 8), rounds: int = ROUNDS, max_tokens: int = MAX_TOKENS,
              times: bool = True, **kw):
    """Lasttest je Concurrency-Stufe; eine Zeile je Anfrage als DataFrame.

    Die erste Stufe (Concurrency 1) läuft mit viermal so vielen Anfragen und liefert
    die Dauer einer Einzelanfrage; um diese Dauer versetzt starten die Clients der
    höheren Stufen. Jede Stufe beginnt mit denselben Prompts. times=False druckt nur den
    Fortschritt, ohne Dauer je Stufe.
    """
    import pandas as pd

    rows, offset = [], 0.0
    for c in stages:
        t = time.perf_counter()
        z = load_test(client, prompts, c, rounds=4 * rounds if c == 1 else rounds,
                     max_tokens=max_tokens, offset=offset, **kw)
        if c == 1:
            offset = sorted(x["e2e"] for x in z)[len(z) // 2]
        rows += z
        duration = f" in {time.perf_counter() - t:5.1f} s" if times else ""
        print(f"Concurrency {c:2d}: {len(z):2d} Anfragen fertig{duration}")
    return pd.DataFrame(rows)


def table(df):
    """Kennzahlen je Concurrency-Stufe: Mediane (p50) und p95 über die Anfragen der Stufe."""
    import pandas as pd

    return pd.DataFrame({c: {
        "Anfragen": len(g),
        "TTFT p50 [s]": g.ttft.median(), "TTFT p95 [s]": g.ttft.quantile(0.95),
        "TPOT p50 [ms]": 1000 * g.tpot.median(), "TPOT p95 [ms]": 1000 * g.tpot.quantile(0.95),
        "längste Pause p50 [ms]": 1000 * g.itl_max.median(),
        "Durchsatz [Token/s]": throughput(g) if "token_t" in g else g["durchsatz_stufe"].iloc[0],
    } for c, g in df.groupby("concurrency")}).T.rename_axis("concurrency")


def server_metrics(url: str) -> dict:
    """Prometheus-Text von /metrics als {name: wert} (nur Proben ohne Labels)."""
    import httpx
    from prometheus_client.parser import text_string_to_metric_families

    text = httpx.get(url + "/metrics", timeout=5).text
    return {s.name: s.value for family in text_string_to_metric_families(text) for s in family.samples}


def machine_info() -> dict:
    """Kurzbeschreibung des Rechners für die Gegenüberstellung mit der Referenz."""
    try:
        version = subprocess.run([llama_server_binary(), "--version"], capture_output=True, text=True,
                                 timeout=20).stderr.strip().splitlines()[0]
    except Exception:  # noqa: BLE001
        version = "unbekannt"
    processor = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            processor = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                       text=True, timeout=5).stdout.strip() or processor
        elif Path("/proc/cpuinfo").exists():
            line = next(z for z in Path("/proc/cpuinfo").read_text().splitlines() if z.startswith("model name"))
            processor = line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return {"plattform": platform.platform(), "prozessor": processor,
            "kerne": os.cpu_count(), "llama_server": version}


def reference_run(ref: dict):
    """Anfragen der Referenzmessung als DataFrame wie `sweep`, für den Ersatzweg ohne llama-server.

    Die Referenz speichert keine Zeitstempel der Token; den Durchsatz je Stufe übernimmt
    `table` deshalb aus der Spalte `durchsatz_stufe` (Wert der Referenz).
    """
    import pandas as pd

    df = pd.DataFrame(ref["messreihe"]["anfragen"])
    stage = {s["concurrency"]: s["Durchsatz [Token/s]"] for s in ref["messreihe"]["stufen"]}
    return df.assign(durchsatz_stufe=df.concurrency.map(stage))


def reference() -> dict | None:
    """Mitgelieferte Referenzmessung des Trainers (labs/reference/lab7_referenz.json), sonst None."""
    if not REFERENCE.exists():
        print(f"{REFERENCE.name} fehlt: Der Vergleich mit der Referenzmessung entfällt.")
        return None
    return json.loads(REFERENCE.read_text(encoding="utf-8"))


if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[1] == "--watch":
    _watch(int(sys.argv[2]), sys.argv[sys.argv.index("--") + 1:])
