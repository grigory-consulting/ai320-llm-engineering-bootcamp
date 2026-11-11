"""Zugang zum Generator-LLM über einen OpenAI-kompatiblen Endpunkt (Werte aus labs/.env)."""
from __future__ import annotations

import os

from common.paths import LAB_DIR


def _env() -> dict:
    values = {}
    file = LAB_DIR / ".env"
    if file.exists():
        for line in file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip()
    return {k: os.environ.get(k, values.get(k, "")) for k in ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")}


def client():
    """OpenAI-Client und Modellname: `cl, model = client()`."""
    from openai import OpenAI

    e = _env()
    if not e["LLM_API_KEY"]:
        raise RuntimeError("LLM_API_KEY fehlt in labs/.env (Schlüssel vom Trainer)")
    return OpenAI(base_url=e["LLM_BASE_URL"] or None, api_key=e["LLM_API_KEY"]), e["LLM_MODEL"]
