import os, time, json
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

ENV_FILE = find_dotenv(usecwd=True)          # sucht .env ab dem Notebook-Ordner (labs/.env)
print("Konfiguration aus:", ENV_FILE or "keine .env gefunden, Defaults (OpenAI, gpt-4.1)")
load_dotenv(ENV_FILE, override=True)   # override: CrewAI lädt beim Import schon eine .env aus einem Elternordner
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1")

from pydantic import BaseModel
from crewai import Agent, Task, Crew, Process, LLM, TaskOutput


llm = LLM(model=f"openai/{LLM_MODEL}", base_url=LLM_BASE_URL, api_key=LLM_API_KEY, temperature=0.1)
print("Modell:", llm.model, "| Endpunkt:", LLM_BASE_URL)
print("LLM antwortet:", llm.call("Antworte nur mit OK.")[:40])
