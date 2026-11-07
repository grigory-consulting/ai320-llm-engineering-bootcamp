"""Feste Pfade relativ zum Lab-Ordner."""
from pathlib import Path

LAB_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = LAB_DIR / "data"
MODELS_DIR = LAB_DIR / "models"
ARTIFACTS_DIR = LAB_DIR / "artifacts"
GESETZE_DIR = DATA_DIR / "gesetze"
NORMEN_PARQUET = GESETZE_DIR / "normen.parquet"

SEED = 1
