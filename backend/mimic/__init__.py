"""MIMIC Backend Integration — streams MIMIC-IV data to FastAPI."""
from backend.mimic.loader import (
    MIMICLoader,
    get_mimic_loader,
    load_mimic_into_backend,
)

__all__ = [
    "MIMICLoader",
    "get_mimic_loader",
    "load_mimic_into_backend",
]
