"""
Project paths to be used in training and deployment
"""

from pathlib import Path

TOP_LEVEL_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = TOP_LEVEL_DIR / "models/rf_pipeline.pkl"
GENRES_PATH = TOP_LEVEL_DIR / "data/genres.csv"
