import json, joblib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]  # xgb → models → src → fraud-detection
sys.path.insert(0, str(ROOT))

from src.data.loader import Loader
from src.data.splitter import split_data
from src.pipeline.pipeline import build_pipeline

PARAMS_PATH = Path(__file__).resolve().parent / "best_params.json"
MODEL_PATH  = Path(__file__).resolve().parent / "pipeline.pkl"

def train():
    with open(PARAMS_PATH, "r") as f:
        params = json.load(f)

    loader = Loader().load()
    X_train, X_test, y_train, y_test = split_data(loader.df)

    pipeline = build_pipeline(params)
    pipeline.fit(X_train, y_train)

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()