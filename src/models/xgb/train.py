import json, joblib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from src.data.loader import Loader
from src.data.splitter import split_data
from src.pipeline.pipeline import build_pipeline
from src.models.xgb.config import DROP_VARIANTS, UID_VARIANTS

PARAMS_PATH = Path(__file__).resolve().parent / "best_params_testing.json"
PIPELINE_PARAMS = Path(__file__).resolve().parent / "best_pipeline_params.json"
MODEL_PATH = Path(__file__).resolve().parent / "pipeline.pkl"


def train():
    with open(PARAMS_PATH, "r") as f:
        params = json.load(f)

    with open(PIPELINE_PARAMS, "r") as f:
        pipeline_params = json.load(f)

    loader = Loader().load()
    X_train, X_test, y_train, y_test = split_data(loader.df)

    pipeline = build_pipeline(
        params,
        drop_cols=DROP_VARIANTS[pipeline_params["drop_variant"]],
        uid_cols=UID_VARIANTS[pipeline_params["uid_variant"]],
        n_components=pipeline_params["pca_n_components"],
    )

    steps = pipeline.steps[:-1]

    Xt_train = X_train.copy()
    for name, step in steps:
        Xt_train = step.fit_transform(Xt_train, y_train)

    Xt_test = X_test.copy()
    for name, step in steps:
        Xt_test = step.transform(Xt_test)

    model = pipeline.named_steps["model"]
    model.fit(
        Xt_train, y_train,
        eval_set=[(Xt_test, y_test)],
        verbose=False,
    )

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
