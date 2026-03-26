import json, joblib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3] 
sys.path.insert(0, str(ROOT))

from src.data.loader import Loader
from src.data.splitter import split_data
from src.pipeline.pipeline import build_pipeline
from src.models.xgb.config import DROP_VARIANTS

PARAMS_PATH = Path(__file__).resolve().parent / "best_params.json"
PIPELINE_PARAMS = Path(__file__).resolve().parent / "best_pipeline_params.json"
MODEL_PATH  = Path(__file__).resolve().parent / "pipeline.pkl"

def train():
    with open(PARAMS_PATH, "r") as f:
        params = json.load(f) 

    with open(PIPELINE_PARAMS, "r") as f:
        pipeline_params = json.load(f)

    loader = Loader().load()
    X_train, X_test, y_train, y_test = split_data(loader.df)

    pipeline = build_pipeline(params, drop_cols=DROP_VARIANTS[pipeline_params["drop_variant"]], number_of_mail_provider=pipeline_params["email_top_n"])
    preprocessor = pipeline[:-1] # exclude last item in pipeline (model)
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_test_transformed  = preprocessor.transform(X_test)

    model = pipeline.named_steps["model"]
    model.fit(
        X_train_transformed, y_train,
        eval_set=[(X_test_transformed, y_test)],
        verbose=False
    )

    joblib.dump(pipeline, MODEL_PATH) 
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()