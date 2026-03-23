from pathlib import Path
import pandas as pd
import sys
import joblib
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[3]  
sys.path.insert(0, str(ROOT))

from src.data.loader import Loader

MODEL_PATH  = Path(__file__).resolve().parent / "pipeline.pkl"
SUBMISSION_FILE_PATH = Path(__file__).resolve().parent / "submission.csv"


def predict():
    loader = Loader().load(include_test=True)

    X_test = loader.test_df
    transaction_ids = X_test["TransactionID"].copy()

    pipeline:Pipeline = joblib.load(filename=MODEL_PATH)
    y_pred = pipeline.predict_proba(X_test)[:,1]

    submission = pd.DataFrame({
        "TransactionID": transaction_ids,
        "isFraud": y_pred
    })

    submission.to_csv(SUBMISSION_FILE_PATH, index=False)


if __name__ == "__main__":
    predict()