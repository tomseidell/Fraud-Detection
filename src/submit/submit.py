from pathlib import Path
import pandas as pd
import sys
import joblib
from sklearn.pipeline import Pipeline
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.data import Data

data = Data()

X = data.prepare_data(test_data=True)

model_path = Path(__file__).resolve().parents[2] / "src" / "models" / "xgb" / "pipeline.pkl"
pipeline:Pipeline = joblib.load(filename=model_path)

y_pred = pipeline.predict_proba(X)[:,1]

submission = pd.DataFrame({
    "TransactionID": X["TransactionID"],
    "isFraud": y_pred
})

submission.to_csv(Path(__file__).resolve().parents[2] / "src" / "submit"/ "submission.csv", index=False)