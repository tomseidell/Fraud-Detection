from xgboost import XGBClassifier
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.data import Data

data = Data()

X = data.prepare_data(test_data=True)

model = XGBClassifier()
model.load_model(str(Path(__file__).resolve().parents[2] / "src" / "models" / "xgboost_final.json"))

y_pred = model.predict_proba(X)[:,1]

submission = pd.DataFrame({
    "TransactionID": X["TransactionID"],
    "isFraud": y_pred
})

submission.to_csv(Path(__file__).resolve().parents[2] / "src" / "submit"/ "submission.csv", index=False)