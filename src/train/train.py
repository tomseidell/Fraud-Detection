from xgboost import XGBClassifier
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.evaluation.evaluation_metrics import EvaluationMetrics
from src.data.data import Data
import json

with open(Path(__file__).resolve().parents[2] / "src" / "models" / "best_params.json", "r") as f:
    best_params = json.load(f)


data = Data()
X_train, X_val, X_test, y_train, y_val, y_test= data.prepare_data()


model = XGBClassifier(**best_params).fit(X_train, y_train)

y_pred_test = model.predict_proba(X_test)[:, 1]
metrics = EvaluationMetrics(y=y_test, y_pred=y_pred_test, treshold=0.075)
metrics.print_eval_report(headline="XGBClassifier")