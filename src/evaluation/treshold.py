import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.evaluation.evaluation_metrics import EvaluationMetrics
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from src.data.loader import Loader
from src.data.splitter import split_data

"""
In this file we will perform treshold tuning. With the given problem (Fraud detection of payments), we especially focused
on optimizing the Recall Metric, as we want to detect as much Fraud payments as possible. However, we also want to keep
the precision above 50, to not detect too many normal payments as fraud.
"""

loader = Loader().load()
X_train, X_test, y_train, y_test= split_data(loader.df)

model_path = Path.cwd().parent / "src" / "models" / "xgb" / "pipeline.pkl"
pipeline:Pipeline = joblib.load(filename=model_path)

# call the model on the test set
y_pred_test = pipeline.predict_proba(X_test)[:, 1]


tresholds = [0.0125, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.275, 0.3, 0.325, 0.35, 0.375, 0.4, 0.425, 0.45, 0.475, 0.5]
results = []
for t in tresholds:
    metrics = EvaluationMetrics(y=y_test, y_pred=y_pred_test, treshold=t)
    results.append({
        "threshold": t,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "roc_auc": metrics.roc_auc
    })

# print overview for all tresholds
df_results = pd.DataFrame(results)
print("\n\n ** Overview **")
print(df_results.to_string(index=False))
