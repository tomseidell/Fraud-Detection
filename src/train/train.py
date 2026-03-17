from xgboost import XGBClassifier
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.data import Data
import json

"""
In this file we will train the final model with optimized parameters we saved in best_params.json.
After final training, we save the model inside the models folder by calling the model.save_model mehtod.
"""


# load best parameter according to HyperParameter tuning
with open(Path(__file__).resolve().parents[2] / "src" / "models" / "best_params.json", "r") as f:
    best_params = json.load(f)


# load perpared data
data = Data()
X_train, X_val, X_test, y_train, y_val, y_test= data.prepare_data()

# add params from above and fit the train data
model = XGBClassifier(**best_params).fit(X_train, y_train)

# save trained model in models folder
model_path = Path(__file__).resolve().parents[2] / "src" / "models" / "xgboost_final.json"
model.save_model(str(model_path))
