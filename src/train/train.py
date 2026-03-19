from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.data import Data
import json, joblib

"""
In this file we will train the final model with optimized parameters we saved in best_params.json.
After final training, we save the model inside the models folder by calling the model.save_model mehtod.
"""


# load best parameter according to HyperParameter tuning
with open(Path(__file__).resolve().parents[2] / "src" / "models" / "xgb" / "best_params.json", "r") as f:
    best_params = json.load(f)


# load perpared data
data = Data()
X_train, X_val, X_test, y_train, y_val, y_test= data.prepare_data()

# extract string columns from train data
cat_cols = X_train.select_dtypes(include="str").columns

pipeline = Pipeline([
    ("encoder", ColumnTransformer([
        ("ordinal", OrdinalEncoder( #  create numeric value from string values in cat_cols
            handle_unknown="use_encoded_value",
            unknown_value=-1, # values that the models sees after fit, which havent been seen yet, will be set to -1
            encoded_missing_value=np.nan # keep nan 
        ), cat_cols)
    ], remainder="passthrough")), # keep cols which are already numeric (not in cat_cols)
    ("model", XGBClassifier(**best_params)) # create xgb model with optimised params 
])

pipeline.fit(X_train, y_train) # fit train data

# save trained model in models folder
model_path = Path(__file__).resolve().parents[2] / "src" / "models" / "xgb" / "pipeline.pkl"
joblib.dump(pipeline, model_path)

