import json
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from src.pipeline.transformers import (
    DropColumnsTransformer,
    EmailTransformer,
    OHETransformer,
    TimeTransformer,
    OrdinalTransformer,
)

OHE_COLS = ["ProductCD", "card4", "card6", "DeviceType"]
DROP_COLS = ["TransactionID"]



def build_pipeline(params:dict) -> Pipeline:
    return Pipeline([
        ("drop", DropColumnsTransformer(columns=DROP_COLS)),
        ("time", TimeTransformer()),
        ("email", EmailTransformer()),
        ("ohe", OHETransformer(columns=OHE_COLS)),
        ("ordinal", OrdinalTransformer()),
        ("model", XGBClassifier(**params))
    ])