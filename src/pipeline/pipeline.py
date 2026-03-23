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



def build_pipeline(params:dict, drop_cols:list[str] = DROP_COLS, number_of_mail_provider: int = 5) -> Pipeline:
    return Pipeline([
        ("drop", DropColumnsTransformer(columns=drop_cols)),
        ("time", TimeTransformer()),
        ("email", EmailTransformer(number_of_mail_provider=number_of_mail_provider)),
        ("ohe", OHETransformer(columns=OHE_COLS)),
        ("ordinal", OrdinalTransformer()),
        ("model", XGBClassifier(**params))
    ])