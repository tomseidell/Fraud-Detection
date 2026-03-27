from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from src.pipeline.transformers import (
    DropColumnsTransformer,
    EmailTransformer,
    OHETransformer,
    TimeTransformer,
    OrdinalTransformer,
    FrequencyTransformer,
    UidTransformer, 
    AmountTransformer,
    PCATransformer
)
from src.models.xgb.config import OHE_COLS, BASE_DROP_COLS, FREQUENCY_COLS, BASE_UID_COLS


def build_pipeline(params:dict, drop_cols:list[str] = BASE_DROP_COLS, uid_cols:list[str] = BASE_UID_COLS, n_components:int = 4) -> Pipeline:
    return Pipeline([
        ("drop", DropColumnsTransformer(columns=drop_cols)),
        ("time", TimeTransformer()),
        ("uid", UidTransformer(columns=uid_cols)),
        #("email", EmailTransformer(number_of_mail_provider=number_of_mail_provider)),
        ("ohe", OHETransformer(columns=OHE_COLS)),
        ("frequency", FrequencyTransformer(columns=FREQUENCY_COLS)),
        ("amount", AmountTransformer()),
        #("ordinal", OrdinalTransformer()),
        ("pca", PCATransformer(n_components=n_components)),
        ("model", XGBClassifier(**params))
    ])