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
    AmountTransformer
)

OHE_COLS = [
    "ProductCD", "card4", "card6", "DeviceType",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
    "id_29", "id_34", "id_35", "id_36", "id_37", "id_38"
]
BASE_DROP_COLS = ["TransactionID"]

BASE_UID_COLS = ["card1"]


FREQUENCY_COLS = [
    "P_emaildomain", "R_emaildomain", "id_30", "id_31",
    "id_33", "DeviceInfo"
]

def build_pipeline(params:dict, drop_cols:list[str] = BASE_DROP_COLS, uid_cols:list[str] = BASE_UID_COLS) -> Pipeline:
    return Pipeline([
        ("drop", DropColumnsTransformer(columns=drop_cols)),
        ("time", TimeTransformer()),
        ("uid", UidTransformer(columns=uid_cols)),
        #("email", EmailTransformer(number_of_mail_provider=number_of_mail_provider)),
        ("ohe", OHETransformer(columns=OHE_COLS)),
        ("frequency", FrequencyTransformer(columns=FREQUENCY_COLS)),
        ("amount", AmountTransformer()),
        #("ordinal", OrdinalTransformer()),
        ("model", XGBClassifier(**params))
    ])