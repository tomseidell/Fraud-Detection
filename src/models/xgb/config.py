from src.data.loader import Loader
from src.data.splitter import split_data
from src.pipeline.pipeline import OHE_COLS

BASE_COLS = ["TransactionID"]
PROTECTED_COLS = OHE_COLS + ["R_emaildomain", "P_emaildomain"]

loader = Loader().load()
X_train, _, _, _ = split_data(loader.df)
missing = X_train.isnull().mean()

DROP_VARIANTS = {
    "none":    BASE_COLS,
    "drop_99": BASE_COLS + [c for c in missing[missing > 0.99].index if c not in PROTECTED_COLS],
    "drop_90": BASE_COLS + [c for c in missing[missing > 0.90].index if c not in PROTECTED_COLS],
    "drop_85": BASE_COLS + [c for c in missing[missing > 0.85].index if c not in PROTECTED_COLS],
    "drop_75": BASE_COLS + [c for c in missing[missing > 0.75].index if c not in PROTECTED_COLS],
}


UID_VARIANTS = {
    "card1_only": ["card1"],
    "card1_addr1": ["card1", "addr1"],
    "card1_addr1_d1": ["card1", "addr1", "D1"],
    "card1_card6_card4_email": ["card1", "card6", "card4", "P_emaildomain"]
}
