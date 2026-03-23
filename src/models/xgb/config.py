from src.data.loader import Loader
from src.data.splitter import split_data
from src.pipeline.pipeline import OHE_COLS

BASE_COLS = ["TransactionID"]
PROTECTED_COLS = OHE_COLS + ["R_emaildomain"]

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
