from src.data.loader import Loader
from src.data.splitter import split_data

loader = Loader().load()
X_train, _, _, _ = split_data(loader.df)
missing = X_train.isnull().mean()



BASE_UID_COLS = ["card1"]
# This object defines the different options for creating a unique user id based
# on certain combinations of columns from our dataset
UID_VARIANTS = {
    "card1_only": BASE_UID_COLS,
    "card1_addr1": ["card1", "addr1"],
    "card1_addr1_d1": ["card1", "addr1", "D1"],
    "card1_card6_card4_email": ["card1", "card6", "card4", "P_emaildomain"]
}

# This array defines the columns where OHE should be applied
# (less than 10 unique values)
OHE_COLS = [
    "ProductCD", "card4", "card6", "DeviceType",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28",
    "id_29", "id_34", "id_35", "id_36", "id_37", "id_38"
]

# This array defines the columns where Frequency Encoding should be applied
# (more than 10 unique values)
FREQUENCY_COLS = [
    "P_emaildomain", "R_emaildomain", "id_30", "id_31",
    "id_33", "DeviceInfo"
]

# this array saves all the columns that should not be dropped 
PROTECTED_COLS = OHE_COLS + FREQUENCY_COLS + ["R_emaildomain", "P_emaildomain", "card1", "card6", "card4", "addr1", "D1"]


BASE_DROP_COLS = ["TransactionID"]
# This object saves the different options for dropping columns with missing values
DROP_VARIANTS = {
    "none":    BASE_DROP_COLS,
    "drop_99": BASE_DROP_COLS + [c for c in missing[missing > 0.99].index if c not in PROTECTED_COLS],
    "drop_90": BASE_DROP_COLS + [c for c in missing[missing > 0.90].index if c not in PROTECTED_COLS],
    "drop_85": BASE_DROP_COLS + [c for c in missing[missing > 0.85].index if c not in PROTECTED_COLS],
    "drop_75": BASE_DROP_COLS + [c for c in missing[missing > 0.75].index if c not in PROTECTED_COLS],
}