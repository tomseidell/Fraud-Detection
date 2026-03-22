from pandas import DataFrame

SPLIT_RATIO = 0.8

def split_data(df: DataFrame):
    y = df["isFraud"]
    X = df.drop(columns=["isFraud"])

    split_idx = int(len(X) * SPLIT_RATIO)

    X_train = X.iloc[:split_idx]
    X_test  = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test  = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test