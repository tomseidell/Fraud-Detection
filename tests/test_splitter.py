import pandas as pd

from src.data.splitter import split_data


def test_splitter_uses_time_order_and_80_20_ratio():
    n_rows = 10

    # mock train df
    df = pd.DataFrame(
        {
            "TransactionID": list(range(1, n_rows + 1)),
            "TransactionDT": list(range(100, 100 + n_rows)),
            "feature": list(range(n_rows)),
            "isFraud": [0, 1] * (n_rows // 2),
        }
    )

    # call split function
    X_train, X_test, y_train, y_test = split_data(df)

    # assert SPLIT_RATIO is applied properly 
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

    # order should be preserved (test latest 20%)
    assert X_train["TransactionID"].tolist() == list(range(1, 9))
    assert X_test["TransactionID"].tolist() == [9, 10]
