import pandas as pd

from src.data.loader import Loader


def test_loader_merges_train_and_test_with_monkeypatched_csv(monkeypatch):

    # mock train_transaction.csv file
    train_trans = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionAmt": [100.0, 200.0],
            "isFraud": [0, 1],
        }
    )
    # mock train_identity.csv file
    train_id = pd.DataFrame({"TransactionID": [1, 2], "id_01": [10.0, 20.0]})
    # mmock test_transaction.csv file
    test_trans = pd.DataFrame(
        {
            "TransactionID": [3, 4],
            "TransactionAmt": [300.0, 400.0],
        }
    )
    # mock test_identity.csv file
    test_id = pd.DataFrame({"TransactionID": [3, 4], "id-01": [30.0, 40.0]})

    # mock .read_csv build in method
    def fake_read_csv(path, *args, **kwargs):
        path_str = str(path)
        if path_str.endswith("train_transaction.csv"):
            return train_trans
        if path_str.endswith("train_identity.csv"):
            return train_id
        if path_str.endswith("test_transaction.csv"):
            return test_trans
        if path_str.endswith("test_identity.csv"):
            return test_id
        raise AssertionError(f"Unexpected path: {path}")

    # use fake_read_csv instead of actual .read_csv method
    monkeypatch.setattr(pd, "read_csv", fake_read_csv)

    # call the class
    loader = Loader().load(include_test=True)

    assert "id_01" in loader.df.columns
    assert "isFraud" in loader.df.columns
    assert len(loader.df) == 2
    # Test columns are normalized from id-01 -> id_01
    assert "id_01" in loader.test_df.columns
    assert len(loader.test_df) == 2
