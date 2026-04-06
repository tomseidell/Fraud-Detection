import numpy as np
import pandas as pd

from src.models.xgb import predict as predict_module


class _DummyPipeline:
    def predict_proba(self, X):
        # Return deterministic fraud probabilities for test assertions.
        probs = np.linspace(0.1, 0.9, len(X))
        return np.column_stack([1.0 - probs, probs])


class _DummyLoader:
    def load(self, include_test=False):
        assert include_test is True
        self.test_df = pd.DataFrame(
            {
                "TransactionID": [1001, 1002, 1003],
                "TransactionAmt": [10.0, 20.0, 30.0],
            }
        )
        return self


def test_predict_writes_submission_csv(monkeypatch, tmp_path):
    #save csv in temporaty folder (auto deletion)
    out_file = tmp_path / "submission.csv"

    # mock loader in predict file
    monkeypatch.setattr(predict_module, "Loader", _DummyLoader)
    # mock joblib.load method (pipeline import)
    monkeypatch.setattr(predict_module.joblib, "load", lambda filename: _DummyPipeline())
    # mock file path for saving final csv
    monkeypatch.setattr(predict_module, "SUBMISSION_FILE_PATH", out_file)

    # call function
    predict_module.predict()

    submission = pd.read_csv(out_file)
    # check for correct output format 
    assert list(submission.columns) == ["TransactionID", "isFraud"]
    assert submission["TransactionID"].tolist() == [1001, 1002, 1003]
    assert submission["isFraud"].between(0.0, 1.0).all()
    # no nan values for isFraud property
    assert submission["isFraud"].isna().sum() == 0
