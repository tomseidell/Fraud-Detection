import importlib
import sys
import types


def test_build_pipeline_creates_expected_steps(monkeypatch):
    # mock constant imports from xgb config 
    fake_config = types.ModuleType("src.models.xgb.config")
    fake_config.OHE_COLS = ["ProductCD"]
    fake_config.BASE_DROP_COLS = ["TransactionID"]
    fake_config.FREQUENCY_COLS = ["P_emaildomain"]
    fake_config.BASE_UID_COLS = ["card1"]
    monkeypatch.setitem(sys.modules, "src.models.xgb.config", fake_config)

    # remove cached pipeline import 
    sys.modules.pop("src.pipeline.pipeline", None)
    # import pipeline module (file) uncached
    pipeline_module = importlib.import_module("src.pipeline.pipeline")

    # call function with mocked params
    pipeline = pipeline_module.build_pipeline(
        params={"n_estimators": 10, "max_depth": 3},
        drop_cols=["TransactionID"],
        uid_cols=["card1"],
        n_components=3,
    )

    step_names = [name for name, _ in pipeline.steps]
    # test transformer steps and model in the end
    assert step_names == [
        "drop",
        "time",
        "uid",
        "ohe",
        "frequency",
        "amount",
        "pca",
        "model",
    ]
