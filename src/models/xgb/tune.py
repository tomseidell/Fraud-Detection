import optuna
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from sklearn.metrics import roc_auc_score
from src.pipeline.pipeline import build_pipeline
from src.data.loader import Loader
from src.data.splitter import split_data
from src.models.xgb.config import DROP_VARIANTS, UID_VARIANTS
from xgboost import XGBClassifier
import json
from utils.log_experiment import log_experiment


"""
In this file we will perform HyperParameter Tuning. In this case we decided to use optuna instead of GridSearch, to save time.
We are optimizing based on the ROC-AUC Metric and want to maximise this. The best parameters found by the algorithm, will 
be saved as a json file in the models folder
"""


EXPERIMENT_NAME = "v1"  
ADDITIONAL_COMMENTS = ""


PARAMS_PATH = Path(__file__).resolve().parent / "best_params.json"
PIPELINE_PARAMS_PATH = Path(__file__).resolve().parent / "best_pipeline_params.json"

loader = Loader().load()
X_train, X_test, y_train, y_test = split_data(loader.df)

scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()


def objective(trial):
    model_params = {
        'max_depth': trial.suggest_int('max_depth', 4, 14),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 5000),
        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 50),
        'gamma': trial.suggest_float('gamma', 0, 10),
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": 20,
        "eval_metric": "auc",
        "tree_method": "hist",
    }

    drop_variant = trial.suggest_categorical("drop_variant", list(DROP_VARIANTS.keys()))
    uid_variant = trial.suggest_categorical("uid_variant", list(UID_VARIANTS.keys()))
    n_components_pca = trial.suggest_int("pca_n_components", 3, 10)

    pipeline = build_pipeline(
        params=model_params,
        drop_cols=DROP_VARIANTS[drop_variant],
        uid_cols=UID_VARIANTS[uid_variant],
        n_components=n_components_pca,
    )

    # save each pipeline step and exclude model (last in pipeline)
    steps = pipeline.steps[:-1]  

    # to avoid skl errors we perform every step of the pipeline autonomosly 
    Xt_train = X_train.copy()
    for name, step in steps:
        Xt_train = step.fit_transform(Xt_train, y_train)

    Xt_test = X_test.copy()
    for name, step in steps:
        Xt_test = step.transform(Xt_test)

    model = XGBClassifier(**model_params)
    model.fit(
        Xt_train, y_train,
        eval_set=[(Xt_test, y_test)],
        verbose=False,
    )

    # evalutaion score, roc value based on test set 
    score = roc_auc_score(y_test, model.predict_proba(Xt_test)[:, 1])
    return score


optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print(f"Best ROC-AUC: {study.best_value:.4f}")

best_trial = study.best_trial

model_params = {
    k: v for k, v in best_trial.params.items()
    if k not in ("drop_variant", "uid_variant", "pca_n_components")
}

model_params["scale_pos_weight"] = scale_pos_weight
model_params["random_state"] = 42
model_params["early_stopping_rounds"] = 20
model_params["eval_metric"] = "auc"

print("model_params: ", model_params)

pipeline_params = {
    "drop_variant": best_trial.params["drop_variant"],
    "uid_variant": best_trial.params["uid_variant"],
    "pca_n_components": best_trial.params["pca_n_components"],
}

print("pipeline_params: ", pipeline_params)

# add new row entry to csv (experiment tracker)
log_experiment(experiment_name=EXPERIMENT_NAME, study=study, additional_comments=ADDITIONAL_COMMENTS)

# save model and pipeline params in separate json file

with open(PARAMS_PATH, "w") as f:
    json.dump(model_params, f, indent=4)

with open(PIPELINE_PARAMS_PATH, "w") as f:
    json.dump(pipeline_params, f, indent=4)
