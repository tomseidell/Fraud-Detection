import optuna
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from sklearn.metrics import roc_auc_score
from src.pipeline.pipeline import build_pipeline
from src.data.loader import Loader
from src.data.splitter import split_data
from src.models.xgb.config import DROP_VARIANTS
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
    # define lower and upper border for each parameter of xgboost
    model_params = {
        'max_depth': trial.suggest_int('max_depth', 3, 7),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), # number of trees in ensemble
        'subsample': trial.suggest_float('subsample', 0.5, 0.8), # train individual tree on x percent of all the data
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.8), # train individual tree on x percent of all features
        'min_child_weight': trial.suggest_int('min_child_weight', 5, 20), # min number of childs for new split
        'gamma': trial.suggest_float('gamma', 1, 10), # regularization size
        # hardcoded:
        "random_state" : 42, 
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds" : 5,
        "eval_metric" : "auc"
    }

    """ 
    use optuna to improve parameter for feature engineering steps.
    - drop_variant = array of columns the pipeline drops during feature engineering
    - number_of_mail_provider = test all numbers from 3-10 and decide on best amount 
      of emaildomains for cols: R_emaildomain" and "P_emaildomain
    """
    drop_variant   = trial.suggest_categorical("drop_variant", list(DROP_VARIANTS.keys()))
    number_of_mail_provider    = trial.suggest_int("email_top_n", 3, 10)


    # build pipeline dynamically with combinations of model params and pipeline params
    pipeline = build_pipeline(params=model_params, drop_cols=DROP_VARIANTS[drop_variant], number_of_mail_provider=number_of_mail_provider) 

    preprocessor = pipeline[:-1]
    X_train_transformed = preprocessor.fit_transform(X_train, y_train)
    X_test_transformed  = preprocessor.transform(X_test)

    model: XGBClassifier = pipeline[-1]
    model.fit(
        X_train_transformed, y_train,
        eval_set=[(X_test_transformed, y_test)],
        verbose=False
    )
     
    # calculate roc score for specific param combination
    score = roc_auc_score(y_test, model.predict_proba(X_test_transformed)[:, 1])

    return score

# remove optuna logs from console
optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction="maximize") # maximize score value
study.optimize(objective, n_trials=50) # n_trials = number of trials -- trial amount of new parameters

print(f"Best ROC-AUC: {study.best_value:.4f}")


best_trial = study.best_trial

model_params = {
    k: v for k, v in best_trial.params.items()
    if k not in ("drop_variant", "email_top_n")
}
# add hardcoded values because .best_value does not return hardcoded values
model_params["scale_pos_weight"] = scale_pos_weight
model_params["random_state"] = 42

print("model_params: ", model_params)

pipeline_params = {
    "drop_variant": best_trial.params["drop_variant"],
    "email_top_n":  best_trial.params["email_top_n"],
}

print("pipeline_params: ",pipeline_params)

log_experiment(experiment_name=EXPERIMENT_NAME, study=study, additional_comments=ADDITIONAL_COMMENTS)

with open(PARAMS_PATH, "w") as f:
    json.dump(model_params, f, indent=4)

with open(PIPELINE_PARAMS_PATH, "w") as f:
    json.dump(pipeline_params, f, indent=4)