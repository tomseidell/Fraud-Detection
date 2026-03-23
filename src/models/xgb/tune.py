import optuna
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from sklearn.metrics import roc_auc_score
from src.pipeline.pipeline import build_pipeline
from src.data.loader import Loader
from src.data.splitter import split_data
from src.models.xgb.config import DROP_VARIANTS
import json

"""
In this file we will perform HyperParameter Tuning. In this case we decided to use optuna instead of GridSearch, to save time.
We are optimizing based on the ROC-AUC Metric and want to maximise this. The best parameters found by the algorithm, will 
be saved as a json file in the models folder
"""
PARAMS_PATH = Path(__file__).resolve().parent / "best_params.json"
PIPELINE_PARAMS_PATH = Path(__file__).resolve().parent / "best_pipeline_params.json"

loader = Loader().load()
X_train, X_test, y_train, y_test = split_data(loader.df)




def objective(trial):
    # define lower and upper border for each parameter
    model_params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), # number of trees in ensemble
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # train individual tree on x percent of all the data
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # train individual tree on x percent of all features
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), # min number of childs for new split
        'gamma': trial.suggest_float('gamma', 0, 7), # regularization size
    }


    drop_variant   = trial.suggest_categorical("drop_variant", list(DROP_VARIANTS.keys()))
    number_of_mail_provider    = trial.suggest_int("email_top_n", 3, 10)


    pipeline = build_pipeline(params=model_params, drop_cols=DROP_VARIANTS[drop_variant], number_of_mail_provider=number_of_mail_provider) 

    pipeline.fit(X_train, y_train)
     
    score = roc_auc_score(y_test, pipeline.predict_proba(X_test)[:, 1])
    """
    cross_val takes our dataset X_train and splits it into subsets of train and test. for each cv iteration, the subsets both differ from
    before. The Score is always relevative to the test subset and not the train subset.
    """

    return score

optuna.logging.set_verbosity(optuna.logging.WARNING)

study = optuna.create_study(direction="maximize") # maximize score value
study.optimize(objective, n_trials=50) # n_trials = number of trials -- trial amount of new parameters

print(f"Best ROC-AUC: {study.best_value:.4f}")
print(f"Best Params: {study.best_params}")


best_trial = study.best_trial

model_params = {
    k: v for k, v in best_trial.params.items()
    if k not in ("drop_variant", "email_top_n")
}
print("model_params: ", model_params)

pipeline_params = {
    "drop_variant": best_trial.params["drop_variant"],
    "email_top_n":  best_trial.params["email_top_n"],
}

print("pipeline_params: ",pipeline_params)

with open(PARAMS_PATH, "w") as f:
    json.dump(model_params, f, indent=4)

with open(PIPELINE_PARAMS_PATH, "w") as f:
    json.dump(pipeline_params, f, indent=4)