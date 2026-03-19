import optuna
from xgboost import XGBClassifier
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.data.data import Data
from sklearn.model_selection import cross_val_score
import json

"""
In this file we will perform HyperParameter Tuning. In this case we decided to use optuna instead of GridSearch, to save time.
We are optimizing based on the ROC-AUC Metric and want to maximise this. The best parameters found by the algorithm, will 
be saved as a json file in the models folder
"""


# load the data
data = Data()
X_train, X_val, X_test, y_train, y_val, y_test= data.prepare_data()


def objective(trial):
    # define lower and upper border for each parameter
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000), # number of trees in ensemble
        'subsample': trial.suggest_float('subsample', 0.5, 1.0), # train individual tree on x percent of all the data
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), # train individual tree on x percent of all features
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10), # min number of childs for new split
        'gamma': trial.suggest_float('gamma', 0, 5), # regularization size
    }


    model = XGBClassifier(**param) 

    score = cross_val_score(model, X_train, y_train, cv=3, scoring="roc_auc").mean() 
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

# save model params in json fle
with open("../../models/xgb/best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4) 