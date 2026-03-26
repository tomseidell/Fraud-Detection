import csv
from pathlib import Path

LOG_PATH = Path(__file__).resolve().parent.parent / "experiment_log.csv"

def log_experiment(study, experiment_name: str, additional_comments:str):
    best = study.best_trial

    # write new row with attributes
    row = {
        "experiment": experiment_name,
        "best_roc_auc": round(study.best_value, 4),
        "n_trials": len(study.trials),
        "additional_comments": additional_comments,
        **best.params  
    }

    # write new row in existing file or create new file
    file_exists = LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Experiment '{experiment_name}' config added to csv.")

