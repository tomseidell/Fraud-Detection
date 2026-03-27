import numpy as np
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score


class EvaluationMetrics:
    """
    This class calculates and outputs a summarized report of our given evaluation Metric based on the input 
    y_pred (prediction of the model). y_pred should always be the test set 
    """

    def __init__(self, y:np.ndarray, y_pred: np.ndarray, threshold:float = 0.3):
        if len(y_pred) != len(y):
            raise ValueError(f"y and y_pred must have the same length. Got {len(y)} and {len(y_pred)}")
        
        self.y = y
        self.y_prob = y_pred
        self.y_pred = (y_pred >= threshold).astype(int)
        
        self._calculate_recall()
        self._calculate_precision()
        self._calculate_f1()
        self._calculate_roc_auc()

    
    def _calculate_recall(self):
        self.recall = recall_score(y_true=self.y, y_pred=self.y_pred)

    def _calculate_precision(self):
        self.precision = precision_score(y_true=self.y, y_pred=self.y_pred)
    
    def _calculate_f1(self):
        self.f1 = f1_score(y_true=self.y, y_pred=self.y_pred)

    def _calculate_roc_auc(self):
        self.roc_auc = roc_auc_score(y_true=self.y, y_score=self.y_prob)

    def print_eval_report(self, headline:str):
        print(f"\n{headline}")
        print(f"- Precision: {self.precision}")
        print(f"- Recall: {self.recall}")
        print(f"- F1: {self.f1}")
        print(f"- ROC-AUC: {self.roc_auc}")
