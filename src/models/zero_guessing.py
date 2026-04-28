import numpy as np

class ZeroGuessing:
    """
    Naive baseline that always predicts 0 (no fraud) for every transaction.
    Useful as a lower-bound reference when comparing model performance.
    """

    def predict_one(self) -> int:
        """
        returns a single zero
        """
        return 0
    
    def predict_many(self, X: np.ndarray) -> np.ndarray:
        """
        Returns a 1D array of zeros, one for each row in X.
        """
        return np.zeros(X.shape[0])




guesser = ZeroGuessing()
print(guesser.predict_one())