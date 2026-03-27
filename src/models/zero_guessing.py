import numpy as np

class ZeroGuessing:
    """
    This class represents a Model, randomly guessing any number between 0-1 which then represent a probability
    of a given transaction being fraud
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