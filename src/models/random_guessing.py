import numpy as np
import random

class RandomGuessing:
    """
    This class represents a Modell, randomly guessing any number between 0-1 which then represent a probability
    of a given transaction being fraud
    """

    def predict_one(self) -> int:
        """
        returns a single random number between 0 and 1.0
        """
        return random.uniform(0, 1.0)
    
    def predict_many(self, X:np.ndarray) -> np.ndarray:
        """
        Returns a 1D array of random probabilities between 0 and 1,
        one for each row in X.
        """
        array = []
        for _ in range(X.shape[0]):
            array.append(self.predict_one())
        
        return np.array(array)



guesser = RandomGuessing()
print(guesser.predict_one())