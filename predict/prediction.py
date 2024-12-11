#prediction.py file that will contain all the code used to predict a new house's price.

#predict() that will take your preprocessed data as an input and return a price as output.

# Imports

import pickle
import numpy as np
from sklearn.neighbors import KNeighborsRegressor

# Code

class Predictor:
    """
    Class to handle price prediction
    """
    def __init__(self, data: np.ndarray) -> None:
        """
        Create an instance of class Predictor

        : param data: np.ndarray: Array of Gower distances of new data point (for which to predict price) to training data points.
        """
        self.data = data
        self.model = pickle.load(open('./model/best_knn_model_reduced.pkl', 'rb'))
        self.pred_price = None

    def predict(self) -> float:
        """
        Predict a price of property based on new input data

        : return: float: Predicted price.
        """

        # Make prediction

        self.pred_price = self.model.predict(self.data)[0][0].round(2)
        return self.pred_price
