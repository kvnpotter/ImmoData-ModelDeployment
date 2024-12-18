# Imports

import pickle
import numpy as np
import pandas as pd
from gower import gower_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.utils import resample


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
        self.model = pickle.load(open("./model/best_knn_model_reduced.pkl", "rb"))
        self.pred_price = None
        self.distance_data = None
        self.X_distance = None
        self.y_train = None

    def predict(self) -> float:
        """
        Predict a price of property based on new input data

        : return: float: Predicted price.
        """

        # Make prediction

        self.pred_price = self.model.predict(self.data)[0][0].round(2)
        return self.pred_price

    def confidence_bootstrap(
        self,
        X_dist: np.ndarray,
        y: pd.DataFrame,
        new_data_dist: np.ndarray,
        n_bootstraps: int = 100,
        alpha: float = 0.05,
    ) -> tuple[float, float]:
        """
        Calculate confidence interval of price estimation from KNN regressor using bootstrapping.

        : param X_dist: np.ndarray: Pairwise Gower distance matrix for X_train.
        : param y: pd.DataFrame: Dataframe of y data.
        : param new_data_dist: np.ndarray: Array of distances of new point to X_train points.
        : param n_bootstraps: int, optional: Number of bootstrap resampling to perform. Defaults to 1000.
        : param alpha: float, optional: Confidence level of estimation. Defaults to 0.05.

        : return lower: float: Lower bound of CI.
        : return upper: float: Upper bound of CI.
        """
        self.X_distance = X_dist
        self.y_train = y
        self.distance_data = new_data_dist
        predictions = []
        knn = KNeighborsRegressor(
            n_neighbors=19, metric="precomputed", weights="distance"
        )

        for _ in range(n_bootstraps):
            X_boot_gower, new_distance_array, y_resampled = self.resample_distances()
            knn.fit(X_boot_gower, y_resampled)
            predictions.append(knn.predict(new_distance_array)[0])

        # Calculate confidence interval
        lower = np.percentile(predictions, 100 * alpha / 2).round(2)
        upper = np.percentile(predictions, 100 * (1 - alpha / 2)).round(2)
        return lower, upper

    def resample_distances(self) -> np.ndarray:
        """
        Resample distances from an NxN distance matrix, with replacement, returning a random NxN matrix constructed from previous elements while respecting pairwise distances.

        : return: np.ndarray: Resampled distance and price matrices.
        """
        # Get the size of the matrix
        n = self.X_distance.shape[0]

        # Generate the list of indices and resample with replacement
        indices = np.arange(n)
        resampled_indices = np.random.choice(indices, size=n, replace=True)

        # Construct a new matrix based on the resampled indices
        new_matrix = self.X_distance[np.ix_(resampled_indices, resampled_indices)]

        # Construct an array of distances from new point to resampled training points
        distance_column = self.distance_data.reshape(-1, 1)
        new_distance_array = distance_column[resampled_indices]
        new_distance_array = new_distance_array.reshape(1, -1)

        # Construct an array of response values relative to new indices
        new_y_array = self.y_train.values[resampled_indices]

        return new_matrix, new_distance_array, new_y_array
