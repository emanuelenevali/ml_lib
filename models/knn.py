import numpy as np
from scipy import stats

class OptimizedClassificationKNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # X shape: (n_samples, n_features)
        # y shape: (n_samples,)
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        # X shape: (m_samples, n_features)

        # 1. Compute distances between ALL test points and ALL training points at once.
        #    We use NumPy's broadcasting feature.
        #    X[:, np.newaxis, :] expands X's shape to: (m_samples, 1, n_features)
        #    self.X_train shape:                          (n_samples, n_features)
        #    The subtraction broadcasts across the dimensions, resulting in a shape of:
        #    (m_samples, n_samples, n_features)
        #    np.linalg.norm then calculates the Euclidean distance along the last axis (features).
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.X_train, axis=2)
        #    The final distances matrix has shape: (m_samples, n_samples)

        # 2. Get the indices of the k nearest neighbors for EACH test sample.
        #    We sort along axis=1 (across the training samples).
        k_indices = np.argsort(distances, axis=1)[:, :self.k]
        #    k_indices shape: (m_samples, k)

        # 3. Get the labels of the k neighbors for EACH test sample.
        #    We use advanced NumPy indexing.
        #    self.y_train shape: (n_samples,)
        k_nearest_labels = self.y_train[k_indices]
        #    k_nearest_labels shape: (m_samples, k)

        # 4. Find the most common label for each row (each test sample).
        #    stats.mode finds the most frequent value along axis=1.
        most_common, _ = stats.mode(k_nearest_labels, axis=1)
        #    most_common is a 2D array of shape (m_samples, 1).
        
        #    We flatten the result to get the final predictions vector.
        #    final shape: (m_samples,)
        return most_common.flatten()