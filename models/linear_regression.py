import numpy as np

class LinearRegressionBatchGD:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        # X: Features matrix, shape (n_samples, n_features)
        # y: Target vector, shape (n_samples,)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)  # shape: (n_features,)
        self.bias = 0                        # shape: scalar

        for _ in range(self.n_iters):
            # Calculate predictions: y = X.w + b
            # y_predicted shape: (n_samples,)
            y_predicted = np.dot(X, self.weights) + self.bias

            # Calculate gradients
            # dw (gradient of weights) shape: (n_features,)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            # db (gradient of bias) shape: scalar
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        # X: Features matrix, shape (n_samples, n_features)
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted


class LinearRegressionSGD:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters 
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # X: Features matrix, shape (n_samples, n_features)
        # y: Target vector, shape (n_samples,)
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features) # shape: (n_features,)
        self.bias = 0                       # shape: scalar

        for _ in range(self.n_iters):
            # Iterate over each sample
            for i in range(n_samples):
                # X_i shape: (n_features,)
                # y_i shape: scalar
                X_i = X[i, :]
                y_i = y[i]

                # Calculate prediction for a single sample
                y_predicted = np.dot(X_i, self.weights) + self.bias # scalar

                # Calculate gradients for a single sample
                dw = (y_predicted - y_i) * X_i # shape: (n_features,)
                db = y_predicted - y_i         # shape: scalar
                
                # Update parameters
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict(self, X):
        # X: Features matrix, shape (n_samples, n_features)
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted


class LinearRegressionNormalEq:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        # X: Features matrix, shape (n_samples, n_features)
        # y: Target vector, shape (n_samples,)
        n_samples, n_features = X.shape

        # 1. Create a column vector of ones for the bias term
        ones_col = np.ones((n_samples, 1)) # shape: (n_samples, 1)

        # 2. Add the bias column to the feature matrix X
        # X_b shape: (n_samples, n_features + 1)
        X_b = np.concatenate((ones_col, X), axis=1)

        # 3. Calculate theta using the Normal Equation: θ = (X_b.T * X_b)^(-1) * X_b.T * y
        # X_b.T shape: (n_features + 1, n_samples)
        # (X_b.T @ X_b) shape: (n_features + 1, n_features + 1)
        # (X_b.T @ y) shape: (n_features + 1,)
        # theta_best shape: (n_features + 1,)
        try:
            theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if the matrix is not invertible
            theta_best = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        
        # 4. Split theta into bias and weights
        self.bias = theta_best[0]          # scalar
        self.weights = theta_best[1:]      # shape: (n_features,)
        
    def predict(self, X):
        # X: Features matrix, shape (n_samples, n_features)
        if self.weights is None or self.bias is None:
            raise RuntimeError("Model has not been trained yet. Please call fit() first.")
        
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted