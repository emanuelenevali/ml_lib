import numpy as np

class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted


class LinearRegressionSGD:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        # In SGD, n_iters is often referred to as the number of epochs
        self.n_iters = n_iters 
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the linear regression model using Stochastic Gradient Descent.
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Outer loop for epochs (passes over the entire dataset)
        for _ in range(self.n_iters):
            # Inner loop for each individual sample
            for i in range(n_samples):
                # Select a single training example
                X_i = X[i, :]
                y_i = y[i]

                # 1. Make a prediction for the single sample
                y_predicted = np.dot(X_i, self.weights) + self.bias
                
                # 2. Compute gradients for the single sample
                # The gradient calculation is no longer averaged
                dw = (y_predicted - y_i) * X_i
                db = y_predicted - y_i
                
                # 3. Update parameters immediately
                self.weights -= self.lr * dw
                self.bias -= self.lr * db

    def predict(self, X):
        """
        Predict values using the trained linear model.
        """
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted