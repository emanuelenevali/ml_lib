import numpy as np

class SVM:
    """
    A simple Support Vector Machine classifier implemented from scratch
    using gradient descent.
    """

    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        """
        Initializes the SVM classifier.

        Args:
            learning_rate (float): The step size for gradient descent.
            lambda_param (float): The regularization parameter.
            n_iters (int): The number of iterations for training.
        """
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None # weights
        self.b = None # bias

    def fit(self, X, y):
        """
        Trains the SVM model using the training data.

        Args:
            X (np.array): Training data features of shape (n_samples, n_features).
            y (np.array): Target labels of shape (n_samples,). Must be -1 or 1.
        """
        n_samples, n_features = X.shape

        # Ensure labels are -1 or 1
        y_ = np.where(y <= 0, -1, 1)

        # Initialize parameters
        self.w = np.zeros(n_features)
        self.b = 0

        # Gradient descent training loop
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                # Condition for hinge loss: y_i * (w.x_i + b) >= 1
                condition = y_[idx] * (np.dot(x_i, self.w) + self.b) >= 1

                if condition:
                    # If the point is correctly classified and outside the margin,
                    # we only update the weights due to the regularization term.
                    # Gradient of regularization term: 2 * lambda * w
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # If the point is a support vector or misclassified,
                    # we update both weights and bias.
                    # Gradient of regularization: 2 * lambda * w
                    # Gradient of hinge loss: -y_i * x_i for w, -y_i for b
                    dw = 2 * self.lambda_param * self.w - y_[idx] * x_i
                    db = -y_[idx]
                    self.w -= self.lr * dw
                    self.b -= self.lr * db


    def predict(self, X):
        """
        Predicts class labels for the given data.

        Args:
            X (np.array): Data to predict on, shape (n_samples, n_features).

        Returns:
            np.array: Predicted class labels (-1 or 1).
        """
        approx = np.dot(X, self.w) + self.b
        # Return the sign of the output
        return np.sign(approx)