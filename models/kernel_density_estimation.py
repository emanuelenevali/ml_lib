import numpy as np

class KernelDensityEstimation:
    def __init__(self, kernel='gaussian', bandwidth=1.0):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.X_train = None

    def fit(self, X):
        self.X_train = X

    def _gaussian_kernel(self, u):
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * u**2)

    def _epanechnikov_kernel(self, u):
        return 0.75 * (1 - u**2) * (np.abs(u) <= 1)

    def predict(self, X):
        n_samples = len(self.X_train)
        n_points = len(X)
        
        if self.X_train is None:
            raise ValueError("Fit the model before predicting.")

        densities = np.zeros(n_points)

        for i, x_point in enumerate(X):
            kernel_sum = 0
            for x_train_sample in self.X_train:
                u = (x_point - x_train_sample) / self.bandwidth
                if self.kernel == 'gaussian':
                    kernel_sum += self._gaussian_kernel(u)
                elif self.kernel == 'epanechnikov':
                    kernel_sum += self._epanechnikov_kernel(u)
                else:
                    raise ValueError(f"Unknown kernel: {self.kernel}. Supported kernels are 'gaussian' and 'epanechnikov'.")
            densities[i] = kernel_sum / (n_samples * self.bandwidth)
            
        return densities
