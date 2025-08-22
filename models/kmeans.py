import numpy as np

class KMeans:
    """
    A Python implementation of both batch and sequential K-Means clustering.

    Attributes:
        k (int): The number of clusters.
        max_iters (int): The maximum number of iterations for the batch algorithm.
        random_state (int): Seed for the random number generator for reproducibility.
        centroids (np.ndarray): The coordinates of the cluster centers.
        cluster_counts (np.ndarray): A count of points assigned to each cluster (for sequential).
    """

    def __init__(self, k=3, max_iters=100, random_state=42):
        """
        Initializes the KMeans instance.

        Args:
            k (int): The number of clusters to form.
            max_iters (int): Maximum iterations for the batch `fit` method.
            random_state (int): Seed for reproducibility of centroid initialization.
        """
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.cluster_counts = np.zeros(k)

    def _euclidean_distance(self, p1, p2):
        """Calculates the Euclidean distance between two points."""
        return np.sqrt(np.sum((p1 - p2)**2))

    # --------------------------------------------------
    # Batch K-Means Implementation
    # --------------------------------------------------

    def fit(self, X):
        """
        Computes k-means clustering using the batch algorithm.

        This method processes the entire dataset at each iteration to update centroids.

        Args:
            X (np.ndarray): The input data of shape (n_samples, n_features).
        """
        n_samples, n_features = X.shape
        
        # 1. Initialize centroids randomly from the data
        rng = np.random.RandomState(self.random_state)
        random_indices = rng.choice(n_samples, self.k, replace=False)
        self.centroids = X[random_indices]

        # Main optimization loop for batch k-means
        for _ in range(self.max_iters):
            # 2. Assign samples to the closest centroid (E-step)
            labels = self._assign_clusters(X)

            # 3. Update centroids based on the mean of assigned samples (M-step)
            new_centroids = np.zeros((self.k, n_features))
            for i in range(self.k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
            
            # 4. Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        # Update cluster counts based on final batch assignment
        final_labels = self._assign_clusters(X)
        for i in range(self.k):
            self.cluster_counts[i] = np.sum(final_labels == i)


    def _assign_clusters(self, X):
        """Helper function to assign each data point in X to the nearest centroid."""
        labels = np.empty(X.shape[0], dtype=int)
        for i, sample in enumerate(X):
            distances = [self._euclidean_distance(sample, centroid) for centroid in self.centroids]
            labels[i] = np.argmin(distances)
        return labels

    # --------------------------------------------------
    # Sequential K-Means Implementation
    # --------------------------------------------------

    def partial_fit(self, x):
        """
        Updates the clusters with a single data point using the sequential algorithm.

        Args:
            x (np.ndarray): A single data sample of shape (n_features,).
        
        Returns:
            int: The cluster index the sample was assigned to.
        """
        # Initialize centroids with the first k unique points if not already done
        if self.centroids is None:
            # Create a temporary list to hold initial centroids
            self._initial_centroids_list = []
            self.centroids = np.zeros((self.k, x.shape[0]))

        if len(getattr(self, '_initial_centroids_list', [])) < self.k:
            # Check for duplicates before adding
            if not any(np.allclose(x, c) for c in self._initial_centroids_list):
                centroid_idx = len(self._initial_centroids_list)
                self.centroids[centroid_idx] = x
                self._initial_centroids_list.append(x)
                self.cluster_counts[centroid_idx] = 1
                return centroid_idx
            else: # If it's a duplicate of an initial point, handle as a normal update
                pass

        # Find the closest centroid
        distances = [self._euclidean_distance(x, centroid) for centroid in self.centroids]
        closest_centroid_idx = np.argmin(distances)
        
        # Update the cluster count and learning rate
        self.cluster_counts[closest_centroid_idx] += 1
        learning_rate = 1 / self.cluster_counts[closest_centroid_idx]
        
        # Update the closest centroid
        # c_new = c_old + eta * (x - c_old)
        self.centroids[closest_centroid_idx] = (
            (1 - learning_rate) * self.centroids[closest_centroid_idx] + learning_rate * x
        )
        
        return closest_centroid_idx

    # --------------------------------------------------
    # Prediction Method
    # --------------------------------------------------
    
    def predict(self, X):
        """
        Assigns each sample in X to the nearest cluster.

        Args:
            X (np.ndarray): New data to predict, shape (n_samples, n_features).

        Returns:
            np.ndarray: An array of cluster labels for each sample.
        """
        if self.centroids is None:
            raise RuntimeError("The model has not been fitted yet. Call 'fit' or 'partial_fit' first.")
        return self._assign_clusters(X)


if __name__ == '__main__':
    # Generate some sample data
    from sklearn.datasets import make_blobs
    import matplotlib.pyplot as plt

    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.80, random_state=0)

    # --- Demo: Batch K-Means ---
    print("--- Running Batch K-Means ---")
    kmeans_batch = KMeans(k=4, random_state=42)
    kmeans_batch.fit(X)
    batch_labels = kmeans_batch.predict(X)
    batch_centroids = kmeans_batch.centroids
    
    print("Batch Centroids:\n", batch_centroids)

    # --- Demo: Sequential K-Means ---
    print("\n--- Running Sequential K-Means ---")
    kmeans_seq = KMeans(k=4)
    # Fit the data one point at a time
    for sample in X:
        kmeans_seq.partial_fit(sample)
    
    seq_labels = kmeans_seq.predict(X)
    seq_centroids = kmeans_seq.centroids
    
    print("Sequential Centroids:\n", seq_centroids)

    # --- Plotting the results ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot Batch Results
    ax1.scatter(X[:, 0], X[:, 1], c=batch_labels, s=50, cmap='viridis', alpha=0.7)
    ax1.scatter(batch_centroids[:, 0], batch_centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    ax1.set_title('Batch K-Means Clustering')
    ax1.set_xlabel('Feature 1')
    ax1.set_ylabel('Feature 2')
    ax1.legend()
    ax1.grid(True)

    # Plot Sequential Results
    ax2.scatter(X[:, 0], X[:, 1], c=seq_labels, s=50, cmap='viridis', alpha=0.7)
    ax2.scatter(seq_centroids[:, 0], seq_centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    ax2.set_title('Sequential K-Means Clustering')
    ax2.set_xlabel('Feature 1')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle('K-Means Implementation Comparison')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()