import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, n_clusters, max_iters=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.centers = None

    def fit(self, data):
        self.centers = self._initialize_centers(data)
        for _ in range(self.max_iters):
            self.labels = self._assign_clusters(data)
            new_centers = self._recompute_centers(data, self.labels)
            if np.all(np.linalg.norm(new_centers - self.centers, axis=1) < self.tol):
                break
            self.centers = new_centers
        return self

    def predict(self, data):
        return self._assign_clusters(data)

    def calculate_sse(self, data):
        distances = np.sqrt(((data[:, np.newaxis] - self.centers)**2).sum(axis=2))
        min_distances = np.min(distances, axis=1)
        return np.sum(min_distances**2)

    def _initialize_centers(self, data):
        indices = np.random.choice(data.shape[0], size=self.n_clusters, replace=False)
        return data[indices]

    def _assign_clusters(self, data):
        distances = np.sqrt(((data[:, np.newaxis] - self.centers)**2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def _recompute_centers(self, data, labels):
        new_centers = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centers

if __name__ == "__main__":
    # Generate some simple test data
    np.random.seed(0)
    cluster1 = np.random.normal(1.0, 0.5, (100, 2))
    cluster2 = np.random.normal(5.0, 0.5, (100, 2))
    data = np.vstack((cluster1, cluster2))
    
    # Create a KMeans instance and fit it
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(data)
    
    # Plotting the results
    labels = kmeans.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.6)
    plt.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1], c='red', s=100, marker='x')
    plt.title("Test k-Means")
    plt.show()

    # Print SSE
    print("SSE:", kmeans.calculate_sse(data))
