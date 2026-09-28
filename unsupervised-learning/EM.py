import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

class EMGMM:
    def __init__(self, n_components, max_iters=100, tol=1e-3):
        self.n_components = n_components
        self.max_iters = max_iters
        self.tol = tol
        self.means = None
        self.covariances = None
        self.pis = None

    def fit(self, data):
        self.means, self.covariances, self.pis = self._initialize_gmm_params(data)
        for _ in range(self.max_iters):
            responsibilities = self._e_step(data)
            new_means, new_covariances, new_pis = self._m_step(data, responsibilities)
            if np.linalg.norm(new_means - self.means) < self.tol:
                break
            self.means, self.covariances, self.pis = new_means, new_covariances, new_pis
        return self

    def predict(self, data):
        responsibilities = self._e_step(data)
        return responsibilities.argmax(axis=1)

    def _initialize_gmm_params(self, data):
        means = self._initialize_centers(data)
        covariances = np.array([np.cov(data, rowvar=False)] * self.n_components)
        pis = np.ones(self.n_components) / self.n_components
        return means, covariances, pis

    def _e_step(self, data):
        likelihoods = np.array([self.pis[k] * multivariate_normal.pdf(data, mean=self.means[k], cov=self.covariances[k]) for k in range(len(self.pis))]).T
        responsibilities = likelihoods / likelihoods.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, data, responsibilities):
        nk = responsibilities.sum(axis=0)
        means = np.dot(responsibilities.T, data) / nk[:, np.newaxis]
        covariances = [np.dot((responsibilities[:, k:(k+1)] * (data - self.means[k])).T, data - self.means[k]) / nk[k] for k in range(len(nk))]
        pis = nk / len(data)
        return means, covariances, pis

    def _initialize_centers(self, data):
        indices = np.random.choice(data.shape[0], size=self.n_components, replace=False)
        return data[indices]


if __name__ == "__main__":
    # Generate some simple test data
    np.random.seed(0)
    cluster1 = np.random.normal(2.0, 0.2, (100, 2))
    cluster2 = np.random.normal(4.0, 0.2, (100, 2))
    data = np.vstack((cluster1, cluster2))

    # Create an EMGMM instance and fit it
    em = EMGMM(n_components=2)
    em.fit(data)

    # Plotting the results
    labels = em.predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=labels, alpha=0.6)
    plt.scatter(em.means[:, 0], em.means[:, 1], c='red', s=100, marker='x')
    plt.title("Test EMGMM")
    plt.show()