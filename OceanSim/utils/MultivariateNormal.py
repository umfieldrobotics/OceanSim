"""
Covariance Initialization: 
The init_sigma and init_cov methods handle both scalar and matrix inputs.
Random Sampling: 
The sampling uses NumPy's random.default_rng() for reproducibility and performance.
Cholesky Decomposition: 
Implemented directly in Python with an equivalent structure to the C++ version.
Data Types: 
Unreal-specific types like TArray are replaced with Python lists or NumPy arrays.

"""

import numpy as np

class MultivariateNormal:
    def __init__(self, N):
        assert N > 0, "MVN: N must be > 0"
        self.N = N
        self.sqrt_cov = np.zeros((N, N))
        self.uncertain = False
        self.gen = np.random.default_rng()

    def init_sigma(self, sigma):
        """Initialize diagonal covariance using a single float or an array."""
        if isinstance(sigma, (float, int)):
            np.fill_diagonal(self.sqrt_cov, sigma)
        elif isinstance(sigma, (list, np.ndarray)):
            assert len(sigma) == self.N, f"Sigma has size {len(sigma)} and should be {self.N}"
            np.fill_diagonal(self.sqrt_cov, sigma)
        self.uncertain = True

    def init_cov(self, cov):
        """Initialize covariance."""
        if isinstance(cov, (float, int)):
            np.fill_diagonal(self.sqrt_cov, np.sqrt(cov))
        elif isinstance(cov, (list, np.ndarray)):
            cov = np.array(cov)
            if cov.ndim == 1:  # Diagonal covariance
                np.fill_diagonal(self.sqrt_cov, np.sqrt(cov))
            elif cov.ndim == 2:  # Full covariance matrix
                assert cov.shape == (self.N, self.N), f"Covariance matrix size {cov.shape} should be ({self.N}, {self.N})"
                self.sqrt_cov = cov.copy()
                success = self.cholesky(self.sqrt_cov)
                if not success:
                    print("Warning: MVN encountered a non-positive definite covariance")
            else:
                raise ValueError("Invalid covariance input")
        self.uncertain = True

    def sample_array(self):
        """Generate a sample from the multivariate normal distribution."""
        if not self.uncertain:
            return np.zeros(self.N)

        # Sample from N(0,1)
        sam = self.gen.standard_normal(self.N)
        # Shift by the covariance
        result = self.sqrt_cov @ sam
        return result

    def sample_list(self):
        return self.sample_array().tolist()

    def sample_vector(self):
        assert self.N == 3, f"Can't use MVN size {self.N} with 3D vector samples"
        sample = self.sample_array()
        return sample.tolist()

    def sample_float(self):
        assert self.N == 1, f"Can't use MVN size {self.N} with float samples"
        return self.sample_array()[0]

    def sample_rayleigh(self):
        assert self.N == 1, f"Can't use MVN size {self.N} with Rayleigh Noise"
        x = self.sample_float()
        y = self.sample_float()
        return np.sqrt(x**2 + y**2)

    @staticmethod
    def cholesky(A):
        """Compute the Cholesky decomposition in place."""
        N = A.shape[0]
        for i in range(N):
            for j in range(i, N):
                sum_val = A[i, j]
                for k in range(i):
                    sum_val -= A[i, k] * A[j, k]
                if i == j:
                    if sum_val <= 0:
                        return False  # Not positive definite
                    A[i, j] = np.sqrt(sum_val)
                else:
                    A[j, i] = sum_val / A[i, i]

        # Zero out upper triangular part
        for i in range(N):
            for j in range(i + 1, N):
                A[i, j] = 0
        return True

    def get_sqrt_cov(self):
        return self.sqrt_cov

    def is_uncertain(self):
        return self.uncertain