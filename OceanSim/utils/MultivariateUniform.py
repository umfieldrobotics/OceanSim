import numpy as np
from typing import List, Union
"""
Key Notes:

Assertions and Error Handling:
Used Python's assert and ValueError for input validation.

Random Sampling:
Used numpy.random.default_rng() for random number generation, analogous to std::mt19937 in C++.

Exponential Sampling and PDF:
Replicated the exponential sampling logic using numpy.log and numpy.exp.

Data Types:
Leveraged numpy.ndarray for arrays and ensured the implementation aligns with Python's dynamic typing.
"""

class MultivariateUniform:
    def __init__(self, N: int):
        assert N > 0, "UNIFORM: N must be > 0"
        self.N = N
        self.uncertain = False
        self.max = np.zeros(N)
        self.rng = np.random.default_rng()

    def init_bounds(self, max_: Union[float, List[float]]):
        if isinstance(max_, float):
            self.max.fill(max_)
        elif isinstance(max_, list) and len(max_) == self.N:
            self.max = np.array(max_)
        else:
            raise ValueError(f"Expected a float or list of size {self.N}, got {max_}")
        self.uncertain = np.any(self.max != 0)

    def sample_array(self) -> np.ndarray:
        if self.uncertain:
            return self.rng.uniform(0, 1, self.N) * self.max
        return np.zeros(self.N)

    def sample_list(self) -> List[float]:
        return self.sample_array().tolist()

    def sample_vector(self):
        if self.N != 3:
            raise ValueError(f"Can't use MVN size {self.N} with vector samples")
        sample = self.sample_array()
        return tuple(sample)

    def sample_float(self) -> float:
        if self.N != 1:
            raise ValueError(f"Can't use MVN size {self.N} with float samples")
        return self.sample_array()[0]

    def sample_exponential(self) -> float:
        if self.N != 1:
            raise ValueError(f"Can't use MVN size {self.N} with exponential samples")
        if self.uncertain:
            x = self.rng.uniform(0, 1)
            return -self.max[0] * np.log(x)
        return 0.0

    def exponential_pdf(self, x: float) -> float:
        if self.uncertain:
            return np.exp(-x / self.max[0]) / self.max[0]
        return 1.0

    def exponential_scaled_pdf(self, x: float) -> float:
        if self.uncertain:
            return np.exp(-x / self.max[0])
        return 1.0

    def is_uncertain(self) -> bool:
        return self.uncertain
