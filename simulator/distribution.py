from abc import ABC, abstractmethod
import numpy as np
import math

class Distribution(ABC):
    @abstractmethod
    def sample(self):
        pass

class UniformDistribution(Distribution):
    def __init__(self, low, high):
        self._low = low
        self._high = high

    def sample(self, seed=0):
        np.random.seed(seed)
        return math.ceil(np.random.uniform(self._low, self._high))

class GaussianDistribution(Distribution):
    def __init__(self, mu, sigma):
        self._mu = mu
        self._sigma = sigma

    def sample(self, seed=0):
        np.random.seed(seed)
        return math.ceil(np.random.normal(self._mu, self._sigma))

if __name__ == "__main__":
    uf = UniformDistribution(1,10)
    print(uf.sample())
    normal = GaussianDistribution(100,10)
    print(normal.sample())