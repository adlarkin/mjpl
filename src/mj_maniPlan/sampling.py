import numpy as np

from scipy.stats import qmc


class HaltonSampler:
    def __init__(self, dim: int, seed: int | None = None):
        self.sampler = qmc.Halton(dim, seed=seed)
        self.seed = seed

    def sample(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
        s = qmc.scale(self.sampler.random(), lower_bounds, upper_bounds)
        return s.flatten()

    def get_seed(self) -> int | None:
        return self.seed
