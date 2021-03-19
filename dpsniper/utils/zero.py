import numpy as np

class ZeroNoisePrng:
    """
    A dummy PRNG returning zeros always.
    """
    def laplace(self, *args, size=1, **kwargs):
        return np.zeros(shape=size)

    def exponential(self, *args, size=1, **kwargs):
        return np.zeros(shape=size)

    def binomial(self, *args, size=1, **kwargs):
        return np.zeros(shape=size)
