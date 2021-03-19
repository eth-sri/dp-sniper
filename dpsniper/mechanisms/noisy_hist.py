import numpy as np
from dpsniper.mechanisms.abstract import Mechanism


class NoisyHist1(Mechanism):
    """
    Alg. 9 from:
        Zeyu Ding, YuxinWang, GuanhongWang, Danfeng Zhang, and Daniel Kifer. 2018.
        Detecting Violations of Differential Privacy. CCS 2018.
    """

    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def m(self, a, n_samples: int = 1):
        l = a.shape[0]
        v = np.atleast_2d(a)

        # each row in m is one sample
        m = v + np.random.laplace(scale=1/self.eps, size=(n_samples, l))
        return m


class NoisyHist2(Mechanism):
    """
    Alg. 10 from:
        Zeyu Ding, YuxinWang, GuanhongWang, Danfeng Zhang, and Daniel Kifer. 2018.
        Detecting Violations of Differential Privacy. CCS 2018.
    """

    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def m(self, a, n_samples: int = 1):
        l = a.shape[0]
        v = np.atleast_2d(a)

        # each row in m is one sample
        m = v + np.random.laplace(scale=self.eps, size=(n_samples, l))
        return m
