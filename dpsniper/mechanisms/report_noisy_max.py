import numpy as np
from dpsniper.mechanisms.abstract import Mechanism


class ReportNoisyMax1(Mechanism):
    """
    Alg. 5 from:
        Zeyu Ding, YuxinWang, GuanhongWang, Danfeng Zhang, and Daniel Kifer. 2018.
        Detecting Violations of Differential Privacy. CCS 2018.
    """

    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def m(self, a, n_samples: int = 1):
        v = np.atleast_2d(a)

        # each row in m is one sample
        m = v + np.random.laplace(scale=2/self.eps, size=(n_samples, a.shape[0]))
        return np.argmax(m, axis=1)


class ReportNoisyMax2(Mechanism):
    """
    Alg. 6 from:
        Zeyu Ding, YuxinWang, GuanhongWang, Danfeng Zhang, and Daniel Kifer. 2018.
        Detecting Violations of Differential Privacy. CCS 2018.
    """

    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def m(self, a, n_samples: int = 1):
        v = np.atleast_2d(a)

        # each row in m is one sample
        m = v + np.random.exponential(scale=2/self.eps, size=(n_samples, a.shape[0]))
        return np.argmax(m, axis=1)


class ReportNoisyMax3(Mechanism):
    """
    Alg. 7 from:
        Zeyu Ding, YuxinWang, GuanhongWang, Danfeng Zhang, and Daniel Kifer. 2018.
        Detecting Violations of Differential Privacy. CCS 2018.
    """

    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def m(self, a, n_samples: int = 1):
        v = np.atleast_2d(a)

        # each row in m is one sample
        m = v + np.random.laplace(scale=2/self.eps, size=(n_samples, a.shape[0]))
        return np.amax(m, axis=1)


class ReportNoisyMax4(Mechanism):
    """
    Alg. 8 from:
        Zeyu Ding, YuxinWang, GuanhongWang, Danfeng Zhang, and Daniel Kifer. 2018.
        Detecting Violations of Differential Privacy. CCS 2018.
    """

    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def m(self, a, n_samples: int = 1):
        v = np.atleast_2d(a)

        # each row in m is one sample
        m = v + np.random.exponential(scale=2/self.eps, size=(n_samples, a.shape[0]))
        return np.amax(m, axis=1)
