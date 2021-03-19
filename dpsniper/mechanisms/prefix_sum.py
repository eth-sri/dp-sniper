from dpsniper.mechanisms.abstract import Mechanism

import numpy as np


class PrefixSum(Mechanism):
    """
    PrefixSum from:
        Y. Wang, Z. Ding, G. Wang, D. Kifer, D. Zhang. 2019.
        Proving differential privacy with shadow execution.
        PLDI 2019.
    """
    def __init__(self, eps: float = 0.1):
        self.eps = eps

    def m(self, a, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            nd float array of shape (n_samples, a.shape[0])
        """

        # columns: queries
        # rows: samples
        x = np.atleast_2d(a)

        rho = np.random.laplace(scale=1/self.eps, size=(n_samples, a.shape[0]))
        m = x + rho     # broadcasts x vertically

        res = np.empty(shape=(n_samples, a.shape[0]), dtype=float)
        partial_sum = np.zeros(shape=(n_samples,))

        for col_idx in range(0, a.shape[0]):
            partial_sum = partial_sum + m[:, col_idx]
            res[:, col_idx] = partial_sum

        return res
