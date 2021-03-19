import numpy as np
import math
from dpsniper.mechanisms.abstract import Mechanism


class TruncatedGeometricMechanism(Mechanism):
    """
    Alg. 4.8 "GeoSample" from:
        V. Balcer and S. Vadhan. 2018.
        Differential Privacy on Finite Computers
        arXiv:1709.05396 [cs.DS]

    According to Thm 4.7 in the above publication, GeoSample(eps, n) is ln(1 + 2^-ceil(ln(2/eps)))-DP
    """

    def __init__(self, eps: float = 0.1, n: int = 5):
        """
        Args:
            eps: epsilon
            n: number of individuals subject to counting query

        Note:
            Choosing eps too small or n too large may lead to integer overflow and failure of the mechanism
        """
        self.n = n
        self.k = math.ceil(math.log(2.0 / eps))
        self.d = int((math.pow(2, self.k + 1) + 1)*math.pow((math.pow(2, self.k) + 1), n - 1))

    def m(self, c, n_samples: int = 1):
        """
        Args:
            c: 1d array of size 1 holding the result of counting query, must be in [0, n]
            n_samples: number of output samples to produce

        Returns:
            1d-array of shape (n_samples,)
        """

        f = self._compute_f(c.item(0))
        u = np.random.randint(1, self.d + 1, size=n_samples)

        # for each entry of u, find smallest z such that f[:, z] >= u
        # (this is fast if self.n is small, unclear how to vectorize)
        z = np.empty(n_samples)
        for idx in reversed(range(0, self.n + 1)):
            z[f[idx] >= u] = idx
        return z

    def _compute_f(self, c: int):
        """
        Computes the function F as defined in Step 2 of Alg. 4.8.

        Returns:
            1d-array f of shape (n+1,), where f[z] = F(z) for z = 0, 1, ..., n
        """

        z_le_c = np.atleast_2d(range(0, c))
        z_geq_c = np.atleast_2d(range(c, self.n))

        # compute the F function
        f = np.empty(self.n + 1, dtype=np.int64)

        # for interval [0, c)
        a = np.power(2, self.k * (c - z_le_c))
        b = np.power((np.power(2, self.k) + 1), self.n - (c - z_le_c))
        f[:c] = np.multiply(a, b)

        # for interval [c, n)
        a = np.power(2, self.k * (z_geq_c - c + 1))
        b = np.power((np.power(2, self.k) + 1), self.n - 1 - (z_geq_c - c))
        f[c:self.n] = self.d - np.multiply(a, b)

        # for n
        f[-1] = self.d

        return f
