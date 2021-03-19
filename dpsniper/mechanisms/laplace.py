import numpy as np
from dpsniper.mechanisms.abstract import Mechanism


def _identity(x):
    return x.item(0)


class LaplaceMechanism(Mechanism):

    def __init__(self, fun=_identity, eps: float = 0.1):
        """
        Create a Laplace mechanism.

        Args:
            fun: The function performed before adding noise. The function must accept a 1d array and produce a scalar.
            eps: target epsilon
        """
        self.fun = fun
        self.scale = 1.0 / eps

    def m(self, a, n_samples: int = 1):
        loc = self.fun(a)
        return np.random.laplace(loc=loc, scale=self.scale, size=n_samples)


class LaplaceFixed(Mechanism):

    def __init__(self, fun=_identity, eps: float = 0.1, B: int = 100):
        """
        Create a safe variant of the Laplace mechanism implementing the Snapping mechanism [1] to prevent
        vulnerabilities arising from floating-point arithmetic.

        [1] Mironov, Ilya. "On Significance of the Least Significant Bits for Differential Privacy."
            In Proceedings of the 2012 ACM Conference on Computer and Communications Security - CCS ’12.
            https://doi.org/10.1145/2382196.2382264.

        Args:
            fun: The function performed before adding noise. The function must accept a 1d array and produce a scalar.
            eps: target epsilon
            B: parameter restricting the output of fun to [-B, B]
        """
        self.fun = fun
        self.B = B

        # compute scale such that the resulting mechanism is eps-DP according to Theorem 1 in [1]
        self.scale = (1.0 + (self.B * (2 ** (-49)))) / eps
        assert(self.scale < self.B < ((2.0 ** 46) * self.scale))

        # find smallest power of 2 greater than or equal to scale
        self.Lambda = 2.0**(-20)
        while self.Lambda < self.scale:
            self.Lambda = self.Lambda * 2

    def clamp(self, x):
        if isinstance(x, float) or isinstance(x, int):
            return min(self.B, max(-self.B, x))
        else:
            return np.clip(x, -self.B, self.B)

    def round_to_Lambda(self, x):
        x = x / self.Lambda
        x = np.round(x)
        return x * self.Lambda

    def m(self, a, n_samples: int = 1):
        loc = self.clamp(self.fun(a))
        sign = 1 - (np.random.randint(2, size=n_samples) * 2)
        u = np.random.uniform(size=n_samples)
        assert(u.dtype == np.float64)   # ensure machine epsilon is 2^(-53) as required by Theorem 1 in [1]
        intermediate = loc + sign * self.scale * np.log(u)
        return self.clamp(self.round_to_Lambda(intermediate))
