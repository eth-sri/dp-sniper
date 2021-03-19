import numpy as np
from dpsniper.mechanisms.abstract import Mechanism


class SparseVectorTechnique1(Mechanism):
    """
    Alg. 1 from:
        M. Lyu, D. Su, and N. Li. 2017.
        Understanding the Sparse Vector Technique for Differential Privacy.
        Proceedings of the VLDB Endowment.
    """

    def __init__(self, eps: float = 0.1, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t

    def m(self, a, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            ndarray of shape (n_samples, a.shape[0]) with entries
                1 = TRUE;
                0 = FALSE;
                -1 = ABORTED;
        """

        # columns: queries
        # rows: samples
        x = np.atleast_2d(a)

        rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))
        nu = np.random.laplace(scale=2*self.c / self.eps2, size=(n_samples, a.shape[0]))

        m = nu + x  # broadcasts x vertically
        cmp = m >= (rho + self.t)   # broadcasts rho horizontally
        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)
        res = cmp.astype(int)

        col_idx = 0
        for column in cmp.T:
            res[aborted, col_idx] = -1
            count = count + column
            aborted = np.logical_or(aborted, count == self.c)
            col_idx = col_idx + 1
        return res


class SparseVectorTechnique2(Mechanism):
    """
    Alg. 2 from:
        M. Lyu, D. Su, and N. Li. 2017.
        Understanding the Sparse Vector Technique for Differential Privacy.
        Proceedings of the VLDB Endowment.
    """

    def __init__(self, eps: float = 0.1, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t

    def m(self, a, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            ndarray of shape (n_samples, a.shape[0]) with entries
                1 = TRUE;
                0 = FALSE;
                -1 = ABORTED;
        """

        # columns: queries
        # rows: samples
        x = np.atleast_2d(a)
        n_queries = a.shape[0]

        rho = np.random.laplace(scale=self.c / self.eps1, size=(n_samples,))
        nu = np.random.laplace(scale=2*self.c / self.eps2, size=(n_samples, n_queries))

        m = nu + x  # broadcasts x vertically

        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)
        res = np.empty(shape=m.shape, dtype=int)
        for col_idx in range(0, n_queries):
            cmp = m[:, col_idx] >= (rho + self.t)
            res[:, col_idx] = cmp.astype(int)
            res[aborted, col_idx] = -1
            count = count + cmp

            # update rho whenever we answer TRUE
            new_rho = np.random.laplace(scale=self.c / self.eps1, size=(n_samples,))
            rho[cmp] = new_rho[cmp]

            aborted = np.logical_or(aborted, count == self.c)
        return res


class SparseVectorTechnique3(Mechanism):
    """
    Alg. 3 from:
        M. Lyu, D. Su, and N. Li. 2017.
        Understanding the Sparse Vector Technique for Differential Privacy.
        Proceedings of the VLDB Endowment.
    """

    def __init__(self, eps: float = 0.1, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t

    def m(self, a, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            float ndarray of shape (n_samples, a.shape[0]) with special entries
                -1000.0 = FALSE;
                -2000.0 = ABORTED;
        """

        # columns: queries
        # rows: samples
        x = np.atleast_2d(a)

        rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))
        nu = np.random.laplace(scale=self.c / self.eps2, size=(n_samples, a.shape[0]))

        m = nu + x  # broadcasts x vertically
        cmp = m >= (rho + self.t)   # broadcasts rho horizontally
        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)

        res = m
        res[np.logical_not(cmp)] = -1000.0

        col_idx = 0
        for column in cmp.T:
            res[aborted, col_idx] = -2000.0
            count = count + column
            aborted = np.logical_or(aborted, count == self.c)
            col_idx = col_idx + 1
        return res


class SparseVectorTechnique4(Mechanism):
    """
    Alg. 4 from:
        M. Lyu, D. Su, and N. Li. 2017.
        Understanding the Sparse Vector Technique for Differential Privacy.
        Proceedings of the VLDB Endowment.
    """

    def __init__(self, eps: float = 0.1, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 4.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t

    def m(self, a, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            ndarray of shape (n_samples, a.shape[0]) with entries
                1 = TRUE;
                0 = FALSE;
                -1 = ABORTED;
        """

        # columns: queries
        # rows: samples
        x = np.atleast_2d(a)

        rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))
        nu = np.random.laplace(scale=1 / self.eps2, size=(n_samples, a.shape[0]))

        m = nu + x  # broadcasts x vertically
        cmp = m >= (rho + self.t)   # broadcasts rho horizontally
        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)
        res = cmp.astype(int)

        col_idx = 0
        for column in cmp.T:
            res[aborted, col_idx] = -1
            count = count + column
            aborted = np.logical_or(aborted, count == self.c)
            col_idx = col_idx + 1
        return res


class SparseVectorTechnique5(Mechanism):
    """
    Alg. 5 from:
        M. Lyu, D. Su, and N. Li. 2017.
        Understanding the Sparse Vector Technique for Differential Privacy.
        Proceedings of the VLDB Endowment.
    """

    def __init__(self, eps: float = 0.1, c: int = 2, t: float = 1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t

    def m(self, a, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            ndarray of shape (n_samples, a.shape[0]) with entries
                1 = TRUE;
                0 = FALSE;
        """

        # columns: queries
        # rows: samples
        x = np.atleast_2d(a)

        rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))

        cmp = x >= (rho + self.t)   # broadcasts rho horizontally, x vertically
        return cmp.astype(int)


class SparseVectorTechnique6(Mechanism):
    """
    Alg. 6 from:
        M. Lyu, D. Su, and N. Li. 2017.
        Understanding the Sparse Vector Technique for Differential Privacy.
        Proceedings of the VLDB Endowment.
    """

    def __init__(self, eps: float = 0.1, c: int = 2, t: float =1.0):
        self.eps1 = eps / 2.0
        self.eps2 = eps - self.eps1
        self.c = c  # maximum number of queries answered with 1
        self.t = t

    def m(self, a, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            ndarray of shape (n_samples, a.shape[0]) with entries
                1 = TRUE;
                0 = FALSE;
        """

        # columns: queries
        # rows: samples
        x = np.atleast_2d(a)

        rho = np.random.laplace(scale=1 / self.eps1, size=(n_samples, 1))
        nu = np.random.laplace(scale=1 / self.eps2, size=(n_samples, a.shape[0]))

        m = nu + x  # broadcasts x vertically
        cmp = m >= (rho + self.t)   # broadcasts rho horizontally
        return cmp.astype(int)


class NumericalSVT(Mechanism):
    """
    Numerical Sparse Vector Technique from:
        Y. Wang, Z. Ding, G. Wang, D. Kifer, D. Zhang. 2019.
        Proving differential privacy with shadow execution.
        PLDI 2019.
    """
    def __init__(self, eps: float = 0.1, c: int = 2, t: float = 1.0):
        self.eps = eps
        self.c = c
        self.t = t

    def m(self, a, n_samples: int = 1):
        """
        Args:
            a: 1d array of query results (sensitivity 1)

        Returns:
            float ndarray of shape (n_samples, a.shape[0]) with special entry
                -1000.0 = ABORTED
        """

        # columns: queries
        # rows: samples
        x = np.atleast_2d(a)

        rho1 = np.random.laplace(scale=3/self.eps, size=(n_samples, 1))
        rho2 = np.random.laplace(scale=6*self.c/self.eps, size=(n_samples, a.shape[0]))
        rho3 = np.random.laplace(scale=3*self.c/self.eps, size=(n_samples, a.shape[0]))

        m = rho2 + x  # broadcasts x vertically
        cmp = m >= (self.t + rho1)  # broadcasts rho1 horizontally
        z = rho3 + x  # broadcasts x vertically

        count = np.zeros(n_samples)
        aborted = np.full(n_samples, False)
        res = np.zeros(shape=(n_samples, a.shape[0]))

        for col_idx in range(0, a.shape[0]):
            above = cmp[:, col_idx]
            res[above, col_idx] = z[above, col_idx]
            count = count + above
            res[aborted, col_idx] = -1000.0
            aborted = np.logical_or(aborted, count == self.c)
        return res
