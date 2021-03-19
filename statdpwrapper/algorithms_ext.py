"""
Algorithm implementations not provided with StatDP
"""
import numpy as np

from dpsniper.mechanisms.rappor import Rappor, OneTimeRappor
from dpsniper.mechanisms.geometric import TruncatedGeometricMechanism
from dpsniper.utils.my_logging import log
from dpsniper.utils.zero import ZeroNoisePrng
from statdpwrapper.algorithms import iSVT3, iSVT4


def laplace_mechanism(prng, queries, epsilon):
    return [prng.laplace(loc=queries[0], scale=1.0 / epsilon)]


def laplace_parallel(prng, queries, epsilon, n_parallel):
    l = []
    for i in range(0, n_parallel):
        l = l + laplace_mechanism(prng, queries, epsilon)
    return l


def noisy_hist_1_sorted(prng, queries, epsilon):
    x = np.asarray(queries, dtype=np.float64) + prng.laplace(scale=1.0 / epsilon, size=len(queries))
    x = np.sort(x)
    return x.tolist()


def svt_34_parallel(prng, queries, epsilon, N, T):
    # NOTE iSVT4 = SVT3, and iSVT3 = SVT4
    svt3_res = iSVT4(prng, queries, epsilon, N, T)
    svt4_res = iSVT3(prng, queries, epsilon, N, T)

    # padding such that parallel runs can be distinguished
    for i in range(len(svt3_res), len(queries)):
        svt3_res.append(None)
    for i in range(len(svt4_res), len(queries)):
        svt4_res.append(None)
    return svt3_res + svt4_res


def SVT2(prng, queries, epsilon, N, T):
    """
    Alg. 2 from:
        M. Lyu, D. Su, and N. Li. 2017.
        Understanding the Sparse Vector Technique for Differential Privacy.
        Proceedings of the VLDB Endowment.

    Modification of implementation by Yuxing Wang
        MIT License, Copyright (c) 2018-2019 Yuxin Wang
    """
    out = []
    eta1 = prng.laplace(scale=2.0 * N / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        eta2 = prng.laplace(scale=4.0 * N / epsilon)
        if query + eta2 >= noisy_T:
            out.append(True)
            eta1 = prng.laplace(scale=2.0 * N / epsilon)
            noisy_T = T + eta1
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out


def rappor(prng, queries, epsilon, n_hashes, filter_size, f, q, p):
    rap = Rappor(n_hashes, filter_size, f, q, p, prng=prng)
    return rap.m(np.array(queries[0], dtype=float)).tolist()[0]


def one_time_rappor(prng, queries, epsilon, n_hashes, filter_size, f):
    rap = OneTimeRappor(n_hashes, filter_size, f, prng=prng)
    return rap.m(np.array(queries[0], dtype=float)).tolist()[0]


def truncated_geometric(prng, queries, epsilon, n):
    if isinstance(prng, ZeroNoisePrng):
        # TruncatedGeometricMechanism with all-zero randomness is not equivalent to
        # noise-free evaluation. Because this mechanism has a numeric output type,
        # HammingDistance postprocessing should not be applied and noise-free evaluation
        # should not be required.
        log.error("truncated geometric mechanism does not support noise-free evaluation")
        raise NotImplementedError()
    geom = TruncatedGeometricMechanism(epsilon, n)
    return [geom.m(np.array(queries[0], dtype=int))]


def prefix_sum(prng, queries, epsilon):
    """
    PrefixSum from:
        Y. Wang, Z. Ding, G. Wang, D. Kifer, D. Zhang. 2019.
        Proving differential privacy with shadow execution.
        PLDI 2019.
    """
    out = []
    s = 0.0
    for query in queries:
        s = s + query + prng.laplace(scale=1.0/epsilon)
        out.append(s)
    return out


def numerical_svt(prng, queries, epsilon, N, T):
    """
    Numerical Sparse Vector Technique from:
        Y. Wang, Z. Ding, G. Wang, D. Kifer, D. Zhang. 2019.
        Proving differential privacy with shadow execution.
        PLDI 2019.

    Modification of SVT implementation by Yuxing Wang
        MIT License, Copyright (c) 2018-2019 Yuxin Wang
    """
    out = []
    eta1 = prng.laplace(scale=3.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        eta2 = prng.laplace(scale=6.0 * N / epsilon)
        if query + eta2 >= noisy_T:
            eta3 = prng.laplace(scale=3.0 * N / epsilon)
            out.append(query + eta3)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(0.0)
    return out
