"""
Original StatDP algorithm implementations by Yuxing Wang
  MIT License, Copyright (c) 2018-2019 Yuxin Wang

The postprocessing step was removed from the algorithms
"""
import numpy as np


def noisy_max_v1a(prng, queries, epsilon):
    # find the largest noisy element and return its index
    return [(np.asarray(queries, dtype=np.float64) + prng.laplace(scale=2.0 / epsilon, size=len(queries))).argmax()]


def noisy_max_v1b(prng, queries, epsilon):
    # INCORRECT: returning maximum value instead of the index
    return [(np.asarray(queries, dtype=np.float64) + prng.laplace(scale=2.0 / epsilon, size=len(queries))).max()]


def noisy_max_v2a(prng, queries, epsilon):
    return [(np.asarray(queries, dtype=np.float64) + prng.exponential(scale=2.0 / epsilon, size=len(queries))).argmax()]


def noisy_max_v2b(prng, queries, epsilon):
    # INCORRECT: returning the maximum value instead of the index
    return [(np.asarray(queries, dtype=np.float64) + prng.exponential(scale=2.0 / epsilon, size=len(queries))).max()]


def histogram_eps(prng, queries, epsilon):
    # INCORRECT: using (epsilon) noise instead of (1 / epsilon)
    noisy_array = np.asarray(queries, dtype=np.float64) + prng.laplace(scale=epsilon, size=len(queries))
    return noisy_array.tolist()


def histogram(prng, queries, epsilon):
    noisy_array = np.asarray(queries, dtype=np.float64) + prng.laplace(scale=1.0 / epsilon, size=len(queries))
    return noisy_array.tolist()


def SVT(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        eta2 = prng.laplace(scale=4.0 * N / epsilon)
        if query + eta2 >= noisy_T:
            out.append(True)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out


def iSVT1(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    for query in queries:
        # INCORRECT: no noise added to the queries
        eta2 = 0
        if (query + eta2) >= noisy_T:
            out.append(True)
        else:
            out.append(False)
    return out


def iSVT2(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    for query in queries:
        # INCORRECT: noise added to queries doesn't scale with N
        eta2 = prng.laplace(scale=2.0 / epsilon)
        if (query + eta2) >= noisy_T:
            out.append(True)
            # INCORRECT: no bounds on the True's to output
        else:
            out.append(False)
    return out


def iSVT3(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=4.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        # INCORRECT: noise added to queries doesn't scale with N
        eta2 = prng.laplace(scale=4.0 / (3.0 * epsilon))
        if query + eta2 > noisy_T:
            out.append(True)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out


def iSVT4(prng, queries, epsilon, N, T):
    out = []
    eta1 = prng.laplace(scale=2.0 / epsilon)
    noisy_T = T + eta1
    c1 = 0
    for query in queries:
        eta2 = prng.laplace(scale=2.0 * N / epsilon)
        if query + eta2 > noisy_T:
            # INCORRECT: Output the noisy query instead of True
            out.append(query + eta2)
            c1 += 1
            if c1 >= N:
                break
        else:
            out.append(False)
    return out
