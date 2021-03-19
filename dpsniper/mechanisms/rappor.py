import numpy as np
from dpsniper.mechanisms.abstract import Mechanism

import mmh3     # murmur hash 3, used for Bloom filter

from dpsniper.utils.zero import ZeroNoisePrng


class Rappor(Mechanism):
    """
    Steps 1--3 from:
        Ulfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. 2014.
        RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response. CCS 2014.
    """

    def __init__(self, n_hashes: int = 4, filter_size: int = 20, f: float = 0.75, p: float = 0.45, q: float = 0.55, prng=None):
        self.f = f
        self.q = q
        self.p = p
        self.n_hashes = n_hashes
        self.filter_size = filter_size

        if prng is None:
            self.prng = np.random.default_rng()
        else:
            self.prng = prng

    def m(self, a, n_samples: int = 1):
        assert(a.shape == (1,) or a.shape == ())
        val = a.item(0)

        # populate bloom filter
        filter = self._populate_bloom_filter(val, n_samples)

        if isinstance(self.prng, ZeroNoisePrng):
            # don't perform any randomization
            return filter

        # permanent randomized response
        self._apply_permanent_randomized_response(filter)

        # instantaneous randomized response
        res = self._get_instantaneous_randomized_response(filter)
        return res

    def _populate_bloom_filter(self, val, n_samples):
        filter = np.zeros((n_samples, self.filter_size))
        for i in range(0, self.n_hashes):
            hashval = mmh3.hash(str(val), seed=i) % self.filter_size
            filter[:, hashval] = 1
        return filter

    def _apply_permanent_randomized_response(self, filter):
        choices = np.random.choice(3, size=filter.shape, p=[0.5*self.f, 0.5*self.f, 1-self.f])
        filter[choices == 0] = 1
        filter[choices == 1] = 0

    def _get_instantaneous_randomized_response(self, filter):
        res = np.zeros(shape=filter.shape)
        set_to_1_q = np.logical_and(filter == 1, self.prng.binomial(n=1, p=self.q, size=filter.shape) == 1)
        set_to_1_p = np.logical_and(filter == 0, self.prng.binomial(n=1, p=self.p, size=filter.shape) == 1)
        res[set_to_1_p] = 1
        res[set_to_1_q] = 1
        return res


class OneTimeRappor(Rappor):
    """
    Steps 1--2 from:
        Ulfar Erlingsson, Vasyl Pihur, and Aleksandra Korolova. 2014.
        RAPPOR: Randomized Aggregatable Privacy-Preserving Ordinal Response. CCS 2014.
    """
    def __init__(self, n_hashes: int = 4, filter_size: int = 20, f: float = 0.95, prng=None):
        super().__init__(n_hashes=n_hashes, filter_size=filter_size, f=f, prng=prng)

    def m(self, a, n_samples=1):
        assert (a.shape == (1,) or a.shape == ())
        val = a.item(0)

        # populate bloom filter
        filter = self._populate_bloom_filter(val, n_samples)

        if isinstance(self.prng, ZeroNoisePrng):
            # don't perform any randomization
            return filter

        # permanent randomized response
        self._apply_permanent_randomized_response(filter)

        return filter
