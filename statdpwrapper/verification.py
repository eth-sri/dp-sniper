from typing import List

import numpy as np

from dpsniper.search.ddconfig import DDConfig
from statdpwrapper.postprocessing import Postprocessing
from dpsniper.probability.estimators import PrEstimator
from dpsniper.attack.attack import Attack

from statdpwrapper.postprocessing import the_zero_noise_prng


class StatDPAttack(Attack):
    """
    A wrapper for a StatDP event.
    Note: Does not really extend dpsniper.attack.Attack due to missing vectorization in check
    """

    def __init__(self, event, postprocessing: Postprocessing):
        """
        Creates an attack from a StatDP event and postprocessing.
        """
        self.event = event
        self.postprocessing = postprocessing
        self.noisefree_reference = None

    def set_noisefree_reference(self, noisefree_reference):
        self.noisefree_reference = noisefree_reference

    def check(self, b: List):
        """
        Computes the probabilities whether given outputs b lie in the attack set

        Args:
            b: list of list (outer list are samples, inner list are dimensions)
            noisefree_reference: noisefree reference evaluation (if required by postprocessing)

        Returns:
            boolean 1d array of shape (n_samples,)
        """
        if self.postprocessing.requires_noisefree_reference and self.noisefree_reference is None:
            raise ValueError("check(...) requires noisefree_reference")

        x = np.empty(shape=(len(b), self.postprocessing.n_output_dimensions()), dtype=float)
        for idx, sample in enumerate(b):    # loop over samples
            x[idx, :] = np.asarray(self.postprocessing.process(sample, noisefree_reference=self.noisefree_reference),
                                    dtype=float)
        return self._check_event(x)

    def _check_event(self, x):
        """
        Args:
            x: ndarray of shape (n_samples, d), dtype float

        Returns:
            ndarray of shape (n_samples,) containing probabilitites in [0.0, 1.0]
        """
        res = np.full(fill_value=True, shape=x.shape[0])
        for col in range(0, x.shape[1]):     # loop cover columns
            if isinstance(self.event[col], int) or isinstance(self.event[col], float):
                # equality check
                res = np.logical_and(res, x[:, col] == self.event[col])
            else:
                # open interval check
                low = self.event[col][0]
                high = self.event[col][1]
                res = np.logical_and(res, np.logical_and(
                    x[:, col] > low,
                    x[:, col] < high
                ))
        return res.astype(float)


class StatDPPrEstimator(PrEstimator):
    """
    A probability estimator based on samples for StatDP algorithm implementations.
    """

    def __init__(self, mechanism, n_samples: int, config: DDConfig, use_parallel_executor=False, **kwargs):
        super().__init__(mechanism, n_samples, config, use_parallel_executor)
        self.mechanism_kwargs = kwargs

    def _get_samples(self, a, n_samples):
        # non-vectorized variant for StatDP
        l = [0] * n_samples
        for i in range(0, n_samples):
            l[i] = self.mechanism(np.random.default_rng(), a, **self.mechanism_kwargs)
        return l

    def _get_noisefree_reference(self, a):
        return self.mechanism(the_zero_noise_prng, a, **self.mechanism_kwargs)
