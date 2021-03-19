from dpsniper.mechanisms.abstract import Mechanism
from dpsniper.attack.attack import Attack
from dpsniper.utils.my_logging import log
from dpsniper.utils.my_multiprocessing import the_parallel_executor, split_by_batch_size, split_into_parts
from dpsniper.probability.binomial_cdf import lcb, ucb
from dpsniper.search.ddconfig import DDConfig

import numpy as np
import math


class PrEstimator:
    """
    Class for computing an estimate of Pr[M(a) in S].
    """

    def __init__(self, mechanism: Mechanism, n_samples: int, config: DDConfig, use_parallel_executor: bool = False):
        """
        Creates an estimator.

        Args:
            mechanism: mechanism
            n_samples: number of samples used to estimate the probability
            use_parallel_executor: whether to use the global parallel executor for probability estimation.
        """
        self.mechanism = mechanism
        self.n_samples = n_samples
        self.use_parallel_executor = use_parallel_executor
        self.config = config

    def compute_pr_estimate(self, a, attack: Attack) -> float:
        """
        Returns:
             An estimate of Pr[M(a) in S]
        """
        if not self.use_parallel_executor:
            frac_cnt = PrEstimator._compute_frac_cnt((self, attack, a, self.n_samples))
        else:
            inputs = [(self, attack, a, batch) for batch in split_into_parts(self.n_samples, self.config.n_processes)]
            res = the_parallel_executor.execute(PrEstimator._compute_frac_cnt, inputs)
            frac_cnt = math.fsum(res)
        pr = frac_cnt / self.n_samples
        return pr

    def _get_samples(self, a, n_samples):
        return self.mechanism.m(a, n_samples=n_samples)

    def _check_attack(self, bs, attack):
        return attack.check(bs)

    @staticmethod
    def _compute_frac_cnt(args):
        pr_estimator, attack, a, n_samples = args

        frac_counts = []
        for sequential_size in split_by_batch_size(n_samples, pr_estimator.config.prediction_batch_size):
            bs = pr_estimator._get_samples(a, sequential_size)
            res = pr_estimator._check_attack(bs, attack)
            frac_counts += [math.fsum(res)]

        return math.fsum(frac_counts)

    def get_variance(self):
        """
        Returns the variance of estimations
        """
        return 1.0/(4.0*self.n_samples)


class EpsEstimator:
    """
    Class for computing an estimate of
        eps(a, a', S) = log(Pr[M(a) in S]) - log(Pr[M(a') in S])
    """

    def __init__(self, pr_estimator: PrEstimator, allow_swap: bool = False):
        """
        Creates an estimator.

        Args:
            pr_estimator: the PrEstimator used to estimate probabilities based on samples
            allow_swap: whether probabilities may be swapped
        """
        self.pr_estimator = pr_estimator
        self.allow_swap = allow_swap

    def compute_eps_estimate(self, a1, a2, attack: Attack) -> (float, float):
        """
        Estimates eps(a2, a2, attack) using samples.

        Returns:
            a tuple (eps, lcb), where eps is the eps estimate and lcb is a lower confidence bound for eps
        """
        p1 = self.pr_estimator.compute_pr_estimate(a1, attack)
        p2 = self.pr_estimator.compute_pr_estimate(a2, attack)
        log.debug("p1=%f, p2=%f", p1, p2)
        log.data("p1", p1)
        log.data("p2", p2)

        if p1 < p2:
            if self.allow_swap:
                p1, p2 = p2, p1
                log.debug("swapped probabilitites p1, p2")
            else:
                log.warning("probability p1 < p2 for eps estimation")

        eps = self._compute_eps(p1, p2)
        lcb = self._compute_lcb(p1, p2)
        return eps, lcb

    @staticmethod
    def _compute_eps(p1, p2):
        if p1 > 0 and p2 == 0:
            eps = float("infinity")
        elif p1 <= 0:
            eps = 0
        else:
            eps = np.log(p1) - np.log(p2)
        return eps

    def _compute_lcb(self, p1, p2):
        n_samples = self.pr_estimator.n_samples
        # confidence accounts for the fact that two bounds could be incorrect (union bound)
        confidence = 1 - (1-self.pr_estimator.config.confidence) / 2
        p1_lcb = lcb(n_samples, int(p1 * n_samples), 1-confidence)
        p2_ucb = ucb(n_samples, int(p2 * n_samples), 1-confidence)
        return self._compute_eps(p1_lcb, p2_ucb)
