from typing import Tuple

from statdpwrapper.my_generate_counterexample import detect_counterexample

from statdpwrapper.postprocessing import PostprocessingConfig, get_postprocessed_algorithms, compose_postprocessing
from dpsniper.utils.my_logging import log, time_measure


class BinarySearch:
    """
    Helper class for transforming StatDP to a maximizing-power approach using binary search.
    """

    def __init__(self, algorithm, num_input, sensitivity, detect_iterations, default_kwargs, pp_config: PostprocessingConfig):
        self.default_kwargs = default_kwargs
        self.num_input = num_input
        self.sensitivity = sensitivity
        self.detect_iterations = detect_iterations

        self.all_postprocessed_algs = get_postprocessed_algorithms(algorithm, pp_config)
        self._nof_probes = 0

    def find(self, p_value_threshold: float, precision: float) -> Tuple:
        """
        Returns the tuple (epsilon, a1, a2, event, postprocessing, a0) with highest epsilon for which the p_value is
        still below p_value_threshold, up to a given precision. Here, a0 is the reference input to be used for
        HammingDistance postprocessing.
        """
        self._nof_probes = 0
        left, right = self._exponential_init(p_value_threshold)
        log.info("bounds for eps: [%f, %f]", left[0], right[0])

        res = self._binary_search(left, right, p_value_threshold, precision)
        log.info("required %d probes", self._nof_probes)
        log.data("statdp_nof_probes", self._nof_probes)
        return res

    def _probe(self, eps) -> Tuple:
        """
        Returns a tuple (p_value, (a1, a2, event, postprocessing, a0))

        a0 is the reference input to be used for HammingDistance postprocessing
        """
        log.info("checking eps = %f", eps)
        self._nof_probes += 1
        with time_measure("statdp_time_one_probe"):
            min_p_value = 1.0
            min_attack = None
            for pps in self.all_postprocessed_algs:
                log.info("trying postprocessing %s...", str(pps))
                self.default_kwargs['alg'] = pps
                result = detect_counterexample(compose_postprocessing,
                                               eps,
                                               num_input=self.num_input,
                                               default_kwargs=self.default_kwargs,
                                               sensitivity=self.sensitivity,
                                               detect_iterations=self.detect_iterations,
                                               quiet=True)
                del self.default_kwargs['alg']
                (_, p_value, d1, d2, kwargs, event) = result[0]
                if min_attack is None or p_value < min_p_value:
                    min_p_value = p_value
                    min_attack = (d1, d2, event, pps.postprocessing, kwargs['_d1'])

            log.info("p_value = %f", min_p_value)
            log.info("event = %s", min_attack)
            log.data("statdp_intermediate_probe", {"eps": eps, "p_value": min_p_value})
            return min_p_value, min_attack

    def _exponential_init(self, p_value_threshold: float):
        log.info("running exponential search")
        eps = 0.005
        p_value = 0.0
        attack = None
        prev_attack = None
        while p_value < p_value_threshold:
            prev_attack = attack
            eps *= 2
            p_value, attack = self._probe(eps)
        return (eps/2, prev_attack), (eps, attack)

    def _binary_search(self, left_tup, right_tup, p_value_threshold: float, precision: float):
        left = left_tup[0]    # eps
        right = right_tup[0]  # eps
        left_attack = left_tup[1]

        # invariant: p_value at left is strictly below p_value_threshold
        while right - left > precision:
            mid = left + ((right - left) / 2)
            p_value, mid_attack = self._probe(mid)
            if p_value < p_value_threshold:
                left = mid
                left_attack = mid_attack
            else:
                right = mid
        log.info("finished binary search")
        log.info("  eps = %f", left)
        log.info("  attack = %s", left_attack)
        return left, left_attack[0], left_attack[1], left_attack[2], left_attack[3], left_attack[4]
