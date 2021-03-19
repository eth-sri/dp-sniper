from dpsniper.attack.dpsniper import DPSniper
from dpsniper.input.input_pair_generator import InputPairGenerator
from dpsniper.probability.estimators import PrEstimator, EpsEstimator
from dpsniper.mechanisms.abstract import Mechanism
from dpsniper.utils.my_logging import log, time_measure
from dpsniper.utils.my_multiprocessing import the_parallel_executor
from dpsniper.search.ddconfig import DDConfig
from dpsniper.search.ddwitness import DDWitness


import os


class DDSearch:
    """
    The main DD-Search algorithm for testing differential privacy.
    """

    def __init__(self,
                 mechanism: Mechanism,
                 attack_optimizer: DPSniper,
                 input_generator: InputPairGenerator,
                 config: DDConfig):
        """
        Creates the optimizer.

        Args:
            mechanism: mechanism to test
            attack_optimizer: optimizer finding attacks for given input pairs
            input_generator: generator of input pairs
            config: configuration
        """
        self.mechanism = mechanism
        self.attack_optimizer = attack_optimizer
        self.input_generator = input_generator
        self.config = config

        self.pr_estimator = PrEstimator(mechanism, self.config.n_check, self.config)

    def run(self) -> DDWitness:
        """
        Runs the optimizer and returns the result.
        """
        # compute intermediate results (approximate eps)
        with time_measure("time_dp_distinguisher_all_inputs"):
            results = self._compute_results_for_all_inputs()

        # find best result
        best = None
        for res in results:
            if best is None or res > best:
                best = res

        log.data('best_result', best.to_json())
        return best

    def _compute_results_for_all_inputs(self):
        log.debug("generating inputs...")
        inputs = []
        for (a1, a2) in self.input_generator.get_input_pairs():
            log.debug("%s, %s", a1, a2)
            inputs.append((self, a1, a2))

        log.debug("submitting parallel tasks...")
        result_files = the_parallel_executor.execute(DDSearch._one_input_pair, inputs)
        log.debug("parallel tasks done!")

        results = []
        for filename in result_files:
            cur = DDWitness.from_file(filename)
            os.remove(filename)
            results.append(cur)
        return results

    @staticmethod
    def _one_input_pair(task):
        optimizer, a1, a2 = task
        pr_estimator = EpsEstimator(optimizer.pr_estimator)

        log.debug("selecting attack...")
        with time_measure("time_dp_distinguisher"):
            attack = optimizer.attack_optimizer.best_attack(a1, a2)
        log.debug("best attack: %s", attack)

        cur = DDWitness(a1, a2, attack)
        log.debug("computing estimate for eps...")
        with time_measure("time_estimate_eps"):
            cur.compute_eps_using_estimator(pr_estimator)
        log.debug("current eps: %s", cur.eps)
        log.data("eps_for_sample", cur.eps)

        log.debug("storing result...")
        filename = cur.to_tmp_file()

        log.debug("done!")
        return filename

