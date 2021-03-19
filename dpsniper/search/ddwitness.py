import pickle
import tempfile
import os
from typing import Tuple

from dpsniper.search.ddconfig import DDConfig
from dpsniper.probability.estimators import PrEstimator, EpsEstimator
from dpsniper.mechanisms.abstract import Mechanism
from dpsniper.utils.my_logging import log
from dpsniper.utils.paths import get_output_directory


class DDWitness:
    """
    A representation of a DD witness.
    """

    def __init__(self, a1, a2, attack: 'Attack'):
        """
        Args:
            a1: 1d array representing the first input
            a2: 1d array representing the second input
            attack: the attack
        """
        self.a1 = a1
        self.a2 = a2
        self.attack = attack
        self.eps = None     # estimate of epsilon
        self.lower_bound = None     # lower bound on epsilon

    def compute_eps_using_estimator(self, estimator: EpsEstimator):
        """
        Computes epsilon and its lower bound using the provided probability estimator.
        Sets self.eps and self.lower_bound.
        """
        self.eps, self.lower_bound = estimator.compute_eps_estimate(self.a1, self.a2, self.attack)

    def compute_eps_high_precision(self, mechanism: Mechanism, config: DDConfig) -> Tuple:
        """
        Computes epsilon and its lower bound using high precision as specified by config.n_final.
        Sets self.eps and self.lower_bound.
        """
        self.compute_eps_using_estimator(EpsEstimator(PrEstimator(mechanism, config.n_final, config, True)))

    def to_tmp_file(self) -> str:
        """
        Stores the result to a temporary file.

        Returns:
            The path of the created temporary file.
        """
        tmp_dir = get_output_directory("tmp")
        fd, filename = tempfile.mkstemp(dir=tmp_dir)
        log.debug("Storing result to file '%s'", filename)
        with os.fdopen(fd, "wb") as f:
            pickle.dump(self, f)
        return filename

    @staticmethod
    def from_file(filename):
        """
        Loads a DDWitness object from a file with given name.
        """
        log.debug("Loading result from file '%s'", filename)
        with open(filename, "rb") as f:
            obj = pickle.load(f)
        assert(type(obj) == DDWitness)
        return obj

    def __lt__(self, other):
        return self.eps < other.eps

    def __eq__(self, other):
        return self.eps == other.eps

    def __str__(self):
        d = {str(k): str(v) for k, v in self.__dict__.items()}
        return str(d)

    def to_json(self):
        d = {str(k): v if isinstance(v, float) else str(v) for k, v in self.__dict__.items()}
        return d
