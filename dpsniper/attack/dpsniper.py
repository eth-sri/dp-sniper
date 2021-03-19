import math
from typing import Tuple

import numpy as np

from dpsniper.attack.ml_attack import MlAttack
from dpsniper.classifiers.classifier_factory import ClassifierFactory
from dpsniper.classifiers.stable_classifier import StableClassifier
from dpsniper.mechanisms.abstract import Mechanism
from dpsniper.search.ddconfig import DDConfig
from dpsniper.utils.my_logging import log, time_measure
from dpsniper.utils.my_multiprocessing import split_into_parts, split_by_batch_size


class DPSniper:
    """
    The main DP-Sniper algorithm. This optimizer finds an MlAttack for a given input pair by
    training a classifier and deriving attack thresholds.
    """

    def __init__(self, mechanism: Mechanism, classifier_factory: ClassifierFactory, config: DDConfig):
        """
        Create an optimizer.

        Args:
            mechanism: mechanism to attack
            classifier_factory: factory creating instances the classifier to be used for the attack
            config: configuration
        """
        self.mechanism = mechanism
        self.classifier_factory = classifier_factory
        self.config = config

    def best_attack(self, a1, a2) -> MlAttack:
        """
        Runs the optimizer to construct an attack for given input pair a1, a2.

        Args:
            a1: 1d array representing the first input
            a2: 1d array representing the second input

        Returns:
            The constructed MlAttack
        """
        log.debug("Searching best attack for mechanism %s, classifier %s...",
                  type(self.mechanism).__name__,
                  type(self.classifier_factory).__name__)

        classifier = self._train_classifier(a1, a2)

        with time_measure("time_determine_threshold"):
            log.debug("Determining threshold...")

            # TODO: maybe parallelize this loop?
            probabilities = []
            for parallel_size in split_into_parts(self.config.n, self.config.n_processes):
                sequential_probabilities = []
                for sequential_size in split_by_batch_size(parallel_size, self.config.prediction_batch_size):
                    # generate samples from a2
                    b_new = self.mechanism.m(a2, sequential_size)
                    if len(b_new.shape) == 1:
                        # make sure b1 and b2 have shape (n_samples, 1)
                        b_new = np.atleast_2d(b_new).T

                    # compute Pr[a1 | M(a1) = b_new]
                    probabilities_new = classifier.predict_probabilities(b_new)

                    # wrap up
                    sequential_probabilities.append(probabilities_new)

                sequential_probabilities = np.concatenate(sequential_probabilities)
                probabilities.append(sequential_probabilities)

            probabilities = np.concatenate(probabilities)
            probabilities[::-1].sort()  # sorts descending, in-place

            assert(probabilities.shape[0] == self.config.n)

        # find optimal threshold
        log.debug("Finding optimal threshold...")
        with time_measure("time_dp_distinguisher_find_threshold"):
            thresh, q = DPSniper._find_threshold(probabilities, self.config.c * probabilities.shape[0])
        log.debug("Selected t = %f, q = %f", thresh, q)

        return MlAttack(classifier, thresh, q)

    def _train_classifier(self, a1, a2) -> StableClassifier:
        """
        Trains the classifier for inputs a1, a2.
        """
        def generate_batches():
            for size in split_by_batch_size(self.config.n_train, self.config.training_batch_size):
                yield self._generate_data_batch(a1, a2, size)

        log.debug("Creating classifier...")
        classifier = self.classifier_factory.create()

        log.debug("Training classifier...")
        with time_measure("time_dp_distinguisher_train"):
            classifier.train(generate_batches())
        log.debug("Done training")

        return classifier

    def _generate_data_batch(self, a1, a2, n) -> Tuple:
        """
        Generates a training data batch of size 2n (n samples for each input a1 and a2).
        """
        log.debug("Generating training data batch of size 2*%d...", n)

        b1 = self.mechanism.m(a1, n)
        b2 = self.mechanism.m(a2, n)
        if len(b1.shape) == 1:
            # make sure b1 and b2 have shape (n_samples, 1)
            b1 = np.atleast_2d(b1).T
            b2 = np.atleast_2d(b2).T

        # rows = samples, columns = features
        x = np.concatenate((b1, b2), axis=0)

        # 1d array of labels
        y = np.zeros(2 * n)
        y[n:] = 1

        return x, y

    @staticmethod
    def _find_threshold(sorted_probs, target: float) -> (float, float):
        """
        Finds threshold t and tie-breaker q such that a given target fraction lies above t.

        Args:
            sorted_probs: 1d array of probabilities, sorted descending
            target: target fraction

        Returns:
            pair (t, q) of threshold t and tie-breaker q
        """
        thresh = sorted_probs[min(math.floor(target), sorted_probs.shape[0] - 1)]

        # find number of samples strictly above thresh
        n_above = np.sum(sorted_probs > thresh)

        # find number of samples equal to thresh
        n_equal = np.sum(sorted_probs == thresh)

        # split remaining weight
        q = (target - n_above) / n_equal

        return thresh, q