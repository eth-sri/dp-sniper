import unittest
import numpy as np

from dpsniper.attack.dpsniper import DPSniper
from dpsniper.classifiers.classifier_factory import LogisticRegressionFactory
from dpsniper.classifiers.torch_optimizer_factory import SGDOptimizerFactory
from dpsniper.input.input_pair_generator import InputPairGenerator
from dpsniper.mechanisms.abstract import Mechanism
from dpsniper.search.ddsearch import DDSearch
from dpsniper.search.ddconfig import DDConfig
from dpsniper.utils.my_multiprocessing import initialize_parallel_executor
from dpsniper.utils.paths import get_output_directory
from tests.my_test_case import MyTestCase


class ConstantMechanism(Mechanism):

    def m(self, a, n_samples=1):
        return a*np.ones(n_samples)


class SingleInputPairGenerator(InputPairGenerator):

    def get_input_pairs(self):
        yield 20, 20


class TestSearch(MyTestCase):

    def test_run(self):
        with initialize_parallel_executor(2, get_output_directory()):
            config = DDConfig(n_train=10, n=10, n_check=10, n_final=10, n_processes=2)
            mechanism = ConstantMechanism()

            optimizer_factory = SGDOptimizerFactory()
            classifier_factory = LogisticRegressionFactory(in_dimensions=1, optimizer_factory=optimizer_factory)
            attack_optimizer = DPSniper(mechanism, classifier_factory, config)

            input_generator = SingleInputPairGenerator()
            optimizer = DDSearch(mechanism, attack_optimizer, input_generator, config)

            best = optimizer.run()
            self.assertEqual(best.a1, 20)
            self.assertEqual(best.a2, 20)
            self.assertEqual(best.eps, 0)
