import numpy as np

from dpsniper.classifiers.stable_classifier import StableClassifier
from dpsniper.classifiers.torch_optimizer_factory import AdamOptimizerFactory
from dpsniper.mechanisms.laplace import LaplaceMechanism
from dpsniper.mechanisms.report_noisy_max import ReportNoisyMax1
from dpsniper.mechanisms.noisy_hist import NoisyHist1
from dpsniper.attack.ml_attack import MlAttack
from dpsniper.attack.dpsniper import DPSniper
from dpsniper.classifiers.classifier_factory import LogisticRegressionFactory
from dpsniper.search.ddsearch import DDConfig
from tests.my_test_case import MyTestCase


class TestAttack(MyTestCase):

    def test_ml_attack_optimizer_1d_1d(self):
        mechanism = LaplaceMechanism()
        aopt = DPSniper(mechanism, LogisticRegressionFactory(in_dimensions=1,
                                                             optimizer_factory=AdamOptimizerFactory()),
                        DDConfig(n_train=10))
        attack = aopt.best_attack(np.array([0.4]), np.array([1.4]))
        res = attack.check(np.array([2.0]))
        self.assertIsInstance(attack, MlAttack)
        self.assertEqual((1,), res.shape)

    def test_ml_attack_optimizer_nd_1d(self):
        mechanism = ReportNoisyMax1()
        aopt = DPSniper(mechanism, LogisticRegressionFactory(in_dimensions=1,
                                                             optimizer_factory=AdamOptimizerFactory()),
                        DDConfig(n_train=10))
        attack = aopt.best_attack(np.array([0.4, 2.8]), np.array([1.4, 1.8]))
        res = attack.check(np.array([0]))
        self.assertIsInstance(attack, MlAttack)
        self.assertEqual((1,), res.shape)

    def test_ml_attack_optimizer_nd_nd(self):
        mechanism = NoisyHist1()
        aopt = DPSniper(mechanism, LogisticRegressionFactory(in_dimensions=2,
                                                             optimizer_factory=AdamOptimizerFactory()),
                        DDConfig(n_train=10))
        attack = aopt.best_attack(np.array([5, 3]), np.array([4, 3]))
        res = attack.check(np.array([[5.1, 3.0], [3.2, 4.3], [0.1, 2.8]]))
        self.assertIsInstance(attack, MlAttack)
        self.assertEqual((3,), res.shape)

    def test_train_internal(self):
        mechanism = NoisyHist1()
        aopt = DPSniper(mechanism, LogisticRegressionFactory(in_dimensions=2,
                                                             optimizer_factory=AdamOptimizerFactory(),
                                                             n_test_batches=1),
                        DDConfig(n_train=30, training_batch_size=10))
        classifier = aopt._train_classifier(np.array([5, 3]), np.array([4, 3]))
        self.assertIsInstance(classifier, StableClassifier)

    def test_find_threshold_1(self):
        probs = np.array([0.9, 0.8, 0.6, 0.6, 0.6, 0.6, 0.3])
        target = 3.2
        thresh, q = DPSniper._find_threshold(probs, target)
        self.assertEqual(0.6, thresh)
        self.assertAlmostEqual(0.3, q)

    def test_find_threshold_2(self):
        probs = np.array([0.9, 0.8, 0.6, 0.6, 0.6, 0.6, 0.3])
        target = 3.0
        thresh, q = DPSniper._find_threshold(probs, target)
        self.assertEqual(0.6, thresh)
        self.assertAlmostEqual(0.25, q)

    def test_find_threshold_3(self):
        probs = np.array([0.9, 0.8, 0.6, 0.6, 0.6, 0.6, 0.3])
        target = 1.0
        thresh, q = DPSniper._find_threshold(probs, target)
        self.assertEqual(0.8, thresh)
        self.assertEqual(0.0, q)

    def test_find_threshold_4(self):
        probs = np.array([0.9, 0.8, 0.6, 0.6, 0.6, 0.6, 0.3])
        target = 2.0
        thresh, q = DPSniper._find_threshold(probs, target)
        self.assertEqual(0.6, thresh)
        self.assertEqual(0.0, q)

    def test_find_threshold_5(self):
        probs = np.array([0.9, 0.8, 0.6, 0.6, 0.6, 0.6, 0.3])
        target = 7.0
        thresh, q = DPSniper._find_threshold(probs, target)
        self.assertEqual(0.3, thresh)
        self.assertEqual(1.0, q)

    def test_find_threshold_6(self):
        probs = np.array([0.9, 0.8, 0.6, 0.6, 0.6, 0.6, 0.3])
        target = 0.0
        thresh, q = DPSniper._find_threshold(probs, target)
        self.assertEqual(0.9, thresh)
        self.assertEqual(0.0, q)

    def test_find_threshold_7(self):
        probs = np.array([0.9, 0.8, 0.6, 0.6, 0.6, 0.6, 0.3])
        target = 0.4
        thresh, q = DPSniper._find_threshold(probs, target)
        self.assertEqual(0.9, thresh)
        self.assertAlmostEqual(0.4, q)

    def test_compute_above_thresh_probs(self):
        probs = np.array([0.9, 0.8, 0.6, 0.3])
        attack = MlAttack(None, 0.6, 0.3)
        res = attack._compute_above_thresh_probs(probs)
        self.assertEqual((4,), res.shape)
        self.assertEqual([1.0, 1.0, 0.3, 0.0], res.tolist())
