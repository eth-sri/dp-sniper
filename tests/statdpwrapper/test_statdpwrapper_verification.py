import unittest

from dpsniper.search.ddconfig import DDConfig
from statdpwrapper.verification import StatDPAttack, StatDPPrEstimator
from statdpwrapper.postprocessing import *
from statdpwrapper.algorithms import iSVT4, SVT
from tests.my_test_case import MyTestCase

config = DDConfig()


class TestStatDPVerification(MyTestCase):

    def test_check_event_1d_tuple(self):
        event = ((4.5, 5.0),)
        attack = StatDPAttack(event, None)
        x = np.array([[4.5], [4.6]])
        res = attack._check_event(x)
        np.array_equal(np.array([0.0, 1.0]), res)

    def test_check_event_1d_singleton(self):
        event = (3,)
        attack = StatDPAttack(event, None)
        x = np.array([[2], [3], [1]])
        res = attack._check_event(x)
        np.array_equal(np.array([0.0, 1.0, 0.0]), res)

    def test_check_event_2d_mixed(self):
        event = (2, (float("-Infinity"), 3.5))
        attack = StatDPAttack(event, None)
        x = np.array([[2, 4.0], [2, 3.0], [1, 3.0]])
        res = attack._check_event(x)
        np.array_equal(np.array([0.0, 1.0, 0.0]), res)

    def test_check_1d_tuple(self):
        event = ((float("-Infinity"), 2.8),)
        pp = AvgPP()
        attack = StatDPAttack(event, pp)
        x = [[2, 4.0], [2, 3.0], [1, 3.0]]
        res = attack.check(x)
        np.array_equal(np.array([0.0, 1.0, 1.0]), res)

    def test_check_1d_singleton(self):
        event = (0,)
        pp = EntryPP(0)
        attack = StatDPAttack(event, pp)
        x = [[1, 0, 0], [0, 1, 1]]
        res = attack.check(x)
        np.array_equal(np.array([0.0, 1.0]), res)

    def test_check_2d_mixed(self):
        event = (2, (float("-Infinity"), 3.5))
        pp = CombinedPP(LengthPP(), AvgPP())
        attack = StatDPAttack(event, pp)
        x = [[2, 4.0], [2, 3.0], [1, 3.0]]
        res = attack.check(x)
        np.array_equal(np.array([0.0, 1.0, 1.0]), res)

    def test_check_hamming_distance(self):
        event = (2,)
        pp = HammingDistancePP()
        attack = StatDPAttack(event, pp)
        x = [[True, False], [False, False], [True, True]]
        ref = [True, True]
        attack.set_noisefree_reference(ref)
        res = attack.check(x)
        np.array_equal(np.array([0.0, 1.0, 0.0]), res)

    def test_svt4_check(self):
        event = (9, (-2.4, 2.4))
        pp = CombinedPP(CountPP(False), EntryPP(9))
        attack = StatDPAttack(event, pp)
        l = []
        for i in range(0, 10000):
            l.append(iSVT4(np.random.default_rng(), [1, 1, 1, 1, 1, 0, 0, 0, 0, 0], 0.1, 2, 0))
        res = attack.check(l)
        self.assertEqual(10000, res.shape[0])

    def test_pr_estimation(self):
        est = StatDPPrEstimator(iSVT4, 10000, config, epsilon=0.1, N=2, T=0)
        event = (9, (-2.4, 2.4))
        pp = CombinedPP(CountPP(False), EntryPP(9))
        attack = StatDPAttack(event, pp)
        pr = est.compute_pr_estimate([1]*10, attack)
        self.assertTrue(0 <= pr <= 1)

    def test_pr_estimation_hamming_distance(self):
        est = StatDPPrEstimator(SVT, 10000, config, epsilon=0.1, N=2, T=1)
        event = (2,)
        pp = HammingDistancePP()
        attack = StatDPAttack(event, pp)
        attack.set_noisefree_reference([0]*10)
        pr = est.compute_pr_estimate([0]*10, attack)
        self.assertTrue(0 <= pr <= 1)
