import unittest

from statdpwrapper.postprocessing import *
from statdpwrapper.algorithms import *
from statdpwrapper.algorithms_ext import *
from tests.my_test_case import MyTestCase


def my_numeric_algorithm(prng, queries, epsilon, **kwargs):
    return [3, 4, 2]


def my_categorical_algorithm(prng, queries, epsilon, **kwargs):
    return [True, True, False]


def my_mixed_algorithm(prng, queries, epsilon, **kwargs):
    return [True, 3]


class TestPostprocessing(MyTestCase):

    def test_numeric_postprocessing(self):
        pp_config = PostprocessingConfig(True, False, True, 3)
        algs = get_postprocessed_algorithms(my_numeric_algorithm, pp_config)

        # entry 0, entry 1, entry 2, avg, min, max, length
        res = []
        for a in algs:
            res.append(compose_postprocessing(None, None, None, alg=a, _d1=None))
        self.assertListEqual([3, 4, 2, 3, 2, 4, 3], res)

    def test_categorical_postprocessing(self):
        pp_config = PostprocessingConfig(False, True, False, 3, categories=[True, False])
        algs = get_postprocessed_algorithms(my_categorical_algorithm, pp_config)

        # count True, count False, hamming
        res = []
        for a in algs:
            res.append(compose_postprocessing(None, None, None, alg=a, _d1=None))
        self.assertListEqual([2, 1, 0], res)

    def test_mixed_postprocessing(self):
        pp_config = PostprocessingConfig(True, True, True, 3, categories=[True, False])
        algs = get_postprocessed_algorithms(my_mixed_algorithm, pp_config)

        # count True, count False, entry 1, entry 2, avg, min, max,
        # count True entry 0, count false entry 0,
        # count True entry 1, count false entry 1,
        # count True entry 2, count false entry 2,
        # count True avg, count false avg,
        # count true min, count false min,
        # count true max, count false max,
        # length entry 0, length entry 1, length entry 2, length avg , length min,
        # length max, count true length, count false length, length
        res = []
        for a in algs:
            res.append(compose_postprocessing(None, None, None, alg=a, _d1=None))
        self.assertListEqual([1, 0, True, 3, 0, 3.0, 3, 3, (1, True), (0, True), (1, 3), (0, 3), (1, 0), (0, 0),
                              (1, 3.0), (0, 3.0), (1, 3), (0, 3), (1, 3), (0, 3), (2, True), (2, 3), (2, 0), (2, 3.0),
                              (2, 3), (2, 3), (1, 2), (0, 2), 2],
                             res)

    def test_postprocessing_interop(self):
        algs_num_1 = get_postprocessed_algorithms(noisy_max_v1a, PostprocessingConfig(True, False, False, 1)) +\
            get_postprocessed_algorithms(noisy_max_v1b, PostprocessingConfig(True, False, False, 1)) +\
            get_postprocessed_algorithms(noisy_max_v2a, PostprocessingConfig(True, False, False, 1)) +\
            get_postprocessed_algorithms(noisy_max_v2b, PostprocessingConfig(True, False, False, 1))
        algs_num_4 = get_postprocessed_algorithms(histogram, PostprocessingConfig(True, False, False, 4)) +\
            get_postprocessed_algorithms(histogram_eps, PostprocessingConfig(True, False, False, 4))
        algs_cat = get_postprocessed_algorithms(SVT, PostprocessingConfig(False, True, True, 4, [True, False])) +\
            get_postprocessed_algorithms(iSVT1, PostprocessingConfig(False, True, False, 4, [True, False])) +\
            get_postprocessed_algorithms(iSVT2, PostprocessingConfig(False, True, False, 4, [True, False])) +\
            get_postprocessed_algorithms(iSVT3, PostprocessingConfig(False, True, True, 4, [True, False]))
        algs_mixed = get_postprocessed_algorithms(iSVT4, PostprocessingConfig(True, True, True, 4, [True, False]))
        for a in algs_num_1:
            compose_postprocessing(np.random.default_rng(), [2.3], 0.1, alg=a, _d1=[2.3])
        for a in algs_num_4:
            compose_postprocessing(np.random.default_rng(), [2.3, 1.0, 4.3, 2.2], 0.1, alg=a, _d1=[2.3, 1.0, 4.3, 2.2])
        for a in algs_cat:
            compose_postprocessing(np.random.default_rng(), [True, True, False, True], 0.1, alg=a, N=2, T=0, _d1=[True, True, False, True])
        for a in algs_mixed:
            compose_postprocessing(np.random.default_rng(), [True, 1.2, False, True], 0.1, alg=a, N=2, T=0, _d1=[True, 1.2, False, True])

    def test_postprocessing_padded(self):
        self.assertEqual(1, HammingDistancePP().process([None, True, False], noisefree_reference=[None, None, False]))
        self.assertEqual(2, CountPP(None).process([None, 2.0, False, None]))
        self.assertEqual(0, EntryPP(0).process([None]))
        self.assertEqual(2.0, AvgPP().process([None, 2.0, None]))
        self.assertEqual(2.0, MinPP().process([None, 2.0, None]))
        self.assertEqual(2.0, MaxPP().process([None, 2.0, None]))

    def test_zero_noise_prng(self):
        a = the_zero_noise_prng.laplace(scale=0.5, size=4)
        b = the_zero_noise_prng.exponential(scale=0.5)
        np.array_equal(np.array([0, 0, 0, 0]), a)
        np.array_equal(np.array([0]), b)

    def test_SVT_with_zero_noise_prng(self):
        x = SVT(the_zero_noise_prng, [0, 1, 0, 1, 1, 0], 0.1, 2, 0.1)
        self.assertListEqual([False, True, False, True], x)

    def test_hamming_distance_postprocessing(self):
        alg = PostprocessedAlgorithm(SVT, HammingDistancePP())
        x = compose_postprocessing(np.random.default_rng(), [0, 1, 0, 1, 1, 0], 0.1, N=2, T=0.1, alg=alg,
                                   _d1=[1, 1, 1, 0, 0, 0])
        self.assertIsInstance(x, int)

    def test_combined_hamming_distance_postprocessing(self):
        alg = PostprocessedAlgorithm(SVT, CombinedPP(LengthPP(), HammingDistancePP()))
        x = compose_postprocessing(np.random.default_rng(), [0, 1, 0, 1, 1, 0], 0.1, N=2, T=0.1, alg=alg,
                                   _d1=[1, 1, 1, 0, 0, 0])
        self.assertIsInstance(x, tuple)

    def test_identity_postprocessing(self):
        alg = PostprocessedAlgorithm(SVT, IdentityPP(6))
        x = compose_postprocessing(np.random.default_rng(), [0, 1, 0, 1, 1, 0], 0.1, N=2, T=0.1, alg=alg,
                                   _d1=[1, 1, 1, 0, 0, 0])
        self.assertIsInstance(x, tuple)
