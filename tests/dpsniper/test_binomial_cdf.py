import unittest
from dpsniper.probability.binomial_cdf import lcb, ucb
from scipy.stats import binom

from tests.my_test_case import MyTestCase


class TestBinomialCDF(MyTestCase):

    def test_lcb(self):
        # requiring full confidence => trivial lower bound
        self.assertEqual(lcb(5, 2, 0), 0)
        # requiting no confidence => strongest possible lower bound
        self.assertEqual(lcb(5, 2, 1), 1)
        # only saw 0 => trivial lower bound
        self.assertEqual(lcb(5, 0, 0.5), 0)
        # no data => trivial lower bound
        self.assertEqual(lcb(0, 0, 0.5), 0)

        for n in [10, 50]:
            for k in [1, n // 2, n-1]:
                for alpha in [0.1, 0.5]:
                    with self.subTest(msg='test_lcb', params={'n': n, 'k': k, 'alpha': alpha}):
                        p = lcb(n, k, alpha)
                        # Pr[Binom[n,p]>=k] = Pr[Binom[n,1-p]<=n-k]
                        alpha_actual = binom.cdf(n-k, n, 1-p)

                        self.assertAlmostEqual(alpha, alpha_actual)

    def test_ucb(self):
        # requiring full confidence => trivial upper bound
        self.assertEqual(ucb(5, 2, 0), 1)
        # requiting no confidence => strongest possible upper bound
        self.assertEqual(ucb(5, 2, 1), 0)
        # only saw 1 => trivial upper bound
        self.assertEqual(ucb(5, 5, 0.5), 1)
        # no data => trivial upper bound
        self.assertEqual(ucb(0, 0, 0.5), 1)

        for n in [10, 50]:
            for k in [1, n // 2, n-1]:
                for alpha in [0.1, 0.5]:
                    with self.subTest(msg='test_ucb', params={'n': n, 'k': k, 'alpha': alpha}):
                        p = ucb(n, k, alpha)
                        alpha_actual = binom.cdf(k, n, p)

                        self.assertAlmostEqual(alpha, alpha_actual)
