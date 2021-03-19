import unittest

from statdpwrapper.algorithms import *
from statdpwrapper.algorithms_ext import *
from statdpwrapper.postprocessing import the_zero_noise_prng, PostprocessedAlgorithm, LengthPP, compose_postprocessing

from statdpwrapper.my_generate_counterexample import detect_counterexample
from tests.my_test_case import MyTestCase


class TestStatDPAlgorithms(MyTestCase):
    @staticmethod
    def t_input_output_format(tester, algorithm, queries, **kwargs):
        res = algorithm(np.random.default_rng(), queries, **kwargs)
        tester.assertIsInstance(res, list)

    @staticmethod
    def t_zero_noise(tester, algorithm, queries, **kwargs):
        res = algorithm(the_zero_noise_prng, queries, **kwargs)
        tester.assertIsInstance(res, list)

    @staticmethod
    def t_accepted_by_statdp(tester, algorithm, queries, **kwargs):
        # just make sure StatDP does not crash with the mechanism
        pp = LengthPP()     # use a postprocessing applicable to all algorithms
        pp_alg = PostprocessedAlgorithm(algorithm, pp)
        kwargs['alg'] = pp_alg
        detect_counterexample(compose_postprocessing, 0.1, num_input=(1, 1),
                              default_kwargs=kwargs, event_iterations=10, detect_iterations=10,
                              quiet=True, loglevel="ERROR")
        del kwargs['alg']

    def t_apply(self, t_method, skip_geom=False):
        t_method(self, noisy_max_v1a, [2.3, 1.4, 4.0], epsilon=0.1)
        t_method(self, noisy_max_v1b, [2.3, 1.4, 4.0], epsilon=0.1)
        t_method(self, noisy_max_v2a, [2.3, 1.4, 4.0], epsilon=0.1)
        t_method(self, noisy_max_v2b, [2.3, 1.4, 4.0], epsilon=0.1)
        t_method(self, histogram_eps, [2.3, 1.4, 4.0], epsilon=0.1)
        t_method(self, histogram, [2.3, 1.4, 4.0], epsilon=0.1)
        t_method(self, SVT, [2.3, 1.4, 4.0], epsilon=0.1, N=2, T=1)
        t_method(self, iSVT1, [2.3, 1.4, 4.0], epsilon=0.1, N=2, T=1)
        t_method(self, iSVT2, [2.3, 1.4, 4.0], epsilon=0.1, N=2, T=1)
        t_method(self, iSVT3, [2.3, 1.4, 4.0], epsilon=0.1, N=2, T=1)
        t_method(self, iSVT4, [2.3, 1.4, 4.0], epsilon=0.1, N=2, T=1)
        t_method(self, laplace_mechanism, [2.3], epsilon=0.1)
        t_method(self, laplace_parallel, [2.3], epsilon=0.1, n_parallel=10)
        t_method(self, svt_34_parallel, [1.4, 5.3, 2.8], epsilon=0.1, N=2, T=1)
        t_method(self, prefix_sum, [3.2, 4.1, 4.5], epsilon=0.1)
        t_method(self, numerical_svt,  [1.4, 5.3, 2.8], epsilon=0.1, N=2, T=1)
        t_method(self, SVT2, [2.3, 1.4, 4.0], epsilon=0.1, N=2, T=1)
        t_method(self, rappor, [4.5], epsilon=0.1, n_hashes=4, filter_size=10, f=0.5, p=0.75, q=0.5)
        t_method(self, one_time_rappor, [4.5], epsilon=0.1, n_hashes=4, filter_size=10, f=0.5)
        if not skip_geom:
            t_method(self, truncated_geometric, [2], epsilon=0.1, n=4)

    def test_all_formats(self):
        self.t_apply(TestStatDPAlgorithms.t_input_output_format)

    def test_zero_prng(self):
        self.t_apply(TestStatDPAlgorithms.t_zero_noise, skip_geom=True)

    def test_geometric_rejects_zero_prng(self):
        with log.temporarily_disable():  # disable logging to avoid printing an expected but confusing error message
            self.assertRaises(NotImplementedError, truncated_geometric, the_zero_noise_prng, [5], 0.1, 4)

    def test_accepted_by_statdp(self):
        self.t_apply(TestStatDPAlgorithms.t_accepted_by_statdp)
