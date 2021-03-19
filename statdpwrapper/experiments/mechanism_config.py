from statdp import ONE_DIFFER, ALL_DIFFER

from statdpwrapper.algorithms_ext import *
from statdpwrapper.algorithms import *
from statdpwrapper.postprocessing import PostprocessingConfig

statdp_mechanism_map = {}
statdp_arguments_map = {}
statdp_postprocessing_map = {}
statdp_num_inputs_map = {}
statdp_sensitivity_map = {}


def register(name, mechanism, pp_config: PostprocessingConfig, num_inputs: tuple, sensitivity, arguments: dict):
    statdp_mechanism_map[name] = mechanism
    statdp_arguments_map[name] = arguments
    statdp_postprocessing_map[name] = pp_config
    statdp_num_inputs_map[name] = num_inputs
    statdp_sensitivity_map[name] = sensitivity


register("LaplaceMechanism", laplace_mechanism, PostprocessingConfig(True, False, False, 1), (1, 1), ONE_DIFFER,
         {'epsilon': 0.1})
register("TruncatedGeometricMechanism", truncated_geometric, PostprocessingConfig(True, False, False, 1),
         (1, 1), ONE_DIFFER, {'epsilon': 0.1, 'n': 5})

register("NoisyHist1", histogram, PostprocessingConfig(True, False, False, 5), (5, 5), ONE_DIFFER,
         {'epsilon': 0.1})
register("NoisyHist2", histogram_eps, PostprocessingConfig(True, False, False, 5), (5, 5), ONE_DIFFER,
         {'epsilon': 0.1})

register("ReportNoisyMax1", noisy_max_v1a, PostprocessingConfig(True, False, False, 1), (5, 5), ALL_DIFFER,
         {'epsilon': 0.1})
register("ReportNoisyMax2", noisy_max_v2a, PostprocessingConfig(True, False, False, 1), (5, 5), ALL_DIFFER,
         {'epsilon': 0.1})
register("ReportNoisyMax3", noisy_max_v1b, PostprocessingConfig(True, False, False, 1), (5, 5), ALL_DIFFER,
         {'epsilon': 0.1})
register("ReportNoisyMax4", noisy_max_v2b, PostprocessingConfig(True, False, False, 1), (5, 5), ALL_DIFFER,
         {'epsilon': 0.1})

register("SparseVectorTechnique1", SVT, PostprocessingConfig(False, True, True, 10, categories=[True, False]),
         (10, 10), ALL_DIFFER, {'epsilon': 0.1, 'N': 1, 'T': 0.5})
register("SparseVectorTechnique2", SVT2, PostprocessingConfig(False, True, True, 10, categories=[True, False]),
         (10, 10), ALL_DIFFER, {'epsilon': 0.1, 'N': 1, 'T': 1})
register("SparseVectorTechnique3", iSVT4, PostprocessingConfig(True, True, True, 10, [False]),
         (10, 10), ALL_DIFFER, {'epsilon': 0.1, 'N': 1, 'T': 1})
register("SparseVectorTechnique4", iSVT3, PostprocessingConfig(False, True, True, 10, categories=[True, False]),
         (10, 10), ALL_DIFFER, {'epsilon': 0.1, 'N': 1, 'T': 1})
register("SparseVectorTechnique5", iSVT1, PostprocessingConfig(False, True, False, 10, categories=[True, False]),
         (10, 10), ALL_DIFFER, {'epsilon': 0.05, 'N': 1, 'T': 1})
register("SparseVectorTechnique6", iSVT2, PostprocessingConfig(False, True, False, 10, categories=[True, False]),
         (10, 10), ALL_DIFFER, {'epsilon': 0.1, 'N': 1, 'T': 1})

register("Rappor", rappor, PostprocessingConfig(False, True, False, 10, [1.0, 0.0]), (1, 1), ONE_DIFFER,
         {'epsilon': 0.0, 'n_hashes': 4, 'filter_size': 20, 'f': 0.75, 'p': 0.45, 'q': 0.55})
register("OneTimeRappor", one_time_rappor, PostprocessingConfig(False, True, False, 10, [1.0, 0.0]), (1, 1),
         ONE_DIFFER, {'epsilon': 0.0, 'n_hashes': 4, 'filter_size': 20, 'f': 0.95})

register("LaplaceParallel", laplace_parallel, PostprocessingConfig(True, False, False, 20), (1, 1), ONE_DIFFER,
         {'epsilon': 0.005, 'n_parallel': 20})

register("SVT34Parallel", svt_34_parallel,
         PostprocessingConfig(True, True, False, 20, [True, False, None]), (10, 10), ALL_DIFFER,
         {'epsilon': 0.1, 'N': 2, 'T': 1})
register("PrefixSum", prefix_sum, PostprocessingConfig(True, False, False, 10), (10, 10), ALL_DIFFER,
         {'epsilon': 0.1})
register("NumericalSVT", numerical_svt, PostprocessingConfig(True, False, True, 10), (10, 10), ALL_DIFFER,
         {'epsilon': 0.1, 'N': 2, 'T': 1})
