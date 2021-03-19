import argparse
import multiprocessing

from statdpwrapper.experiments.exp_with_pp import run_with_postprocessing
from statdpwrapper.experiments.exp_without_pp import run_without_postprocessing

"""
Overview of algorithm implementations

Ours                    StatDP

LaplaceMechanism        algorithms_ext.laplace_mechanism
TruncatedGeometric      algorithms_ext.truncated_geometric
NoisyHist1              algorithms.histogram
NoisyHist2              algorithms.histogram_eps
ReportNoisyMax1         algorithms.noisy_max_v1a               
ReportNoisyMax2         algorithms.noisy_max_v2a
ReportNoisyMax3         algorithms.noisy_max_v1b
ReportNoisyMax4         algorithms.noisy_max_v2b
SparseVectorTechnique1  algorithms.SVT
SparseVectorTechnique2  algorithms_ext.SVT2
SparseVectorTechnique3  algorithms.iSVT4
SparseVectorTechnique4  algorithms.iSVT3
SparseVectorTechnique5  algorithms.iSVT1 (eps = our_eps/2)
SparseVectorTechnique6  algorithms.iSVT2
Rappor                  algorithms_ext.rappor
OneTimeRappor           algorithms_ext.one_time_rappor
LaplaceParallel         algorithms_ext.laplace_parallel
SVT34Parallel           algorithms_ext.svt_34_parallel
PrefixSum               algorithms_ext.prefix_sum
NumericalSVT            algorithms_ext.numerical_svt
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processes', type=int, help='number of processes to use', default=multiprocessing.cpu_count())
    parser.add_argument('--output-dir', required=True, help='Directory for output data')
    parser.add_argument('--no-postprocessing', help='deactivates postprocessing (equivalent to original StatDP, requires --alg)', action="store_true")
    parser.add_argument('--alg', type=str, help='algorithm to evaluate (all if not specified)')
    parser.add_argument('--postfix', type=str, help='postfix for logs', default="")
    args = parser.parse_args()
    n_processes = args.processes

    if args.no_postprocessing:
        if not args.alg:
            print("--no_postprocessing requires --alg flag")
            exit(1)
        else:
            run_without_postprocessing(n_processes, alg_name=args.alg, out_dir=args.output_dir)
    else:
        run_with_postprocessing(n_processes, only_mechanism=args.alg, postfix=args.postfix, out_dir=args.output_dir)
