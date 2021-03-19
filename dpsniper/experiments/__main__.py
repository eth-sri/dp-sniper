import argparse
import multiprocessing

from dpsniper.search.ddsearch import DDConfig
from dpsniper.experiments.exp_default import run_exp_default
from dpsniper.experiments.exp_floating_point import *
from dpsniper.utils.torch import torch_initialize
from dpsniper.utils.my_logging import log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True, help='directory for output data')
    parser.add_argument('--reg', action="store_true",
                        help='run experiments on all algorithms using logistic regression classifier')
    parser.add_argument('--mlp', action="store_true",
                        help='run experiments on all algorithms using neural network classifier')
    parser.add_argument('--floating', action="store_true",
                        help='run floating point vulnerability experiment')
    parser.add_argument('--floating-fixed', action="store_true",
                        help='run floating point vulnerability experiment with snapping mechanism fix')
    parser.add_argument('--processes', type=int, help='number of processes to use', default=multiprocessing.cpu_count())
    parser.add_argument('--torch-device', default='cpu', help='the pytorch device use (untested on GPU!)')
    parser.add_argument('--torch-threads', default=8, help='the number of threads for pytorch to use')
    args = parser.parse_args()

    if (not args.reg) and (not args.mlp) and (not args.floating) and (not args.floating_fixed):
        print("ERROR: At least one of --reg, --mlp, --floating, --floating-fixed must be set")
        exit(1)

    n_processes = args.processes
    torch_initialize(args.torch_threads, args.torch_device)
    log.configure("WARNING")

    if args.reg:
        run_exp_default("dd_search_reg", False, args.output_dir, DDConfig(n_processes=n_processes))

    if args.mlp:
        run_exp_default("dd_search_mlp", True, args.output_dir, DDConfig(n_processes=n_processes))

    if args.floating:
        run_exp_floating_point(args.output_dir, DDConfig(n_processes=n_processes))

    if args.floating_fixed:
        run_exp_floating_fixed(args.output_dir, DDConfig(n_processes=n_processes))
