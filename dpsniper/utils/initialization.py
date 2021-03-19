import contextlib
import os
from typing import Optional

from dpsniper.search.ddconfig import DDConfig
from dpsniper.utils.my_logging import log
from dpsniper.utils.my_multiprocessing import initialize_parallel_executor
from dpsniper.utils.paths import set_output_directory, get_output_directory
from dpsniper.utils.torch import torch_initialize


@contextlib.contextmanager
def initialize_dpsniper(config: DDConfig, out_dir: Optional[str] = None, torch_threads: Optional[int] = None):
    """
    Helper context manager to initialize DP-Sniper.

    Args:
        config: configuration
        out_dir: directory for temporary files, logs and torch models (temporary directory is created if not specified)
        torch_threads: number of threads to be used by pytorch during training (default torch configuration is used if not specified)
    """
    set_output_directory(out_dir)
    torch_initialize(torch_threads=torch_threads)
    log.configure("WARNING", log_file=os.path.join(get_output_directory("logs"), "dpsniper.log"), file_level="DEBUG")
    with initialize_parallel_executor(config.n_processes, out_dir=out_dir):
        yield
