import contextlib
from multiprocessing import Pool, current_process, get_context
from typing import Iterable, Optional
import numpy as np
import random
import math
import traceback

from dpsniper.utils.paths import set_output_directory
from dpsniper.utils.torch import torch_initialize

from dpsniper.utils.my_logging import log

try:
    import torch
except ImportError:
    torch = None


def exception_log_wrapper(f_args):
    try:
        f, args = f_args
        return f(args)
    except Exception as e:
        log.error(e)
        log.error(traceback.format_exc())
        raise


class MyParallelExecutor:
    def __init__(self):
        self._pool = None

    def initialize(self, n_processes, out_dir=None):
        """
        Initialize the executor with a given number of processes. This method must be called
        before any calls to execute. It should be called after the logger has been configured, because
        the process logs inherit the logging configuration from the parent.

        Args:
            n_processes: The number of processes to use for this executor.
            out_dir: The output directory to be used for all processes in this executor.
        """
        if self._pool is not None:
            raise Exception("MyParallelExecutor is already initialized")
        ctx = get_context('spawn') # to address https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork
        self._pool = ctx.Pool(n_processes, initializer=MyParallelExecutor._init_worker_process,
                          initargs=(log.get_config(), out_dir))

    def shutdown(self):
        """
        Shuts down the process pool associated with this executor. In order to perform any calls to
        execute later, need to call initialize again.

        It is recommended (but not required) to call this method at application exit.
        """
        if self._pool is None:
            raise Exception("MyParallelExecutor is already shut down")
        self._pool.close()
        self._pool.join()
        self._pool = None

    def execute(self, function, input_list: Iterable, chunk_size: int = 1) -> list:
        """
        Execute a function for many arguments using the parallel executor. Logging is configured
        to use different log files for each worker process.

        Args:
            function: A function with a single argument of type A, and returning a return value of type B
            input_list: The list of inputs of type A to be processed
            chunk_size: The number of inputs which should be processed together in one go.
                        If evaluating the function for a single input is very cheap, efficiency can
                        be improved by increasing the chunk_size.

        Returns:
            A list of return values of type B
        """
        if self._pool is None:
            raise Exception("MyParallelExecutor has not been initialized")

        def new_args():
            for i in input_list:
                yield (function, i)

        return self._pool.map(exception_log_wrapper, new_args(), chunksize=chunk_size)

    @staticmethod
    def _init_worker_process(parent_logging_config, out_dir: Optional[str]):
        """
        Makes sure each worker process uses a different log file (to overcome file synchronization issues)
        """
        id = current_process().name.split("-")[-1]
        log.configure_for_subprocess(parent_logging_config, id)

        # set the output directory
        if out_dir is not None:
            set_output_directory(out_dir)

        # freshly initialize the PRNG such that different processes use different randomness
        random.seed()
        np.random.seed()
        if torch is not None:
            torch.seed()
            torch.cuda.seed_all()
            torch_initialize()


# globally available executor
# needs to call initialize at application start
the_parallel_executor = MyParallelExecutor()


def split_by_batch_size(n, max_batch_size):
    if max_batch_size <= 0:
        raise ValueError('Invalid batch_size')
    n_tasks = math.ceil(n / max_batch_size)
    for i in range(0, n_tasks):
        if i == n_tasks - 1:
            # cover remaining samples
            remaining = n - i * max_batch_size
            assert remaining > 0
            yield remaining
        else:
            yield max_batch_size


def split_into_parts(n, n_parts):
    if n_parts <= 0:
        raise ValueError('Invalid n_parts')
    batch_size = n // n_parts
    for i in range(0, n_parts):
        if i == 0:
            first = n - (n_parts - 1) * batch_size
            assert first >= 0
            yield first
        else:
            yield batch_size


@contextlib.contextmanager
def initialize_parallel_executor(n_processes, out_dir=None):
    the_parallel_executor.initialize(n_processes, out_dir)
    yield
    the_parallel_executor.shutdown()
