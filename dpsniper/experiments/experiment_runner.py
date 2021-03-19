from abc import ABC, abstractmethod

import os
import glob
from typing import List

from dpsniper.utils.my_logging import log, log_context
from dpsniper.utils.my_multiprocessing import initialize_parallel_executor, the_parallel_executor
from dpsniper.utils.paths import set_output_directory, get_output_directory


class BaseExperiment(ABC):
    """
    Base class for an experiment.
    """

    def __init__(self, experiment_name: str):
        """
        Creates a new experiment with the given name.

        Args:
            experiment_name: a name for the experiment used to identify it in the output logs
        """
        self.experiment_name = experiment_name

    @abstractmethod
    def run(self):
        """
        Runs the experiment.
        """
        pass


class ExperimentRunner:
    """
    A runner for running a series of experiments in parallel. Once the runner is constructed,
    append implementations of BaseExperiment to the experiments list.
    """

    def __init__(self, output_dir: str, series_name: str, series_comment: str = "", log_debug=False):
        """
        Create a new series of experiments. You can add experiments to the field 'experiments'
        of the constructed ExperimentRunner.

        Args:
            output_dir: The output folder to be used for the experiment.
            series_name: A prefix for this experiment series. The output files will be named
                            {series_name}_log.log  and
                            {series_name}_data.log
            series_comment: A comment describing the experiment series
        """
        self.output_dir = output_dir
        self.series_name = series_name
        self.series_comment = series_comment
        self.experiments: List[BaseExperiment] = []
        self.file_level = "INFO"
        if log_debug:
            self.file_level = "DEBUG"

    def run_all(self, n_processes: int, sequential=True):
        """
        Runs all experiments in the experiments list. Because this function initializes the global parallel executor,
        the latter must be uninitialized.

        Args:
            n_processes: Number of processes used for the global parallel executor.
            sequential: Whether the experiments should be ran sequentially (True) or should be parallelized
                    at the highest level using the global executor (False). If this flag is False, experiments
                    _must not_ use the parallel executor themselves (risk of deadlock!).
        """
        set_output_directory(self.output_dir)
        logs_dir = get_output_directory("logs")

        for fname in glob.glob(os.path.join(logs_dir, self.series_name + "_*.log")):
            log.warning("removing existing log file '%s'", fname)
            os.remove(fname)

        log_file = os.path.join(logs_dir, "{}_log.log".format(self.series_name))
        data_file = os.path.join(logs_dir, "{}_data.log".format(self.series_name))
        log.configure("ERROR", log_file=log_file, file_level=self.file_level, data_file=data_file)

        with initialize_parallel_executor(n_processes, self.output_dir):
            if sequential:
                for exp in self.experiments:
                    ExperimentRunner._run_one_experiment((exp, self.series_comment))
            else:
                inputs = []
                for exp in self.experiments:
                    inputs.append((exp, self.series_comment))
                the_parallel_executor.execute(ExperimentRunner._run_one_experiment, inputs)

        log.info("finished experiments!")

    @staticmethod
    def _run_one_experiment(args):
        exp, comment = args
        log.info("running experiment %s", exp.experiment_name)
        log.info("comment: %s", comment)
        with log_context(exp.experiment_name):
            try:
                exp.run()
            except Exception:
                log.error("Exception while running experiment %s", exp.experiment_name, exc_info=True)
        log.info("finished experiment %s", exp.experiment_name)
