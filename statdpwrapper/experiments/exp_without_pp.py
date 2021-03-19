import os

from dpsniper.utils.my_multiprocessing import initialize_parallel_executor
from dpsniper.utils.paths import get_output_directory, set_output_directory
from statdpwrapper.algorithms_ext import *

from statdpwrapper.experiments.base import run_statdp
from statdpwrapper.experiments.mechanism_config import statdp_mechanism_map, statdp_arguments_map,\
    statdp_postprocessing_map, statdp_sensitivity_map, statdp_num_inputs_map


def _get_mechanism(alg_name: str):
    if alg_name not in statdp_mechanism_map:
        raise ValueError("Unknown mechanism {}".format(alg_name))
    return statdp_mechanism_map[alg_name]


def run_without_postprocessing(n_processes: int, out_dir: str, alg_name: str):
    mechanism = _get_mechanism(alg_name)
    kwargs = statdp_arguments_map[alg_name]
    pp_config = statdp_postprocessing_map[alg_name]
    num_inputs = statdp_num_inputs_map[alg_name]
    sensitivity = statdp_sensitivity_map[alg_name]

    log.configure("WARNING")
    set_output_directory(out_dir)
    logs_dir = get_output_directory("logs")
    log_file = os.path.join(logs_dir, "original_statdp_{}_log.log".format(alg_name))
    data_file = os.path.join(logs_dir, "original_statdp_{}_data.log".format(alg_name))

    if os.path.exists(log_file):
        log.warning("removing existing log file '%s'", log_file)
        os.remove(log_file)
    if os.path.exists(data_file):
        log.warning("removing existing log file '%s'", data_file)
        os.remove(data_file)

    log.configure("INFO", log_file=log_file, data_file=data_file, file_level="INFO")
    with initialize_parallel_executor(n_processes, out_dir):
        # run StatDP with disabled postprocessing
        pp_config.disable_pp = True
        run_statdp(alg_name, mechanism, pp_config, num_inputs, sensitivity, kwargs)
    log.info("finished experiment")
