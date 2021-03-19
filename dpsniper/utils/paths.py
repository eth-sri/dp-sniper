import os
import tempfile

global_output_dir = None


def set_output_directory(out_dir: str):
    """
    Sets the output directory to be used by DP-Sniper for logs, temporary files, plots, etc.
    If this function is not called, a temporary folder in the current working directory is used.
    """
    global global_output_dir
    global_output_dir = out_dir


def get_output_directory(*subdirs: str):
    global global_output_dir
    if global_output_dir is None:
        # output directory not set, use a temporary output directory in current working directory
        global_output_dir = tempfile.mkdtemp(dir=os.getcwd())

    target = global_output_dir
    for sub in subdirs:
        target = os.path.join(target, sub)
    if not os.path.exists(target):
        os.makedirs(target, exist_ok=True)
    return target
