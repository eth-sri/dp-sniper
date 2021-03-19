import torch


def torch_initialize(torch_threads=None, torch_device=None, default_dtype=torch.float64):
    # torch configuration
    if torch_threads is not None:
        torch.set_num_threads(torch_threads)  # have to call this before doing anything else with pytorch
    if torch_device is not None:
        torch.device(torch_device)  # device to use (CPU or GPU)
    if default_dtype is not None:
        torch.set_default_dtype(default_dtype)  # otherwise, torch.Tensor converts 64bit to 32bit floats