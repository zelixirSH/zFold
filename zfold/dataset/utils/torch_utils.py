"""PyTorch-related utility functions."""

import torch


def get_tensor_size(tensor):
    """Get the PyTorch tensor's memory consumption (in MB).

    Args:
    * tensor: PyTorch tensor

    Returns:
    * mem: PyTorch tensor's memory consumption (in MB)
    """

    mem = tensor.element_size() * torch.numel(tensor) / 1024.0 / 1024.0

    return mem


def get_peak_memory():
    """Get the peak memory consumption (in GB).

    Args: n/a

    Returns:
    * mem: peak memory consumption (in GB)
    """

    mem = torch.cuda.memory_stats()['allocated_bytes.all.peak'] / 1024.0 / 1024.0

    return mem


def send_to_device(data_dict, device):
    """Send the data dict to the specified device.

    Args:
    * data_dict: dict of PyTorch tensors (other elements are allowed but will be not sent)
    * device: PyTorch device, e.g. torch.device('cpu') OR torch.device('cuda:0')

    Returns:
    * data_dict: dict of PyTorch tensors located at the specified device
    """

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
