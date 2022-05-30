from typing import Dict
import torch


NUM_DEVICES_AVAILABLE = torch.cuda.device_count()
DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def dict_to_device(inputs: Dict, device: torch.device) -> None:
    ''' Move all tensors in input to device '''
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    return inputs


def dict_to_serializable(inputs):
    ''' Convert dict to a serializable format '''
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.tolist()
    return inputs