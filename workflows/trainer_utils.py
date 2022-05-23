from typing import Dict
import torch


NUM_DEVICES_AVAILABLE = torch.cuda.device_count()
DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def inputs_to_device(inputs: Dict, device: torch.device) -> None:
    ''' Move all tensors in input to device '''
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
