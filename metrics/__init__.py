from importlib.resources import path
from typing import Callable, Union, List

import torch

from metrics.classification_metrics import (
    accuracy,
    label_frequency,
    kl_divergence,
    kl_divergence_symmetric,
)


Logits, Labels = torch.Tensor, torch.Tensor


CLASSIFICATION_METRICS = {
    'accuracy': accuracy,
    'label_frequency': label_frequency,
    'kl_divergence': kl_divergence,
    'kl_divergence_symmetric': kl_divergence_symmetric,
}


METRIC_LIBRARIES = {
    'classification': CLASSIFICATION_METRICS,
}


def load_metric(path: Union[str, List[str]]) -> Callable:
    """ Loads a metric specified by metric path. When path is a list, 
        return list of metrics in same order. 
    """
    if isinstance(path, List):
        return [load_metric(metric_name) for metric_name in path]
    path_components = path.split('-')
    metric = METRIC_LIBRARIES
    for component in path_components:
        if component not in metric:
            raise ValueError(f'Metric {path} does not exist')
        metric = metric[component]
    return metric
