from typing import Callable

from metrics.classification_metrics import (
    accuracy,
    label_frequency,
    label_frequency_norm,
)


CLASSIFICATION_METRICS = {
    'accuracy': accuracy,
    'label_frequency': label_frequency,
    'label_frequency_norm': label_frequency_norm,
}


METRIC_LIBRARIES = {
    'classification': CLASSIFICATION_METRICS,
}


def load_metric(library: str, metric: str) -> Callable:
    """ Loads a metric from the respective metric library """
    assert library in METRIC_LIBRARIES, \
        f'{library} is not a valid metric library.'
    assert metric in METRIC_LIBRARIES[library], \
        f'{metric} is not a valid metric in {library}.'

    return METRIC_LIBRARIES[library][metric]

