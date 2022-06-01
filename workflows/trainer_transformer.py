import os
import json
from typing import List, Dict, Iterable

from transformers import Trainer
from transformers import TrainerCallback
from transformers import TrainingArguments

from workflows.trainer_utils import dict_to_serializable


def get_entry_name(metrics: Dict) -> str:
    ''' Return the entry name for metrics '''
    if 'eval_loss' in metrics:
        return 'eval'
     ## TODO: a predict case prob needs to be added
    return 'train'


def remove_prefix(metric_name: str) -> str:
    ''' Remove prefix from metric name '''
    components = metric_name.split('_')
    if len(components) > 1:
        components = components[1:]
    return '_'.join(components)


def remove_metrics_prefix(metrics: Dict) -> Dict:
    ''' Remove prefix from all metric name keys '''
    return {remove_prefix(k): v for k, v in metrics.items()}


def filter_metrics_by_keys(metrics: Dict, metrics_to_log: Iterable) -> Dict:
    ''' Return metrics filtered by metric names present in metrics_to_log '''
    if not isinstance(metrics_to_log, set):
        metrics_to_log = set(metrics_to_log)
    return {k: v for k, v in metrics.items() if k in metrics_to_log}


class LoggingCallback(TrainerCallback):
    def __init__(
        self, 
        logging_dir: str, 
        entries_to_log: List[str] = None,
        metrics_to_log: List[str] = None
    ):
        super().__init__()
        self.logging_dir = logging_dir
        self.entries_to_log = entries_to_log
        self.metrics_to_log = metrics_to_log
        
        os.makedirs(logging_dir, exist_ok=True)

    def on_log(self, args, state, control, logs=None, **kwargs):
        ''' Replicates logging of native trainer. Filter which entries are logged by 
            specifying entries_to_log and metrics by metrics_to_log 
        '''
        epoch = int(logs.pop('epoch')) # take floor of fractional epoch to get current epoch
        entry_name = get_entry_name(logs)
        if not entry_name in self.entries_to_log:
            return

        metrics = remove_metrics_prefix(logs)
        if self.metrics_to_log:
            metrics = filter_metrics_by_keys(metrics, self.metrics_to_log)
        logging_filename = f'{self.logging_dir}/{entry_name}_{epoch:03d}.json'
        with open(logging_filename, 'w') as f:
            json.dump(dict_to_serializable(metrics), f)