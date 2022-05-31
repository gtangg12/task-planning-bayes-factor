import os
import time
import logging

exec_logging_dir = 'logs/exec'
current_time_str = time.strftime("%Y%m%d-%H%M%S")

os.makedirs(exec_logging_dir, exist_ok=True)
logging.basicConfig(filename=f'{exec_logging_dir}/{current_time_str}.log', level=logging.INFO)


from experiments.experiment import Experiment
from experiments.experiment_args import ExperimentArguments
