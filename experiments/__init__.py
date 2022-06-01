import logging

from experiments.experiment import Experiment
from experiments.experiment_args import ExperimentArguments

from experiments.logging_utils import default_make_log_filename


# setup execution logging for any module that imports experiment
logging_filename = default_make_log_filename('logs/exec')

logging.basicConfig(filename=logging_filename, level=logging.INFO)