import os

from experiment import Experiment
from experiment_args import ExperimentArgs


EXPERIMENT_NAME = 'babyai_classifier_train_num_data'

experiment_args = ExperimentArgs()

experiment = Experiment(
    experiment_name=EXPERIMENT_NAME,
    dataset='babyai/task_sequence_chunked',
    slurm_spec='task_sequence_classifier_train.sh',
    args=experiment_args
)

NUM_DATA = [100, 500, 1000, 2500, 5000, 10000, 50000, 100000]

for n in NUM_DATA:
    run_name = f'n{n}'
    experiment.setup_run(run_name)
    run_params = {
        'num_data': n,
        'data_dir': experiment.data_dir,
        'checkpoints_dir': experiment.checkpoints_dir,
        'logging_dir': os.path.join(experiment.logging_dir, 'tensorboard'),
    }
    experiment.run(run_params)
    exit()