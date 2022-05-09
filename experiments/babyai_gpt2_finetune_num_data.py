import os

from experiment_utils import Experiment, ExperimentArgs


EXPERIMENT_NAME = 'babyai_gpt2_finetune_num_data'

experiment_args = ExperimentArgs()

experiment = Experiment(
    experiment_name=EXPERIMENT_NAME,
    dataset_name='babya/env_description_chunked',
    slurm_spec='task_sequence_gpt2_finetune.sh',
    args=experiment_args
)

NUM_DATA = [100, 500, 1000, 2500, 5000, 10000, 50000, 100000]

for n in NUM_DATA:
    run_name = f'{EXPERIMENT_NAME}_{n}'
    run_params = {
        'num_data': n,
        'data_dir': experiment.data_dir,
        'logging_dir': experiment.logging_dir,
        'checkpoint_dir': experiment.checkpoints_dir,
    }
    experiment.run(run_name, run_params)


