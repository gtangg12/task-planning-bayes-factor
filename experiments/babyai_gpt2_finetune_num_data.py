import os

from experiment_utils import Experiment, ExperimentArgs


EXPERIMENT_NAME = 'babyai_gpt2_finetune_num_data'

experiment_args = ExperimentArgs()

experiment = Experiment(
    experiment_name=EXPERIMENT_NAME,
    dataset='babyai/env_description_chunked',
    slurm_spec='task_sequence_gpt2_finetune.sh',
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

