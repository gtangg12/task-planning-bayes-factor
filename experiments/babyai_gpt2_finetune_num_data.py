import os

from experiment_utils import (
    make_args,
    make_sbatch,
    LOGGING_DIR,
)


EXPERIMENT_NAME = 'babyai_gpt2_finetune'

BASE_DIR = '/nobackup/users/gtangg12/task_planning_bayes_factor'
DATA_DIR = f'{BASE_DIR}/data/babyai/env_description_chunked'
CHECKPOINTS_DIR = f'{BASE_DIR}/checkpoints'
SLURM_SPEC = f'{BASE_DIR}/slurm/task_sequence_gpt2_finetune.sh'


NUM_DATA = [100, 500, 1000, 2500, 5000, 10000, 50000, 100000]

for n in NUM_DATA:
    run_name = f'{EXPERIMENT_NAME}_{n}'
    sbatch_cmd = make_sbatch(f'{LOGGING_DIR}/{run_name}', SLURM_SPEC)
    experiment_params = {
        'num_data': n,
        'data_dir': DATA_DIR,
        'output_dir': f'{CHECKPOINTS_DIR}/{run_name}',
    }
    experiment_params = make_args(experiment_params)
    cmd = f'{sbatch_cmd} {experiment_params}'
    os.system(cmd)
    exit()


