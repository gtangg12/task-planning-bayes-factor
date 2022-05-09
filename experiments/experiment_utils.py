import os
from typing import Dict


# absolute directory of work ecosystem
BASE_DIR = '/nobackup/users/gtangg12/task_planning_bayes_factor'

# respective absolute directories for each module
DATA_DIR = os.path.join(BASE_DIR, 'data')
SLURM_DIR = os.path.join(BASE_DIR, 'slurm')
CHECKPOINTS_DIR = os.path.join(BASE_DIR, 'checkpoints')
EXPERIMENTS_DIR = os.path.join(BASE_DIR, 'experiments')


def wrap_quotes(str: str):
    return f'"{str}"'


def make_args(args: Dict) -> str:
    cmd = []
    for key, value in args.items():
        cmd.append(f'--{key} {value}')
    # wrap in quotes for bash script to pass args as string to python script
    return wrap_quotes(' '.join(cmd))


def make_sbatch(run_name: str, logging_dir: str, slurm_path: str) -> str:
    cmd = [
        'sbatch',
        f'-J {run_name}',
        f'-o {logging_dir}/{run_name}.out',
        f'-e {logging_dir}/{run_name}.err',
        slurm_path,
    ]
    return ' '.join(cmd)


class ExperimentArgs:
    pass
    # TODO gpus cpus etc pipelines


class Experiment:
    def __init__(
        self, 
        experiment_name: str, 
        dataset_name: str, 
        slurm_spec: str, 
        args: ExperimentArgs
    ) -> None:
        self.experiment_name = experiment_name
        self.args = args
        os.makedirs(experiment_name, exist_ok=True)

        self.data_dir = os.path.join(DATA_DIR, dataset_name)
        self.logging_dir = os.path.join(EXPERIMENTS_DIR, experiment_name)
        self.checkpoints_dir = os.path.join(CHECKPOINTS_DIR, experiment_name)
        self.slurm_spec_path = os.path.join(SLURM_DIR, slurm_spec)

    def run(self, run_name, params):
        sbatch_cmd = make_sbatch(run_name, self.logging_dir, self.slurm_spec_path)
        params_cmd = make_args(params)
        cmd = f'{sbatch_cmd} {params_cmd}'
        os.system(cmd)