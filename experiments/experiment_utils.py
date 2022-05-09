import os
from os.path import join
from dataclasses import dataclass
from typing import Dict


# absolute directory of work ecosystem
BASE_DIR = '/nobackup/users/gtangg12/task_planning_bayes_factor'

# respective absolute directories for each module
DATA_DIR = join(BASE_DIR, 'data')
SLURM_DIR = join(BASE_DIR, 'slurm')
CHECKPOINTS_DIR = join(BASE_DIR, 'checkpoints')
EXPERIMENTS_DIR = join(BASE_DIR, 'experiments')


def wrap_quotes(str: str):
    return f'"{str}"'


def make_args(args: Dict) -> str:
    """ Helper function to make a string of command line arguments from args dict """
    cmd = []
    for key, value in args.items():
        cmd.append(f'--{key} {value}')
    # wrap in quotes for bash script to pass args as string to python script
    return wrap_quotes(' '.join(cmd))


@dataclass
class ExperimentArgs:
    pass
    # TODO gpus cpus etc pipelines


class Experiment:
    def __init__(self, experiment_name: str, dataset: str, slurm_spec: str, args: ExperimentArgs) -> None:
        self.experiment_name = experiment_name
        self.args = args

        self.data_dir = join(DATA_DIR, dataset)
        self.slurm_spec_path = join(SLURM_DIR, slurm_spec)
        self.logging_base_dir = join(EXPERIMENTS_DIR, experiment_name)
        self.checkpoints_base_dir = join(CHECKPOINTS_DIR, experiment_name)

        # Flag for whether experiment has been setup
        self.setup = False

    def setup_run(self, run_name: str) -> None:
        """ Update internal experiment state, including setting up logging and 
            checkpoint directories 
        """
        self.run_name = run_name
        self.logging_dir = join(self.logging_base_dir, run_name)
        self.checkpoints_dir = join(self.checkpoints_base_dir, run_name)
        self.setup = True

    def run(self, params: Dict) -> None:
        """ Run experiment with given parameters """
        assert self.setup, 'Must setup run first.'

        # slurm cannot write to nonexistent directory
        os.makedirs(self.logging_dir, exist_ok=True)

        # build and run sbatch command
        sbatch_cmd = self._make_sbatch()
        params_cmd = make_args(params)
        cmd = f'sbatch {sbatch_cmd} {params_cmd}'
        os.system(cmd)
        
        # clear experiment state to avoid overlap run states
        self._clear()

    def _make_sbatch(self) -> str:
        """ Make sbatch command for slurm based on slurm arguments provided in args """
        sbatch_args = [
            f'-J {self.run_name}',
            f'-o {self.logging_dir}/run.out',
            f'-e {self.logging_dir}/run.err',
            self.slurm_spec_path,
        ]
        for key, value in self.args.__dict__.items():
            sbatch_args.append(f'--{key}={value}')
        return ' '.join(sbatch_args)

    def _clear(self) -> None:
        """ Clear experiment state to prevent run from being reran or variables reused """
        self.setup = False
        self.logging_dir = None
        self.checkpoints_dir = None
