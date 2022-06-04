import warnings
from typing import List
from dataclasses import dataclass, field

from experiments.slurm_config import SLURM_MAX_CONCURRENT_RUNNING_JOBS


@dataclass
class ExperimentArguments:
    name: str = field(
        metadata={'help': 'Name of experiment'}
    )

    script: str = field(
        metadata={'help': 'Path to script to run'}
    )
    conda_env: str = field(
        metadata={'help': 'Conda environment to use'}
    )

    num_trials: int = field(
        default=1, metadata={'help': 'Number of trials to run'}
    )
    max_concurrent_running_trials: int = field(
        default=None,
        metadata={
            'help': (
                'Maximum number of jobs to run concurrently from this experiment instance.'
                'It\'s possible that there may not be enough resources to run all max_concurrent_runs jobs,'
                'in which case the remaining (max_concurrent_runs - number of launched runs) will be queued.'
            )
        }
    )

    data_dir: str = field(
        default=None, metadata={'help': 'Path to dataset directory'}
    )
    logging_dir: str = field(
        default=None, metadata={'help': 'Path to logging directory'}
    )
    checkpoints_dir: str = field(
        default=None, metadata={'help': 'Path to checkpoints directory'}
    )
    auto_make_logging_checkpoint_dirs: bool = field(
        default=False, metadata={'help': 'Automatically generate logging/checkpoint directories based on name'}
    )

    def __post_init__(self):
        if self.num_trials <= 0:
            raise ValueError('Number of trials must be greater than 0')
        
        if self.max_concurrent_running_trials is not None:
            if not (0 < self.max_concurrent_running_trials <= SLURM_MAX_CONCURRENT_RUNNING_JOBS):
                raise ValueError(f'max_concurrent_running_trials must be an integer \
                    between 1 and {SLURM_MAX_CONCURRENT_RUNNING_JOBS}, inclusive')
        
        if self.auto_make_logging_checkpoint_dirs:
            if self.logging_dir or self.checkpoints_dir:
                warnings.warn('Auto logging/checkpoint directories will overwrite provided logging_dir/checkpoints_dir')
            self.logging_dir = 'logs/' + self.name
            self.checkpoints_dir = 'checkpoints/' + self.name    
