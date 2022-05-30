import logging
from typing import List
from dataclasses import dataclass, field


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

    n_trials: int = field(
        default=1, metadata={'help': 'Number of trials to run'}
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
    auto_logging_checkpoint_dirs: bool = field(
        default=False, 
        metadata={'help': 'Automatically generate logging/checkpoint directories based on name'}
    )
    
    sbatch_args: List[str] = field(
        default_factory=list, 
        metadata={'help': 'List of sbatch arguments in format of [arg1, arg2=value2, ...]. \
            If logging_dir is provided, do not provide job name and output/error filename \
            as they are specified by the experiment object. Otherwise logs will be written to \
            where the batch script was run.'
        }
    )

    def __post_init__(self):
        if not (self.n_trials > 0):
            raise ValueError('Number of trials must be greater than 0')
            
        if self.auto_logging_checkpoint_dirs:
            if self.logging_dir or self.checkpoints_dir:
                logging.warning('auto logging/checkpoint directories will overwrite provided logging_dir/checkpoints_dir')
            self.logging_dir = 'logs/' + self.name
            self.checkpoints_dir = 'checkpoints/' + self.name
        
            
