import logging
from typing import List
from dataclasses import dataclass, field


logger = logging.getLogger(__name__)


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

    def __post_init__(self):
        if not (self.num_trials > 0):
            raise ValueError('Number of trials must be greater than 0')
        
        if self.auto_logging_checkpoint_dirs:
            if self.logging_dir or self.checkpoints_dir:
                logger.warning('auto logging/checkpoint directories will overwrite provided logging_dir/checkpoints_dir')
            self.logging_dir = 'logs/' + self.name
            self.checkpoints_dir = 'checkpoints/' + self.name

        elif not self.logging_dir:
            raise ValueError('Must provide logging_dir if auto_logging_checkpoint_dirs is False')
            
