import os
import time
import copy
import logging
from typing import Dict, List

from experiments.experiment_args import ExperimentArguments
from experiments.experiment_utils import (
    list_of_dicts,
    check_keys_consistent,
    default_make_run_name,
    make_args_command_sbatch,
    make_args_command
)


SBATCH_TEMPLATE_PATH = os.path.dirname(__file__) + '/template.sh'
SBATCH_LOGGING_PREFIXES = ['job_name==', 'output==', 'error==']


logger = logging.getLogger(__name__)


class Experiment:
    def __init__(
        self, 
        args: ExperimentArguments, 
        sbatch_args: List[str],
        params_list: List[Dict] = None,
        max_concurrent_runs: int = None,
    ):
        if max_concurrent_runs and not (max_concurrent_runs > 0):
            raise ValueError(f'{max_concurrent_runs} must be > 0')
        self.sbatch_args = sbatch_args
        self.max_concurrent_runs = max_concurrent_runs

        # sbatch args should not provide logging if logging_dir is already provided
        for arg in sbatch_args:
            if any([arg.startswith(x) for x in SBATCH_LOGGING_PREFIXES]):
                logger.warning(f'slurm argument {arg} will be overriden by args.logging_dir')
        self.args = args
        
        # setup logging/checkpoint dirs if provided
        os.makedirs(self.args.logging_dir, exist_ok=True)
        if args.checkpoints_dir:
            os.makedirs(self.args.checkpoints_dir, exist_ok=True)

        self.runs_params = list_of_dicts(args.num_trials)
        # setup init parameters if provided
        if params_list:
            self.add_from_params_list(params_list)

    def add_variable(self, name, values):
        ''' Add a variable parameter to the experiment. '''
        if len(values) != self.args.num_trials:
            raise ValueError(f'Number of values must match number of trials: {self.args.num_trials}')
        for i, param in enumerate(self.runs_params):
            param[name] = values[i]

    def add_constant(self, name, value):
        ''' Add a constant parameter to the experiment. '''
        for param in self.runs_params:
            param[name] = value

    def add_from_params_list(self, params_list: List[Dict]) -> None:
        ''' Initialize experiment from a list of variable parameter dictionaries '''
        if len(params_list) != self.args.num_trials:
            raise ValueError(f'Number of params must match number of trials: {self.args.num_trials}')
        if not check_keys_consistent(params_list):
            raise ValueError('All params must have the same keys')

        keys = params_list[0].keys()
        collated = {}
        for key in keys:
            collated[key] = [elm[key] for elm in params_list]
        for key, values in collated.items():
            self.add_variable(key, values)
    
    def run_trial(self, params, run_index) -> None:
        ''' Run experiment trial with given parameters '''
        if 'name' in params:
            run_name = params.pop('name')
        else:
            run_name = default_make_run_name(params, run_index)
        logger.info(f'Running trial: {run_name}...')

        # automatically specify relevant dirs
        logging_subdir = self.args.logging_dir + '/' + run_name
        params['logging_dir'] = logging_subdir
        if self.args.data_dir:
            params['data_dir'] = self.args.data_dir
        if self.args.checkpoints_dir:
            params['checkpoints_dir'] = self.args.checkpoints_dir + '/' + run_name

        # automatically assign job name and generate slurm logs 
        sbatch_args = copy.deepcopy(self.sbatch_args)
        sbatch_args.extend([
            'quiet', # don't print slurm output to sdout
            f'job-name={run_name}',
            f'output={logging_subdir}/slurm.out',
            f'error={logging_subdir}/slurm.err',
        ])
        # slurm won't log/run if dir doesn't exist
        os.makedirs(logging_subdir, exist_ok=True)

        sbatch_command = make_args_command_sbatch(sbatch_args)
        params_command = make_args_command(params)
        command = f'sbatch {sbatch_command} {SBATCH_TEMPLATE_PATH} \
            {self.args.conda_env} {self.args.script} {params_command}'
        #print(command)
        os.system(command)

    def list_active_runs(self) -> List[str]:
        ''' Get list of active job names '''
        run_names_log = os.popen('squeue -u $USER -o %j').read() 
        # index zero contains 'NAMES' and last index empty string
        return run_names_log.split('\n')[1:-1]
    
    def run_scheduled(self) -> None:
        ''' Run experiment with at most max_concurrent_runs at a time '''
        num_total =  self.args.num_trials
        num_completed = 0
        while num_completed < num_total:
            num_current_runs = len(self.list_active_runs())
            num_to_launch = self.max_concurrent_runs - num_current_runs
            num_to_launch = min(num_total - num_completed, num_to_launch)

            for i in range(num_to_launch):
                run_index = num_completed + i
                self.run_trial(self.runs_params[run_index], run_index)

            num_completed += num_to_launch
            if num_completed == num_total:
                break
            # interval in seconds between checking for available resources 
            time.sleep(2)
        
    def run_all(self) -> None:
        ''' Run all trials in the experiment. If resource limit exceeded, queue the remaining runs. '''
        for i, params in enumerate(self.runs_params):
            self.run_trial(params, i)

    def run(self) -> None:
        logger.info(f'Running experiment {self.args.name}...')
        if self.max_concurrent_runs:
            # schedule runs given limited resource allocation, exiting once complete
            self.run_scheduled()
        else:
            # queue all experiments allocating all resources
            self.run_all()