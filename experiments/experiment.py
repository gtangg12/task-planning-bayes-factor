import os
import time
import logging
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Callable, Optional, Any

from experiments.experiment_args import ExperimentArguments
from experiments.experiment_utils import (
    list_of_dicts,
    list_of_dicts_keys_consistent,
    dict_has_keys,
    make_python_args_command
)
from experiments.slurm_utils import (
    list_active_run_names,
    params_slurm_list_to_dict,
    make_slurm_args_command
)
from experiments.logging_utils import current_time_str


logger = logging.getLogger(__name__)


# slurm template file for running experiments
SLURM_TEMPLATE_PATH = os.path.dirname(__file__) + '/template.sh' 


@dataclass
class ExperimentState:
    ''' 
    Class containing the experiment inner state. 
    
    *Optional arguments set only after load_next_trial() 

    Args:
        run_index: index of current run (-1 denotes no trials running yet)
        run_name: current run name 
        logging_subdir: directory where current run output is logged. 
        params_trial: current run trial parameters
        params_slurm: current run slurm parameters
    '''
    run_index: int = 0
    run_name: Optional[str] = None
    logging_subdir: Optional[str] = None
    params_trial: Optional[Dict] = None
    params_slurm: Optional[Dict] = None


class Experiment:
    def __init__(
        self, 
        args: ExperimentArguments, 
        params_slurm: List[str],
        params_trial_list: Optional[List[Dict]] = None,
    ):
        self.args = args
        args.name += ':' + current_time_str() # unique experiment name in case of multiple runs
        if args.logging_dir:
            os.makedirs(self.args.logging_dir, exist_ok=True)
        if args.checkpoints_dir:
            os.makedirs(self.args.checkpoints_dir, exist_ok=True)

        self.params_slurm = params_slurm_list_to_dict(params_slurm)
        if dict_has_keys(
            self.params_slurm, ['--job_name', '--output', '--error', '-J', '-o', '-e']
        ):
            warnings.warn(f'slurm logging args will be overriden by values provided by experiment')

        self.params_trial = list_of_dicts(args.num_trials)
        if params_trial_list:
            self.add_params_trial_list(params_trial_list)

        self.state = ExperimentState(
            params_slurm=self.params_slurm
        )

    def add_params_trial_list(self, params_list: List[Dict]) -> None:
        ''' Initialize experiment from a list of variable parameter dictionaries '''
        if len(params_list) != self.args.num_trials:
            raise ValueError(f'Number of params must match number of trials: {self.args.num_trials}')
        if not list_of_dicts_keys_consistent(params_list):
            raise ValueError('All param dicts must have the same keys')

        for i, params in enumerate(params_list):
            self.params_trial[i].update(params)

    def add_variable(self, name: str, values: List[Any]) -> None:
        ''' Add a variable parameter to the experiment. '''
        if len(values) != self.args.num_trials:
            raise ValueError(f'Number of values must match number of trials: {self.args.num_trials}')

        for i, param in enumerate(self.params_trial):
            param[name] = values[i]

    def add_constant(self, name: str, value: Any) -> None:
        ''' Add a constant parameter to the experiment. '''
        for param in self.params_trial:
            param[name] = value

    def list_active_experiment_run_names(self):
        return list_active_run_names(
            filter_fn=lambda run_name: run_name.startswith(f'{self.args.name}:')
        )

    ''' Following class methods can be overriden to customize experiment logging, slurm logging, and 
        trial param preparation based on experiment state/args
    '''
    @classmethod
    def make_run_name(cls, state: ExperimentState, args: ExperimentArguments):
        ''' Returns name for running trial. 
            * run_index, params_trial, and params_slurm will be initialized when this function is called.
        '''
        if len(state.params_trial) == 0:
            return f'run{state.run_index}'
        name = []
        for key, value in state.params_trial.items():
            name.append(f'{key}-{value}')
        return '_'.join(name)

    @classmethod
    def make_logging_subdir(cls, state: ExperimentState, args: ExperimentArguments):
        ''' Make subdirectory of logging_dir where results of specific trial as well as slurm 
            execution output logs are written to (slurm run unless dir where execution output 
            logs are written to exists). 
        '''
        if args.logging_dir:
            logging_subdir = f'{args.logging_dir}/{state.run_name}'
        else:
            logging_subdir = f'logs/{args.name}/{state.run_name}'
        os.makedirs(logging_subdir, exist_ok=True)
        return logging_subdir

    @classmethod
    def prepare_params_trial(cls, state: ExperimentState, args: ExperimentArguments) -> Dict:
        ''' Default function to prepare params for run. Allows user to inject additional params
            that are dependent on experiment state/args.
            * called after trial initialized. 
        '''
        if args.data_dir:
            state.params_trial['data_dir'] = args.data_dir
        if args.logging_dir:
            state.params_trial['logging_dir'] = state.logging_subdir
        if args.checkpoints_dir:
            state.params_trial['checkpoints_dir'] = f'{args.checkpoints_dir}/{state.run_name}'

    @classmethod
    def prepare_params_slurm(cls, state: ExperimentState, args: ExperimentArguments) -> Dict:
        ''' Default function to prepare slurm args for run. Allows user to configure slurm based
            on experiment state/args i.e. where slurm logs are written to.
            * called after trial initialized. 
        '''
        state.params_slurm.update({
            '--job-name': f'{args.name}:{state.run_name}', 
            '--output': f'{state.logging_subdir}/slurm.out',
            '--error': f'{state.logging_subdir}/slurm.err',
        })

    def init_next_trial(self, state: ExperimentState, args: ExperimentArguments) -> None:
        ''' Load next experiment state given current state '''            
        state.params_trial = self.params_trial[state.run_index]
        state.run_name = self.make_run_name(state, args)
        state.logging_subdir = self.make_logging_subdir(state, args)

    def run_trial(self, state: ExperimentState, args: ExperimentArguments, verbose=False) -> None:
        self.prepare_params_trial(state, args)
        self.prepare_params_slurm(state, args)

        logger.info(f'Running trial: {state.run_name}...')

        sbatch_command = make_slurm_args_command(state.params_slurm)
        python_command = make_python_args_command(state.params_trial)
        command = f'sbatch {sbatch_command} {SLURM_TEMPLATE_PATH} {args.conda_env} {args.script} {python_command}'
        
        if verbose:
            logger.info('Command: ' + command)
        os.system(command)
    
    def run_trials(self, n_trials):
        ''' Run next n trials in experiment '''
        for _ in range(n_trials):
            self.init_next_trial(self.state, self.args)
            self.run_trial(self.state, self.args)
            self.state.run_index += 1

    def run_dynamic(self):
        ''' Run experiment with at most args.max_concurrent_running_trials at a time. Blocks until all runs launched. '''
        SLEEP_SECONDS = 30 # interval in seconds between experiment checking for available resources

        num_total =  self.args.num_trials
        num_completed = 0
        while num_completed < num_total:
            num_current_runs = len(self.list_active_experiment_run_names())
            num_to_launch = self.args.max_concurrent_running_trials - num_current_runs
            num_to_launch = min(num_to_launch, num_total - num_completed)

            self.run_trials(num_to_launch)
            
            num_completed += num_to_launch
            if num_completed == num_total:
                break
            time.sleep(SLEEP_SECONDS) 

    def run(self) -> None:
        ''' Run experiment. If args.max_concurrent_running_trials is set, run dynamically i.e. schedule runs given limited 
            resource allocation, blocking until all runs launched. Otherwise run all trials in the experiment. 
            If resource limit allocated by slurm admin exceeded, queue the remaining runs.

            <Tip> Launch in and detach from tmux session to circumvent blocking if running dynamically.
        '''
        logger.info(f'Running experiment {self.args.name}...')

        if self.args.max_concurrent_running_trials:
            self.run_dynamic()
        else:
           self.run_trials(self.args.num_trials)