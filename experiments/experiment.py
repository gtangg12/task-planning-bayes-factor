import os
import copy
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


class Experiment:
    def __init__(self, args: ExperimentArguments, params_list: List[Dict] = None):
        self.args = args
        self.setup_experiment(args)
        self.runs_params = list_of_dicts(args.n_trials)
        if params_list:
            self.add_from_params_list(params_list)

    def setup_experiment(self, args):
        """ Helper to initialize experiment based on args """
        if args.logging_dir:
            os.makedirs(self.args.logging_dir, exist_ok=True)
            os.makedirs(self.args.logging_dir + '/slurm', exist_ok=True)
        if args.checkpoints_dir:
            os.makedirs(self.args.checkpoints_dir, exist_ok=True)

    def add_variable(self, name, values):
        """ Add a variable parameter to the experiment. """
        if len(values) != self.args.n_trials:
            raise ValueError(f'Number of values must match number of trials: {self.args.n_trials}')
        for i, param in enumerate(self.runs_params):
            param[name] = values[i]

    def add_constant(self, name, value):
        """ Add a constant parameter to the experiment. """
        for param in self.runs_params:
            param[name] = value

    def add_from_params_list(self, params_list: List[Dict]) -> None:
        """ Initialize experiment from a list of variable parameter dictionaries """
        if len(params_list) != self.args.n_trials:
            raise ValueError(f'Number of params must match number of trials: {self.args.n_trials}')
        if not check_keys_consistent(params_list):
            raise ValueError('All params must have the same keys')

        keys = params_list[0].keys()
        collated = {}
        for key in keys:
            collated[key] = [elm[key] for elm in params_list]
        for key, values in collated.items():
            self.add_variable(key, values)
            
    def run(self) -> None:
        """ Run all experiment trials specified in self.runs_params """
        print(f'Running experiment {self.args.name}...')
        for i, params in enumerate(self.runs_params):
            self.run_trial(params, i)
    
    def run_trial(self, params, run_index) -> None:
        """ Run experiment trial with given parameters """
        if 'name' in params:
            run_name = params.pop('name')
        else:
            run_name = default_make_run_name(params, run_index)
        print(f'Running trial: {run_name}...')

        # automatically generate slurm logs 
        sbatch_args = copy.deepcopy(self.args.sbatch_args)
        if self.args.logging_dir:
            sbatch_args.extend([
                f'job-name={run_name}',
                f'output={self.args.logging_dir}/slurm/{run_name}.out',
                f'error={self.args.logging_dir}/slurm/{run_name}.err',
            ])
        # automatically generate experiment logs
        if self.args.data_dir:
            params['data_dir'] = self.args.data_dir
        if self.args.logging_dir:
            params['logging_dir'] = self.args.logging_dir + '/' + run_name
        if self.args.checkpoints_dir:
            params['checkpoints_dir'] = self.args.checkpoints_dir + '/' + run_name

        sbatch_command = make_args_command_sbatch(sbatch_args)
        params_command = make_args_command(params)
        command = f'sbatch {sbatch_command} {SBATCH_TEMPLATE_PATH} \
            {self.args.conda_env} {self.args.script} {params_command}'
        print(command)
        os.system(command)


