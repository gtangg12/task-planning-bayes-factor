import os
from typing import List, Dict, Callable


def list_active_run_names(filter_fn: Callable[[str], bool] = None) -> List[str]:
    ''' Return all active job names by current user, optionally filtered 
        by filter_fn 
    '''
    run_names = os.popen('squeue -u $USER -o %j').read() 
    run_names = run_names.split('\n')[1:-1] # index zero contains 'NAMES' and last index ''
    if filter_fn:
        run_names = list(filter(filter_fn, run_names))
    return run_names


def params_slurm_list_to_dict(args_list: List[str]) -> Dict[str, str]:
    ''' Convert a list of slurm args to a dict. Arguments with no value are
        mapped to None.
    '''
    args = {}
    for arg in args_list:
        components = arg.split('=')
        if len(components) == 1:
            components.append(None)
        args[components[0]] = components[1]
    return args


def make_slurm_args_command(args_dict: Dict[str, str]) -> str:
    ''' Make a slurm command from args_dict '''
    args_list = []
    for key, value in args_dict.items():
        args_list.append(key if value is None else f'{key}={value}')
    return ' '.join(args_list)