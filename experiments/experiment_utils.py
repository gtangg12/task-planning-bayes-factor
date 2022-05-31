from typing import Dict, List, Set


def list_of_dicts(n: int) -> List[Dict]:
    ''' Helper function to make a list of n empty dicts '''
    return [{} for _ in range(n)]


def check_keys_consistent(params_list: List[Dict]) -> bool:
    ''' Helper function to check if all params have the same keys. 
        Assumes params_list has nonzero length. 
    '''
    keys = params_list[0].keys()
    for params in params_list:
        if params.keys() != keys:
            return False
    return True


def default_make_run_name(run_params: Dict, run_index: int) -> str:
    ''' Helper function to make a run name from run param dict '''
    if len(run_params) == 0:
        return f'run{run_index}'
    name = []
    for key, value in run_params.items():
        name.append(f'{key}-{value}')
    return '_'.join(name)


def make_args_command_sbatch(args: List) -> str:
    ''' Helper function to make a string of sbatch arguments from args dict '''
    cmd = []
    for value in args:
        cmd.append(f'--{value}')
    return ' '.join(cmd)


def make_args_command(args: Dict) -> str:
    ''' Helper function to make a string of command line arguments from args dict '''
    cmd = []
    for key, value in args.items():
        cmd.append(f'--{key} {value}')
    cmd = ' '.join(cmd)
    # wrap in quotes for bash to pass as args string to python
    return f'"{cmd}"'