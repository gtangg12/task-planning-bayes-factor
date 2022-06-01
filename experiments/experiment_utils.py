from typing import Dict, List, Iterable


def list_of_dicts(n: int) -> List[Dict]:
    ''' Make a list of n empty dicts '''
    return [{} for _ in range(n)]


def check_keys_consistent(list_of_dicts: List[Dict]) -> bool:
    ''' Check if all dicts in list have the same keys '''
    if len(list_of_dicts) == 0:
        return True
    keys = list_of_dicts[0].keys()
    for d in list_of_dicts:
        if d.keys() != keys:
            return False
    return True
    

def check_has_keys(d: Dict, keys: Iterable) -> bool:
    ''' Check if dict d contains keys in keys '''
    if not isinstance(keys, set):
        keys = set(keys)
    return len(d.keys() & keys)


def default_make_run_name(params: Dict, run_index: int) -> str:
    ''' Make default run name from params dict '''
    if len(params) == 0:
        return f'run{run_index}'
    name = []
    for key, value in params.items():
        name.append(f'{key}-{value}')
    return '_'.join(name)


def make_python_args_command(args: Dict) -> str:
    ''' Make python command line arguments string from args dict '''
    cmd = []
    for key, value in args.items():
        cmd.append(f'--{key} {value}')
    cmd = ' '.join(cmd)
    # wrap in quotes for bash to pass to python
    return f'"{cmd}"'