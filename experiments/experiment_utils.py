import os
from typing import Dict


LOGGING_DIR = f'runs/'
os.makedirs(LOGGING_DIR, exist_ok=True)


def wrap_quotes(str: str):
    return f'"{str}"'


def make_args(args: Dict) -> str:
    cmd = []
    for key, value in args.items():
        cmd.append(f'--{key} {value}')
    # wrap in quotes for bash script to pass args as string to python script
    return wrap_quotes(' '.join(cmd))


def make_sbatch(run_name: str, slurm_path: str) -> str:
    cmd = [
        'sbatch',
        f'-J {run_name}',
        f'-o {run_name}.out',
        f'-e {run_name}.err',
        slurm_path,
    ]
    return ' '.join(cmd)