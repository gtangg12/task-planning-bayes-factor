import argparse 

from experiments import (
    Experiment,
    ExperimentArguments,
    experiment,
)


experiments = {
    'babyai_gpt2_finetune_n_data': {
        'script'  : 'task_sequence_gpt2_finetune.py',
        'data_dir': 'data/babyai/env_description_chunked',
    },
    'babyai_classifier_train_n_data': {
        'script'  : 'task_sequence_classifier_train.py',
        'data_dir': 'data/babyai/task_sequence_chunked',
    },
}


parser = argparse.ArgumentParser(
    description='Select task sequence num data experiment')
parser.add_argument(
    '--name', '-n', type=str, help='Name of experiment. Must be in experiments dict.')
args = parser.parse_args()

if args.name not in experiments:
    raise ValueError(f'{args.name} not in experiments dict.')
experiment_dict = experiments[args.name]


sbatch_args = [
    'mail-user=gtangg12@mit.edu',
    'mail-type=NONE',
    'nodes=1',
    'ntasks-per-node=1',
    'cpus-per-task=4',
    'gres=gpu:1',
    'mem=256G',
    'time=24:00:00',
]

experiment_args = ExperimentArguments(
    name=args.name,
    script=experiment_dict['script'],
    conda_env='task_planning_babyai',
    n_trials=1,
    data_dir=experiment_dict['data_dir'],
    auto_logging_checkpoint_dirs=True,
    sbatch_args=sbatch_args,
)

experiment = Experiment(experiment_args, [{'num_data': 100}])

#experiment.add_variable('n_data', [100, 500, 1000, 2500, 5000, 10000, 50000, 100000])

experiment.run()
