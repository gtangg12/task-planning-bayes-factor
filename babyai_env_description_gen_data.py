"""  Script to generate dataset of textual descriptions of a timestamp during a babyai task sequence
"""

import os
import itertools
import argparse
import glob
import pickle
import multiprocessing

from babyai.common import *

from babyai_task_sequence import load_sequences, taskname_from_path
from babyai_env_description import generate_env_description


def generate_samples(path, args):
    taskname = taskname_from_path(path)
    filename = path.split('/')[-1]
    print(f'Generating samples from {filename}...')
    
    sequences = load_sequences(path)
    texts = []
    for sequence in sequences:
        texts.append((*generate_env_description(sequence, args.view_encoding), taskname))
    pickle.dump(texts, open(f'{args.output_dir}/{filename}', 'wb'))

    print(f'Done: {filename}')


def make_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    filenames = glob.glob(f'{args.input_dir}/*.pkl')
    n_tasks = len(filenames)

    pool = multiprocessing.Pool(processes=args.n_processes)
    pool.starmap(
        generate_samples, 
        zip(
            filenames, 
            itertools.repeat(args, n_tasks)
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate a dataset of textual description dataset of time snapshots of sequences.')
    parser.add_argument(
        '--input_dir', type=str, default='data/babyai/task_sequence_chunked')
    parser.add_argument(
        '--output_dir', type=str, default='data/babyai/env_description_chunked')
    parser.add_argument(
        '--view_encoding', type=str, default='cardinal_diagonal', \
        help='View encoding of ego view to use for generating textual descriptions.')
    parser.add_argument(
        '--n_processes', type=int, default=4, help='Number of processes to spawn')
    args = parser.parse_args()

    make_dataset(args)