import os
import glob
import argparse
import itertools
import pickle
import random
import multiprocessing


def load_sequences(path):
    return pickle.load(open(path, 'rb'))


def save_sequences(path, sequences):
    pickle.dump(sequences, open(path, 'wb'))


def taskname_from_path_raw(path):
    return path.split('/')[-1].split('-')[1]


def chunk_filename_fn(path, args):
    taskname = taskname_from_path_raw(path)
    print(f'Processing: {taskname}...')

    sequences = load_sequences(path)
    assert args.n_chunks * args.chunk_size <= len(sequences), f'Insufficient data for {path}'
    random.shuffle(sequences)
    for i in range(args.n_chunks):
        chunk = sequences[i * args.chunk_size: (i + 1) * args.chunk_size]
        save_sequences(f'{args.output_dir}/{taskname}_{i:03}.pkl', chunk)
        
    print(f'Done: {taskname}')


def make_dataset(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read all non validation files 
    filenames = glob.glob(f'{args.input_dir}/*-v0.pkl')
    n_tasks = len(filenames)

    # WARNING: ensure enough memory to read all data if many processes spawned
    pool = multiprocessing.Pool(processes=args.n_processes)
    pool.starmap(
        chunk_filename_fn, 
        zip(
            filenames, 
            itertools.repeat(args, n_tasks)
        )
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Raw Babyai files are too large and need to be chunked.')
    parser.add_argument(
        '--input_dir', type=str, default='../../data/babyai/task_sequence_raw')
    parser.add_argument(
        '--output_dir', type=str, default='../../data/babyai/task_sequence_chunked')
    parser.add_argument(
        '--chunk_size', type=int, default=10000, help='Size of each chunk')
    parser.add_argument(
        '--n_chunks', type=int, default=10, help='Number of chunks to split the dataset into')
    parser.add_argument(
        '--n_processes', type=int, default=4, help='Number of processes to spawn')
    args = parser.parse_args()

    make_dataset(args)