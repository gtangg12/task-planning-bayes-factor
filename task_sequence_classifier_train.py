import torch

from babyai.common import *

from metrics import load_metric
from datasets.load_data_utils import (
    load_from_dir,
    compute_train_eval_split
)
from babyai_task_sequence import (
    load_sequences,
    chunknum_from_path,
    taskname_from_path
)
from babyai_task_sequence_dataset import BabyaiSequenceDataset
from task_sequence_classifier import ClassifierFilmRNN


''' Metrics '''
classification_accuracy = load_metric('classification', 'accuracy')
label_frequency = load_metric('classification', 'label_frequency')


''' Loading Data '''
FULLPATH = '/nobackup/users/gtangg12/task_planning_bayes_factor'
NUM_CHUNKS = 1

def filter_fn(filename):
   return chunknum_from_path(filename) < NUM_CHUNKS

dataset_path = f'{FULLPATH}/data/babyai/task_sequence_chunked'
sequences = load_from_dir(
    dataset_path, filename_filter_fn=filter_fn, load_fn=load_sequences
)


''' Datasets '''
babyai_sequence_dataset = BabyaiSequenceDataset(sequences)
num_train, num_eval = compute_train_eval_split(len(babyai_sequence_dataset))
train_dataset, eval_dataset = \
    torch.utils.data.random_split(babyai_sequence_dataset , [num_train, num_eval])


''' Model '''
model = ClassifierFilmRNN(
    num_channels=19, 
    vocab_size=VOCAB_SIZE, 
    action_embedding_dim=babyai_sequence_dataset.EMBEDDING_DIM
)

