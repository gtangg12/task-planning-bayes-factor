import os
import argparse

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
from workflows import Trainer, TrainingArguments


''' Experiment params '''
parser = argparse.ArgumentParser(
    description='Train classifier for task sequence completition classification.')
parser.add_argument(
    '--data_dir', type=str, help='Task sequence dataset directory')
parser.add_argument(
    '--logging_dir', type=str, help='Output directory for trainer logs.')
parser.add_argument(
    '--checkpoints_dir', type=str, help='Output directory for saved model checkpoints')
parser.add_argument(
    '--num_data', type=int, default=None, help='Defaults to all data used.')
args = parser.parse_args()

os.makedirs(args.logging_dir, exist_ok=True)
os.makedirs(args.checkpoints_dir, exist_ok=True)

exit()

''' Metrics '''
classification_accuracy = load_metric('classification', 'accuracy')
label_frequency = load_metric('classification', 'label_frequency')

def compute_metrics(outputs):
    logits, labels = outputs
    logits, labels = torch.from_numpy(logits), torch.from_numpy(labels)
    _, preds = torch.max(logits, dim=1)
    return {
        'accuracy': classification_accuracy(preds, labels),
        'preds_freq': label_frequency(preds),
        'labels_freq': label_frequency(labels),
    }


''' Loading Data '''
NUM_CHUNKS = 1

sequences = load_from_dir(
    args.data_dir, 
    load_fn=load_sequences,
    filename_filter_fn=lambda filename: chunknum_from_path(filename) < NUM_CHUNKS 
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


''' Training '''
training_args = TrainingArguments(
    logging_dir=args.logging_dir,
    save_dir=args.checkpoints_dir, 
    save_epochs=5,
    num_train_epochs=100,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()