import os
import argparse

import torch
import torch.nn as nn

from babyai.common import *

from metrics import load_metric
from datasets.load_data_utils import (
    load_from_dir,
    compute_train_eval_split
)
from datasets.task_sequence_dataset import collate_fn
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


''' Metrics '''
classification_accuracy = load_metric('classification', 'accuracy')
label_frequency = load_metric('classification', 'label_frequency')

def compute_metrics(outputs):
    logits, labels = outputs
    labels = labels.squeeze()
    _, preds = torch.max(logits, dim=1)
    print(preds)
    return {
        'accuracy': classification_accuracy(preds, labels),
        'preds_freq': label_frequency(preds),
        'labels_freq': label_frequency(labels),
    }


''' Loss '''
bce_loss = nn.BCELoss()

def loss_fn(logits, labels):
    labels = torch.unsqueeze(labels, dim=1)
    labels = labels.float()
    return bce_loss(logits, labels)


''' Loading Data '''
NUM_CHUNKS = 1

sequences = load_from_dir(
    args.data_dir, 
    num_data=args.num_data,
    filename_load_fn=load_sequences,
    filename_filter_fn=lambda filename: chunknum_from_path(filename) < NUM_CHUNKS and taskname_from_path(filename) == 'GoToLocal' 
)


''' Datasets '''
babyai_sequence_dataset = BabyaiSequenceDataset(sequences)
num_train, num_eval = compute_train_eval_split(len(babyai_sequence_dataset))
train_dataset, eval_dataset = \
    torch.utils.data.random_split(babyai_sequence_dataset , [num_train, num_eval])


''' Model '''
model = ClassifierFilmRNN(
    num_channels=NUM_VIEW_FEATURES, 
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

optimizer=torch.optim.Adam(model.parameters(), lr=0.001)
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=collate_fn,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    criterion=loss_fn,
    optimizer=torch.optim.Adam(model.parameters(), lr=5e-5),
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min'),
)

trainer.train()