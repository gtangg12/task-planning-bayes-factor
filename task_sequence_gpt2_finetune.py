import os
import argparse

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

from babyai.common import *
from babyai_task_sequence import chunknum_from_path

from metrics import load_metric
from datasets.load_data_utils import (
    load_from_dir,
    compute_train_eval_split
)
from datasets.text_classification_dataset import TextSequenceClassificationDataset
from workflows import (
    TransformersTrainer, 
    TransformersLoggingCallback,
    TransformersTrainingArguments
)
from workflows.trainer_utils import dict_to_serializable


''' Experiment params '''
parser = argparse.ArgumentParser(
    description='Finetune GPT2 for text sequence classification.')
parser.add_argument(
    '--data_dir', type=str, help='Text classification dataset directory')
parser.add_argument(
    '--logging_dir', type=str, help='Output directory for trainer logs.')
parser.add_argument(
    '--checkpoints_dir', type=str, help='Output directory for saved model checkpoints')
parser.add_argument(
    '--num_data', type=int, default=None, help='Defaults to all data used.')
args = parser.parse_args()

os.makedirs(args.logging_dir, exist_ok=True)
os.makedirs(args.checkpoints_dir, exist_ok=True)


## TODO: modify transformers trainer so metrics are written as json when trainer.log is called in trainer.evaluate
## TODO: train metrics (see metric_key_prefix)
## TODO: compute separate states divs for each task based on inputs(recursive compute_metrics)
##      include_inputs_for_metrics=True (currently satori doesnt have package to support, wait until 4.19 transformers available)

''' Metrics '''
accuracy = load_metric('classification-accuracy')
kl_divergence = load_metric('classification-kl_divergence')
label_frequency = load_metric('classification-label_frequency')


def compute_metrics(outputs):
    # transformers trainer returns outputs as numpy arrays
    logits, labels = outputs
    logits, labels = torch.from_numpy(logits), torch.from_numpy(labels)
    _, preds = torch.max(logits, dim=1)

    preds_freq, labels_freq = \
        label_frequency(preds, NUM_ACTIONS), label_frequency(labels, NUM_ACTIONS)

    metrics = {
        'accuracy': accuracy(preds, labels),
        'preds_freq': preds_freq,
        'labels_freq': labels_freq,
        'kl_divergence': kl_divergence(preds_freq, labels_freq),
    }
    return dict_to_serializable(metrics)


''' Tokenizer '''
tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenize_fn = lambda text: tokenizer(text, padding='max_length', truncation=True)

''' Loading data '''
NUM_CHUNKS_USED = 1
inputs = load_from_dir(
    args.data_dir, 
    num_data=args.num_data,
    filter_fn=lambda filename: chunknum_from_path(filename) < NUM_CHUNKS_USED
)
texts, labels, tasknames = list(zip(*inputs))

# numerically encode actions
labels = list(map(lambda action: ACTIONS_TO_INDEX[action], labels))


''' Datasets '''
text_sequence_dataset = TextSequenceClassificationDataset(texts, labels, tokenize_fn)
num_train, num_eval = compute_train_eval_split(len(text_sequence_dataset))
train_dataset, eval_dataset = \
    torch.utils.data.random_split(text_sequence_dataset , [num_train, num_eval])


''' Model '''
model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=len(ACTIONS))
model.config.pad_token_id = tokenizer.pad_token_id


''' Training '''
logging_callback = TransformersLoggingCallback(
    logging_dir=args.logging_dir + '/metrics',
    entries_to_log=['eval'],
    metrics_to_log=['loss', 'accuracy', 'preds_freq', 'labels_freq', 'kl_divergence'],
)

training_args = TransformersTrainingArguments(
    output_dir=args.checkpoints_dir, 
    evaluation_strategy='epoch', 
    save_strategy='epoch',
    logging_strategy='epoch',
    logging_dir=args.logging_dir + '/tensorboard', # transformers logging_dir refers to tensorboard
    gradient_accumulation_steps=4, 
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4
)

trainer = TransformersTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    callbacks=[logging_callback]
)

trainer.train()
