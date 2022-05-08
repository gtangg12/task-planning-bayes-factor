import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from babyai.common import *
from babyai_task_sequence import chunknum_from_path

from metrics import load_metric
from datasets.load_data_utils import (
    load_from_dir,
    compute_train_eval_split
)
from datasets.text_classification_dataset import TextSequenceClassificationDataset


# load data
FULLPATH = '/nobackup/users/gtangg12/task_planning_bayes_factor'
NUM_CHUNKS = 1
    
def filter_fn(filename):
   return chunknum_from_path(filename) < NUM_CHUNKS

dataset_path = f'{FULLPATH}/data/babyai/env_description_chunked'
inputs = load_from_dir(dataset_path, filter_fn)
texts, labels, tasknames = list(zip(*inputs))

# numerically encode actions
labels = list(map(lambda action: ACTIONS_TO_INDEX[action], labels))


# Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenize_fn = lambda text: tokenizer(text, padding='max_length', truncation=True)

model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=len(ACTIONS))
model.config.pad_token_id = tokenizer.pad_token_id


# Metrics
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


# Datasets
text_sequence_dataset = TextSequenceClassificationDataset(texts, labels, tokenize_fn)
num_train, num_eval = compute_train_eval_split(len(text_sequence_dataset))
train_dataset, eval_dataset = \
    torch.utils.data.random_split(text_sequence_dataset , [num_train, num_eval])

# Training 
# WARNING: huggingface trainer will use all gpus on device
training_args = TrainingArguments(
    output_dir=f'{FULLPATH}/checkpoints/babyai_lm', 
    evaluation_strategy='steps', 
    eval_steps=128,
    save_strategy='epoch',
    gradient_accumulation_steps=64, 
    num_train_epochs=10,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
