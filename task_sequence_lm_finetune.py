import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from babyai.common import *
from babyai_task_sequence import chunknum_from_path

from datasets.text_sequence_classification import (
    TextSequenceClassificationDataset,
    load_text_sequences_from_dir
)
from metrics import load_metric


FULLPATH = '/nobackup/users/gtangg12/task_planning_bayes_factor'

NUM_CLASSES = len(ACTIONS)
NUM_CHUNKS = 1


# Load data 
filter_fn = lambda filename: chunknum_from_path(filename) < NUM_CHUNKS
texts, labels = load_text_sequences_from_dir(f'{FULLPATH}/data/babyai/env_description_chunked', filter_fn=filter_fn)

# numeric encode actions
labels = list(map(lambda action: ACTIONS_TO_INDEX[action], labels))


# Tokenizer, Model, and Metrics 
tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenize_fn = lambda text: tokenizer(text, padding='max_length', truncation=True)

model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=NUM_CLASSES)
model.config.pad_token_id = tokenizer.pad_token_id


# Metrics
classification_accuracy = load_metric('classification', 'accuracy')

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return {
        'accuracy': classification_accuracy(torch.from_numpy(logits), torch.from_numpy(labels)),
    }


# Datasets
text_sequence_dataset = TextSequenceClassificationDataset(texts, labels, tokenize_fn)

num_train = int(len(text_sequence_dataset) * 0.85)
num_eval = len(text_sequence_dataset ) - num_train

train_dataset, eval_dataset = \
    torch.utils.data.random_split(text_sequence_dataset , [num_train, num_eval])


# Training 
training_args = TrainingArguments(
    output_dir=f'{FULLPATH}/checkpoints/babyai_lm', 
    evaluation_strategy='steps', 
    eval_steps=4,
    save_strategy='epoch',
    gradient_accumulation_steps=2, # 512 * n_gpus effective bs
    num_train_epochs=5,
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
