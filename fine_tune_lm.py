from typing import List, Dict, Tuple, Callable

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from babyai.common import *
from babyai_task_sequence import chunknum_from_path

from datasets.text_sequence_classification import (
    TextSequenceClassificationDataset,
    load_from_dir
)


NUM_CLASSES = len(ACTIONS)
NUM_CHUNKS = 2

filter_fn = lambda filename: chunknum_from_path(filename) <= NUM_CHUNKS
texts, labels = load_from_dir('data/babyai/env_description_chunked', filter_fn)

tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)

def tokenize_function(text):
    return tokenizer(text, padding='max_length', truncation=True)

'''
combined_dataset = TextSequenceClassificationDataset(texts, labels, tokenize_function)
num_data = len(combined_dataset)
train_dataset, eval_dataset = \
    torch.utils.data.random_split(combined_dataset, [0.85 * num_data, 0.15 * num_data])


accuracy_metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=preds, references=labels)


model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=NUM_CLASSES)


training_args = TrainingArguments(
    output_dir='checkpoints/babyai_lm', 
    evaluation_strategy='epoch', 
    save_strategy='epoch',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
'''