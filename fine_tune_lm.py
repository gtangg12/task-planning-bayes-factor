from typing import List, Dict, Tuple, Callable

import numpy as np
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from babyai.common import *
from babyai_task_sequence import chunknum_from_path

from datasets.text_sequence_classification import (
    TextSequenceClassificationDataset,
    load_sequences_from_dir
)


NUM_CLASSES = len(ACTIONS)
NUM_CHUNKS = 1


filter_fn = lambda filename: chunknum_from_path(filename) < NUM_CHUNKS
texts, labels = load_sequences_from_dir('data/babyai/env_description_chunked', filter_fn)

# numeric encode actions
labels = list(map(lambda action: ACTIONS_TO_INDEX[action], labels))


tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenize_fn = lambda text: tokenizer(text, padding='max_length', truncation=True)

model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=NUM_CLASSES)
model.config.pad_token_id = tokenizer.pad_token_id

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    accuracy = np.mean(preds == labels)
    return {
        'accuracy': accuracy,
    }


combined_dataset = TextSequenceClassificationDataset(texts, labels, tokenize_fn)
num_train = int(len(combined_dataset) * 0.85)
train_dataset, eval_dataset = \
    torch.utils.data.random_split(combined_dataset, [num_train, len(combined_dataset) - num_train])


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
