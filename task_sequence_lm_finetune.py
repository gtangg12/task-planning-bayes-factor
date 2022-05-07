import glob 
import random
import pickle

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer

from babyai.common import *
from babyai_task_sequence import chunknum_from_path

from datasets.text_classification_dataset import TextSequenceClassificationDataset
from metrics import load_metric


FULLPATH = '/nobackup/users/gtangg12/task_planning_bayes_factor'
NUM_CLASSES = len(ACTIONS)
NUM_CHUNKS = 1


def load_from_dir(path, filter_fn=None, shuffle=True, num_data=None):
    """ Load BabyAI env descriptions and their auxiliary data from path, 
        filtering filename by filter_fn 
    """
    filenames = glob.glob(path + '/*.pkl')
    if filter_fn:
        filenames = list(filter(filter_fn, filenames))

    inputs = []
    for filename in filenames:
        with open(filename, 'rb') as f:
            _inputs = pickle.load(f)
        inputs.extend(_inputs)
    if shuffle:
        random.shuffle(inputs)
    if num_data:
        inputs = inputs[:num_data]
        
    texts, labels, tasknames = list(zip(*inputs))
    return texts, labels, tasknames
    

if __name__ == '__main__':
    # load data 
    texts, labels, tasknames = load_from_dir(
        f'{FULLPATH}/data/babyai/env_description_chunked', 
        filter_fn=lambda filename: chunknum_from_path(filename) < NUM_CHUNKS
    )
    # numerically encode actions
    print(ACTIONS_TO_INDEX)
    labels = list(map(lambda action: ACTIONS_TO_INDEX[action], labels))

    
    # Tokenizer, Model, and Metrics 
    tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenize_fn = lambda text: tokenizer(text, padding='max_length', truncation=True)

    model = AutoModelForSequenceClassification.from_pretrained('gpt2', num_labels=NUM_CLASSES)
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
            'labels_freq': label_frequency(labels),
            'preds_freq': label_frequency(preds),
        }


    # Datasets
    text_sequence_dataset = TextSequenceClassificationDataset(texts, labels, tokenize_fn)

    num_train = int(len(text_sequence_dataset) * 0.85)
    num_eval = len(text_sequence_dataset ) - num_train

    train_dataset, eval_dataset = \
        torch.utils.data.random_split(text_sequence_dataset , [num_train, num_eval])
    
    # Training 
    # WARNING: huggingface trainer will use all gpus on device
    training_args = TrainingArguments(
        output_dir=f'{FULLPATH}/checkpoints/babyai_lm', 
        evaluation_strategy='steps', 
        eval_steps=512,
        save_strategy='epoch',
        gradient_accumulation_steps=4, 
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
