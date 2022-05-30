import os
import json
import logging
from typing import Any, List, Dict, Callable, Optional, NewType

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from metrics import Logits, Labels
from datasets.collate_utils import DataCollatorFunc
from workflows.training_args import TrainingArguments
from workflows.trainer_utils import (
    NUM_DEVICES_AVAILABLE,
    DEFAULT_DEVICE,
    dict_to_device,
    dict_to_serializable,
)

Loss = float


#TODO raytune


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        data_collator: Optional[DataCollatorFunc] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[Logits, Labels], Dict]] = None,
        criterion: Optional[Callable[[Logits, Labels], Loss]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        # use all gpus available
        model = nn.DataParallel(model).to(DEFAULT_DEVICE)
        model.to(DEFAULT_DEVICE)
        
        self.args = args

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = self.get_train_dataloader(train_dataset) if train_dataset else None
        self.eval_dataloader = self.get_eval_dataloader(eval_dataset) if eval_dataset else None

        self.compute_metrics = compute_metrics
        self.tb_writer = SummaryWriter(log_dir=self.args.logging_dir + '/tensorboard')
        os.makedirs(self.args.logging_dir + '/metrics', exist_ok=True)

        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.optimizer.zero_grad()

    def get_train_dataloader(self, dataset):
        ''' Returns a new instance of DataLoader constructed from the training dataset.'''
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size * NUM_DEVICES_AVAILABLE,
            shuffle=self.args.shuffle,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, dataset):
        ''' Returns a new instance of DataLoader constructed from the evaluation dataset.'''
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a train_dataset.")

        return DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size * NUM_DEVICES_AVAILABLE,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, inputs, return_outputs=False):
        if self.criterion is None:
            raise ValueError("Trainer: training requires a criterion or compute_loss() to be overridden.")

        labels = inputs['label']
        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        return (loss, logits) if return_outputs else loss
    
    def training_step(self, inputs, epoch):
        loss, logits = self.compute_loss(inputs, return_outputs=True)
        
        if (epoch + 1) % self.args.gradient_accumulation_epochs == 0:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss, logits

    def prediction_step(self, inputs):
        loss, logits = self.compute_loss(inputs, return_outputs=True)
        return loss, logits
    
    def train(self):
        if not self.train_dataloader:
            raise ValueError("Trainer: training requires a train_dataset.")
        if not self.optimizer:
            raise ValueError("Trainer: training requires an optimizer.")
        
        for epoch in range(self.args.num_train_epochs):
            print(f'Epoch-{epoch}')

            train_loss = 0
            logits, labels = [], []
            self.model.train()
            for step, inputs in enumerate(tqdm(self.train_dataloader)):
                dict_to_device(inputs, DEFAULT_DEVICE)
                loss, _logits = self.training_step(inputs, epoch)
                train_loss += loss.item()
                logits.append(_logits)
                labels.append(inputs['label'])
            train_loss /= len(self.train_dataloader)

            train_metrics = {'train_loss': train_loss}
            eval_metrics = self.evaluate(self.eval_dataloader) if self.eval_dataloader else None

            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(eval_metrics['eval_loss'])
                else:
                    self.scheduler.step()

            if self.args.logging_dir and (epoch + 1) % self.args.logging_epochs == 0:
                self.log(f'train', epoch, train_metrics)
                self.log(f'eval', epoch, eval_metrics)

            if self.args.save_dir and (epoch + 1) % self.args.save_epochs == 0:
                self.save_checkpoint(epoch)

    def evaluate(self, loader: DataLoader):
        eval_loss = 0
        logits, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(loader)):
                inputs = dict_to_device(inputs, DEFAULT_DEVICE)
                loss, _logits = self.prediction_step(inputs)
                eval_loss += loss.item()
                logits.append(_logits)
                labels.append(inputs['label'])
        eval_loss /= len(self.eval_dataloader)
        
        metrics = {'eval_loss': eval_loss}
        if self.compute_metrics:
           metrics.update(
               self.compute_metrics((torch.cat(logits), torch.cat(labels)))
            )
        return metrics

    def predict(self, test_dataset: Dataset):
        test_loader = self.get_eval_dataloader(test_dataset)
        test_metrics = self.evaluate(test_loader)
        return test_metrics

    def log(self, entry_name, epoch, metrics):
        ''' Dumps metrics to json and logs to tensorboard. '''
        logging_filename = f'{self.args.logging_dir}/metrics/{entry_name}_{epoch:03d}.json'
        with open(logging_filename, 'w') as f:
            json.dump(dict_to_serializable(metrics), f)
            
        self.log_tensorboard(entry_name, epoch, metrics)

    def log_tensorboard(self, entry_name, epoch, metrics):
        ''' Logs metrics to tensorboard. User can override default behavior, 
            which is to log all scalars. 
        '''
        for key, value in metrics:
            if isinstance(value, (int, float)):
                self.tb_writer.add_scalar(f'{entry_name}/{key}', value, epoch)
            else:
                logging.warning(f'Attempted to log non-scalar value {key} to tensorboard, so we dropped attribute.')
        self.tb_writer.flush()

    def save_checkpoint(self, epoch):
        checkpoint_filename = f'{self.args.save_dir}/{epoch:03d}_checkpoint.pt'
        print(f'Saved checkpoint for epoch {epoch} at {checkpoint_filename}')
        torch.save({
            'model': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }, checkpoint_filename)

    def load_checkpoint(self, checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['scheduler']:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.args.num_train_epochs -= checkpoint['epoch']
        print(f'Loaded checkpoint {checkpoint_filename}')