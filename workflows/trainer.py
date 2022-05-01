import os
import json
from typing import Any, Callable, NewType, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from training_args import TrainingArguments


# TODO figure out how to make inputdata consistent
InputData = NewType('InputDataClass', Any)

DataCollator = Callable[[list[InputData]], dict[str, Any]]

Loss = float
OutputDataList, LabelDataList = list[InputData], list[Any]
Prediction = [Loss, OutputDataList, LabelDataList, Dataset]


#TODO support start from epoch
#TODO multiple GPUs

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable[[Prediction], dict]] = None,
        criterion: Optional[Callable[[InputData, InputData], Loss]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model
        self.args = args

        self.data_collator = data_collator
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.train_dataloader = self.get_train_dataloader(train_dataset) if train_dataset else None
        self.eval_dataloader = self.get_eval_dataloader(eval_dataset) if eval_dataset else None

        self.compute_metrics = compute_metrics

        self.criterion = criterion
        self.optimizer = optimizer
        self.optimizer.zero_grad()
        self.scheduler = scheduler

    def get_train_dataloader(self, dataset) -> DataLoader:
        ''' Returns a new instance of DataLoader constructed from the training dataset.'''
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        
        return DataLoader(
            dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=self.args.shuffle,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def get_eval_dataloader(self, dataset) -> DataLoader:
        ''' Returns a new instance of DataLoader constructed from the evaluation dataset.'''
        if self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires a train_dataset.")

        return DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            shuffle=False,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
        )

    def compute_loss(self, inputs, return_outputs=False):
        if self.criterion is None:
            raise ValueError("Trainer: training requires a criterion or compute_loss() to be overridden.")

        labels = inputs.pop('labels')
        outputs = self.model(*inputs)
        loss = self.criterion(outputs, labels)
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, inputs, epoch):
        loss, outputs = self.compute_loss(inputs, return_outputs=True)

        if epoch % self.args.gradient_accumulation_epochs == 0:
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return loss, outputs

    def prediction_step(self, inputs):
        loss, outputs = self.compute_loss(inputs, return_outputs=True)
        return loss, outputs
    
    def train(self):
        if not self.train_dataloader:
            raise ValueError("Trainer: training requires a train_dataset.")
        if not self.optimizer:
            raise ValueError("Trainer: training requires an optimizer.")
            
        for epoch in range(self.args.num_train_epochs):
            print(f'Epoch-{epoch}')

            train_loss = 0
            outputs, labels = [], []
            self.model.train()
            for step, inputs in enumerate(tqdm(self.train_dataloader)):
                loss, outputs = self.training_step(inputs, epoch)
                train_loss += loss.item()
                outputs.append(outputs)
                labels.append(inputs['labels'])
            train_loss /= len(self.train_dataloader)

            eval_metrics = self.evaluate(self.eval_loader, epoch) if self.eval_dataloader else None
            
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(eval_metrics['eval_loss'])
                else:
                    self.scheduler.step()

            train_metrics = {'train_loss': train_loss}
            if self.compute_metrics:
                train_metrics.update(self.compute_metrics(train_loss, outputs, labels, self.train_dataset))

            if self.args.logging_dir and epoch % self.args.logging_interval == 0:
                self.log(f'train_{epoch:03d}', train_metrics)
                self.log(f'eval_{epoch:03d}', eval_metrics)

            if self.args.save_dir and epoch % self.args.save_epochs == 0:
                self.save_checkpoint(epoch)

    def evaluate(self, loader: DataLoader):
        eval_loss = 0
        outputs, labels = [], []
        self.model.eval()
        with torch.no_grad():
            for step, inputs in enumerate(tqdm(loader)):
                loss, outputs = self.prediction_step(inputs)
                eval_loss += loss.item()
                outputs.append(outputs)
                labels.append(inputs['labels'])
        eval_loss /= len(self.eval_loader)
        
        metrics = {'eval_loss': eval_loss}
        if self.compute_metrics:
           metrics.update(self.compute_metrics(eval_loss, outputs, labels, self.eval_dataset))
        return metrics

    def predict(self, test_dataset: Dataset):
        test_loader = self.get_eval_dataloader(test_dataset)
        test_metrics = self.evaluate(test_loader)
        return test_metrics

    def log(self, entry_name, metrics):
        logging_filename = f'{self.args.logging_dir}/{entry_name}_metrics.json'
        with open(logging_filename, 'w') as f:
            json.dump(metrics, f)

    def save_checkpoint(self, epoch):
        checkpoint_filename = f'{self.args.save_dir}/{epoch:03d}_checkpoint.pt'
        torch.save({
            'model': self.model.state_dict(),
            'epoch': epoch,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
        }, checkpoint_filename)

    # TODO load_checkpoint ?