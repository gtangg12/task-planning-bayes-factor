import random
from typing import List, Dict, TypedDict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from datasets.formats.task_sequence import TaskSequence
from datasets.data_collator import collate_list_of_dict


class TaskSequenceDict(TypedDict):
    taskname     : str
    task_len     : int
    task         : torch.Tensor
    images       : torch.Tensor
    actions      : torch.Tensor
    sequence_len : int


class TaskSequenceDataset(Dataset):
    def __init__(self, sequences: List[TaskSequence]) -> None:
        self.sequences = list(map(self.encode, sequences))

    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict:
        return self.sequences[idx]

    @classmethod
    def encode(cls, sequence: TaskSequence) -> TaskSequenceDict:
        raise NotImplementedError


class TaskCompletitionDataset(TaskSequenceDataset):
    def __init__(self, sequences: List[TaskSequence], negative_sample_rate: float = 0.5) -> None:
        super().__init__(sequences)
        self.negative_sample_rate = negative_sample_rate  

    def __getitem__(self, idx: int) -> Dict:
        sequence = self.sequences[idx]
        sequence['label'] = True
        if random.random() < self.negative_sample_rate:
            sequence = self.negative_sample(sequence)
            sequence['label'] = False
        return sequence
    
    @classmethod
    def negative_sample(cls, sequence: TaskSequenceDict) -> TaskSequenceDict:
       raise NotImplementedError


def collate_fn(batch: List[TaskSequenceDict]) -> Dict:
    batched = { 
        'task_len': [], 'sequence_len': [] 
    }
    for name in TaskSequenceDict.__annotations__.keys():
        batched[name] = collate_list_of_dict(batch, {name})
        if name != 'taskname': 
            batched[name] = pad_sequence(batched[name], batch_first=True)
        batched['task_len'].append(batch['task'].shape[0])
        batched['sequence_len'].append(batch['images'].shape[0])
    return batched