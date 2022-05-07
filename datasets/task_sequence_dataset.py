import random
from typing import List, Dict, TypedDict

import torch
from torch.utils.data import Dataset

from formats.task_sequence import TaskSequence


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
        if random.random() > self.negative_sample_rate:
            sequence = self.negative_sample(sequence)
            sequence['label'] = False
        return sequence
    
    @classmethod
    def negative_sample(cls, sequence: TaskSequenceDict) -> TaskSequenceDict:
       raise NotImplementedError