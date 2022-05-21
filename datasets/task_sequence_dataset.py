import random
from typing import List, Dict, TypedDict

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from datasets.formats.task_sequence import TaskSequence
from datasets.collate_utils import collate_list_of_dict


class TaskSequenceDict(TypedDict):
    taskname     : str
    task_len     : int
    task         : torch.Tensor
    images       : torch.Tensor
    actions      : torch.Tensor
    sequence_len : int


class TaskCompletitionDict(TaskSequenceDict):
    label: bool


class TaskSequenceDataset(Dataset):
    def __init__(self, sequences: List[TaskSequence]) -> None:
        self.sequences = sequences
        self.encoded = list(map(self.encode, sequences))

    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> TaskSequenceDict:
        return self.sequences[idx]

    @classmethod
    def encode(cls, sequence: TaskSequence) -> TaskSequenceDict:
        raise NotImplementedError


class TaskCompletitionDataset(TaskSequenceDataset):
    def __init__(self, sequences: List[TaskSequence], negative_sample_rate: float = 0.5) -> None:
        super().__init__(sequences)
        self.negative_sample_rate = negative_sample_rate  

    def __getitem__(self, idx: int) -> TaskCompletitionDict:
        encoded = self.encoded[idx]
        if random.random() < self.negative_sample_rate:
            encoded = self.negative_sample(encoded)
        return encoded
    
    @classmethod
    def negative_sample(cls, encoded: TaskCompletitionDict) -> TaskCompletitionDict:
       raise NotImplementedError


def collate_fn(batch: List[TaskCompletitionDict]) -> Dict:
    batched = {}
    batched.update(
        collate_list_of_dict(batch, {'taskname', 'task', 'images', 'actions'})
    )
    for name in ['images', 'actions', 'task']:
        batched[name] = pad_sequence(batched[name], batch_first=True)
    batched.update(
        collate_list_of_dict(batch, {'task_len', 'sequence_len', 'label'}, map_list_as_tensor=True)
    )
    return batched