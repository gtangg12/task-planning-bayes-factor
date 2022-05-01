import random
from torch.utils.data import Dataset
from formats.task_sequence import TaskSequence


class TaskSequenceDataset(Dataset):
    def __init__(self, sequences: list[TaskSequence]) -> None:
        self.sequences = list(map(self.encode, sequences))

    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> dict:
        return self.sequences[idx]

    @classmethod
    def encode(cls, sequence: TaskSequence) -> dict:
        raise NotImplementedError


class TaskCompletitionDataset(TaskSequenceDataset):
    def __init__(self, sequences: list[TaskSequence], negative_sample_rate: float = 0.5) -> None:
        super().__init__(sequences)
        self.negative_sample_rate = negative_sample_rate  


    def __getitem__(self, idx: int) -> dict:
        sequence = self.sequences[idx]
        sequence['label'] = True
        if random.random() > self.negative_sample_rate:
            sequence = self.negative_sample(sequence)
            sequence['label'] = False
        return sequence
    
    @classmethod
    def negative_sample(cls, sequence: dict) -> dict:
       raise NotImplementedError