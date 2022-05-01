from typing import Callable
from torch.utils.data import Dataset


class TextSequenceClassificationDataset(Dataset):
    def __init__(
        self, 
        texts: list[str], 
        labels: list[int], 
        tokenize_fn: Callable[[str], dict],
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenized = self.map(tokenize_fn, self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        data = { 'label', self.labels[idx], 'text', self.texts[idx] }
        data.update(self.tokenized[idx])
        return data

