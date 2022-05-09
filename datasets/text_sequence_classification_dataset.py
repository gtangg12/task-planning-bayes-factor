import pickle
import glob
import random
from typing import List, Dict, Callable

from torch.utils.data import Dataset
from tqdm import tqdm


class TextSequenceClassificationDataset(Dataset):
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenize_fn: Callable[[str], Dict],
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenized = list(map(tokenize_fn, tqdm(self.texts)))

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        data = { 'label': self.labels[idx], 'text': self.texts[idx] }
        data.update(self.tokenized[idx])
        return data