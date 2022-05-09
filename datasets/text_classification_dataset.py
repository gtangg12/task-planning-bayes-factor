from typing import List, Dict, Callable, TypedDict

from torch.utils.data import Dataset
from tqdm import tqdm


class TextLabelDict(TypedDict):
    text: str
    label: int
    input_ids: List[int]
    attention_mask: List[int]


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

    def __getitem__(self, idx: int) -> TextLabelDict:
        data = { 
            'text': self.texts[idx], 'label': self.labels[idx] 
        }
        # tokenized dict has keys 'input_ids', 'attention_mask'
        data.update(self.tokenized[idx])
        return data