import pickle
import glob
import random
from typing import List, Dict, Tuple, Callable, Optional

from torch.utils.data import Dataset
from tqdm import tqdm


Texts = List[str]
NumericLabels = List[int]


class TextSequenceClassificationDataset(Dataset):
    def __init__(
        self, 
        texts: Texts, 
        labels: NumericLabels, 
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


def load_text_sequences_from_dir(
    path: str, 
    shuffle: bool = True, 
    num_data: Optional[int] = None,
    filter_fn: Optional[Callable] = None) -> Tuple[Texts, NumericLabels]:

    filenames = glob.glob(path + '/*.pkl')
    if filter_fn:
        filenames = list(filter(filter_fn, filenames))

    texts, labels = [], []
    for filename in filenames:
        with open(filename, 'rb') as f:
            inputs = pickle.load(f)
            texts_, labels_ = list(zip(*inputs))
        texts.extend(texts_)
        labels.extend(labels_)
    
    if shuffle:
        random.shuffle(texts)
        random.shuffle(labels)
    if num_data:
        texts, labels = texts[:num_data], labels[:num_data]
        
    return texts, labels
    