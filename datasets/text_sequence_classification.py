import pickle
import glob
from typing import List, Dict, Tuple, Callable
from torch.utils.data import Dataset


class TextSequenceClassificationDataset(Dataset):
    def __init__(
        self, 
        texts: List[str], 
        labels: List[int], 
        tokenize_fn: Callable[[str], Dict],
    ) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenized = self.map(tokenize_fn, self.texts)

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict:
        data = { 'label', self.labels[idx], 'text', self.texts[idx] }
        data.update(self.tokenized[idx])
        return data


def load_from_dir(path: str, filter_fn: Callable) -> Tuple[List[str], List[int]]:
    filenames = glob.glob(path + '/*.pkl')
    filenames = list(filter(filter_fn, filenames))
    print(filenames)
    exit()
    
    texts, labels = [], []
    
    for filename in filenames:
        with open(filename, 'rb') as f:
            texts_, labels_ = pickle.load(f)
        texts.extend(texts_)
        labels.extend(labels_)

    return texts, labels
    