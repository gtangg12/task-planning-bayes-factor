import pickle
import glob
from typing import List, Dict, Tuple, Callable
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


def load_sequences_from_dir(path: str, filter_fn: Callable) -> Tuple[List[str], List[int]]:
    filenames = glob.glob(path + '/*.pkl')
    filenames = list(filter(filter_fn, filenames))

    texts, labels = [], []

    for filename in filenames:
        with open(filename, 'rb') as f:
            inputs = pickle.load(f)
            texts_, labels_ = list(zip(*inputs))
        texts.extend(texts_)
        labels.extend(labels_)

    return texts, labels
    