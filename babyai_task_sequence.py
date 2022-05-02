import pickle
import blosc
import torch
from typing import List
from dataclasses import dataclass

from babyai.common import *

from datasets.formats.task_sequence import *


@dataclass
class BabyaiFrame(TaskSequenceFrame):
    image: torch.Tensor
    direction: int
    
    def __repr__(self) -> str:
        return f'BabyaiFrame(img: {self.image.shape}, ' + \
            f'action: {self.action.name}, dir: {DIRECTIONS[self.direction]})'
        

def taskname_from_path(path):
    return path.split('/')[-1].split('_')[0]


def chunknum_from_path(path):
    return int(path.split('/')[-1].split('_')[1].strip('.pkl'))


def numeric_encode_task(task):
    words = task.replace(',', '').split(' ')
    return torch.tensor([VOCAB_TO_INDEX[w] for w in words])


def unpack_images(images):
    unpacked = blosc.unpack_array(images)
    return torch.from_numpy(unpacked)


FEATURE_PROCESS_FN = [
    lambda task: task,
    unpack_images,
    torch.tensor,
    lambda actions: actions,
]

def format_raw_sequence(raw_sequence: List, taskname: str) -> TaskSequence:
    features = [fn(feature) for fn, feature in zip(FEATURE_PROCESS_FN, raw_sequence)]
    features = dict(zip(FEATURE_NAMES, features))
    
    task = Task(taskname, features['task'])
    sequence = []
    for action_enum, image, direction in zip(
        features['actions'], features['images'], features['directions']
    ):
        action = Action(ACTIONS[action_enum.value])
        sequence.append(BabyaiFrame(image, action, direction))
    return TaskSequence(task, sequence)


def load_sequences(path: str) -> List[TaskSequence]:
    sequences = pickle.load(open(path, 'rb'))
    taskname = taskname_from_path(path)
    return [format_raw_sequence(sequence, taskname) for sequence in sequences]


if __name__ == '__main__':
    sequences = load_sequences('data/babyai/task_sequence_chunked/Goto_000.pkl')
    for sequence in sequences:
        print(sequence[1:5])
        print(sequence.frames[0])
        exit()