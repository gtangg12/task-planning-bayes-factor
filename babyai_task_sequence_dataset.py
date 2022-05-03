import copy
import random
from typing import TypedDict, List, Dict
from dataclasses import asdict

import torch
from torch.nn.utils.rnn import pad_sequence
from babyai.common import *

from datasets.formats.task_sequence import TaskSequence
from datasets.task_sequence_dataset import TaskCompletitionDataset
from datasets.data_collator import collate_list_of_dict
from babyai_task_sequence import numeric_encode_task


NEGATIVE_SAMPLE_SUFFIX_LEN = 1


class BabyaiSequenceDict(TypedDict):
    taskname     : str
    task         : torch.Tensor
    images       : torch.Tensor
    actions      : torch.Tensor
    directions   : torch.Tensor
    sequence_len : int


class BabyaiSequenceDataset(TaskCompletitionDataset):
    """ Dataset for valid babyai task sequence classification. 
    """
    # used to specify classifier embedding dim
    SEQUENCE_HISTORY_EMBEDDING_DIM = 128

    @classmethod
    def encode(cls, sequence: TaskSequence) -> BabyaiSequenceDict:
        encoded = {}
        sequence_len = len(sequence)
        sequence = asdict(sequence)
        
        encoded['taskname'] = sequence['task']['name']
        encoded['task'] = numeric_encode_task(sequence['task']['description'])
        encoded['sequence_len'] = sequence_len
        for feature_name in ['images', 'actions', 'directions']:
            encoded[feature_name] = \
                collate_list_of_dict(sequence['frames'][feature_name], map_list_as_tensor=True)   
        encoded['images'].permute(0, 3, 1, 2)  # HWC to CHW

        # embedding_dim = cls.SEQUENCE_HISTORY_EMBEDDING_DIM
        # tensor passed into model
        encoded['sequence_history'] = 

        return encoded

    @classmethod
    def negative_sample(cls, encoded: Dict) -> Dict:
        sequence_len = encoded['sequence_len'] 
        suffix_begin = sequence_len - NEGATIVE_SAMPLE_SUFFIX_LEN

        resampled_encoded = copy.deepcopy(encoded)
        resampled_actions = torch.randint(0, NUM_ACTIONS, (NEGATIVE_SAMPLE_SUFFIX_LEN,))
        # Guarantee at least one different action in new sequence
        idx = random.randint(suffix_begin, sequence_len)
        while resampled_actions[idx] == encoded['actions'][idx]:
            resampled_actions[idx] = random.randint(0, NUM_ACTIONS - 1)
        resampled_encoded['actions'][suffix_begin:] = resampled_actions
        
        return resampled_encoded


def collate_fn(batch: List[BabyaiSequenceDict]) -> BabyaiSequenceDict:
    batched = {}
    for name in BabyaiSequenceDict.__annotations__.keys():
        batched[name] = collate_list_of_dict(batch, {name})
        if name not in ['taskname', 'sequence_len']: 
            batched[name] = pad_sequence(batched[name])
    return batched
