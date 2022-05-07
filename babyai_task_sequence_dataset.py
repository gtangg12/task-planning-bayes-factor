import copy
import random
from typing import List, Dict
from dataclasses import asdict

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from babyai.common import *

from datasets.formats.task_sequence import TaskSequence
from datasets.task_sequence_dataset import TaskCompletitionDataset, TaskSequenceDict
from datasets.data_collator import collate_list_of_dict
from babyai_task_sequence import numeric_encode_task


class BabyaiSequenceDataset(TaskCompletitionDataset):
    """ Dataset for valid babyai task sequence classification. 
    """
    # used to specify classifier embedding dim
    EMBEDDING_DIM = NUM_ACTIONS + NUM_DIRECTIONS
    # length of suffix of actions to corrupt for negative sample
    NEGATIVE_SAMPLE_SUFFIX_LEN = 1

    @classmethod
    def encode(cls, sequence: TaskSequence) -> TaskSequenceDict:
        encoded = {}
        sequence_len = len(sequence)
        sequence = asdict(sequence)
        
        encoded['taskname'] = sequence['task']['name']
        encoded['task'] = numeric_encode_task(sequence['task']['description'])
        encoded['task_len'] = len(encoded['task']) # sequence is str, encoded['task] is split
        encoded['sequence_len'] = sequence_len
        for feature_name, num_classes in zip(
            FEATURE_NAMES[1:], 
            [NUM_VIEW_FEATURES, NUM_ACTIONS, NUM_DIRECTIONS]
        ):
            collated_feature = collate_list_of_dict(sequence['frames'][feature_name], map_list_as_tensor=True)   
            encoded[feature_name] = F.one_hot(collated_feature, num_classes)

        # HWC to CHW (torch nn format)
        encoded['images'].permute(0, 3, 1, 2)  

        # merge directions into actions into tensor of size (sequence_len, EMBEDDING_DIM)
        encoded['actions'] = torch.cat((encoded['actions'], encoded['directions']), dim=1)
        assert(encoded['actions'].shape == (sequence_len, cls.EMBEDDING_DIM))
        encoded.pop('directions')

        return TaskSequenceDict(encoded)

    @classmethod
    def negative_sample(cls, encoded: TaskSequenceDict) -> TaskSequenceDict:
        sequence_len = encoded['sequence_len'] 
        suffix_begin = sequence_len - cls.NEGATIVE_SAMPLE_SUFFIX_LEN
        resampled_encoded = copy.deepcopy(encoded)
        resampled_actions = torch.randint(0, NUM_ACTIONS, (cls.NEGATIVE_SAMPLE_SUFFIX_LEN,))

        # Guarantee at least one different action in new sequence
        idx = random.randint(suffix_begin, sequence_len)
        while resampled_actions[idx] == encoded['actions'][idx]:
            resampled_actions[idx] = random.randint(0, NUM_ACTIONS - 1)

        # Replace suffix with corrupted actions
        resampled_encoded['actions'][suffix_begin:] = resampled_actions
        return resampled_encoded


def collate_fn(batch: List[TaskSequenceDict]) -> Dict:
    batched = { 
        'task_len': [], 'sequence_len': [] 
    }
    for name in TaskSequenceDict.__annotations__.keys():
        batched[name] = collate_list_of_dict(batch, {name})
        if name != 'taskname': 
            batched[name] = pad_sequence(batched[name], batch_first=True)
        batched['task_len'].append(batch['task'].shape[0])
        batched['sequence_len'].append(batch['images'].shape[0])
    return batched


if __name__ == '__main__':
    pass