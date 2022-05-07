import copy
import random
from typing import List, Dict
from dataclasses import asdict

import torch
import torch.nn.functional as F

from babyai.common import *

from datasets.data_collator import collate_list_of_dict
from datasets.task_sequence_dataset import (
    TaskCompletitionDataset, 
    TaskSequenceDict
)
from babyai_task_sequence import (
    BabyaiTaskSequence, 
    numeric_encode_task
)


class BabyaiSequenceDataset(TaskCompletitionDataset):
    """ Dataset for valid babyai task sequence classification. 
    """
    # used to specify classifier embedding dim
    EMBEDDING_DIM = NUM_ACTIONS + NUM_DIRECTIONS
    # length of suffix of actions to corrupt for negative sample
    NEGATIVE_SAMPLE_SUFFIX_LEN = 1

    ENCODED_FEATURE_NAMES = ['images', 'directions', 'actions']
    ENCODED_FEATURE_NUM_CLASSES = [NUM_VIEW_FEATURES, NUM_DIRECTIONS, NUM_ACTIONS]

    @classmethod
    def encode(cls, sequence: BabyaiTaskSequence) -> TaskSequenceDict:
        encoded = {}
        encoded['taskname'] = sequence.task.name
        encoded['task'] = numeric_encode_task(sequence.task.description)
        encoded['task_len'] = len(encoded['task']) 
        encoded['sequence_len'] = len(sequence)

        # convert list frames objects to list of correspoding dicts
        frame_dicts = list(map(asdict, sequence.frames))

        print(frame_dicts)
        exit()

        # collate respective frame features into one dict`
        collated_features = collate_list_of_dict(
            frame_dicts, cls.ENCODED_FEATURE_NAMES, map_list_as_tensor=True
        )  

        # one hot encode features
        for name, num_classes in zip(cls.ENCODED_FEATURE_NAMES, cls.ENCODED_FEATURE_NUM_CLASSES):
            encoded[name] = F.one_hot(collated_features[name], num_classes)
        
        # HWC to CHW (torch nn format)
        encoded['images'].permute(0, 3, 1, 2)  

        # merge directions into actions into tensor of size (sequence_len, EMBEDDING_DIM)
        encoded['actions'] = torch.cat((encoded['actions'], encoded['directions']), dim=1)
        assert(encoded['actions'].shape == (len(sequence), cls.EMBEDDING_DIM))
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


if __name__ == '__main__':
    from babyai_task_sequence import load_sequences
    sequences = load_sequences('data/babyai/task_sequence_chunked/GoTo_000.pkl')
    dataset = BabyaiSequenceDataset(sequences)