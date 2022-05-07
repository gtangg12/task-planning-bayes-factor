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
from babyai_task_sequence import BabyaiTaskSequence


def encode_babyai_task(task):
    """ Numerically encode the words of task str """
    words = task.replace(',', '').split(' ')
    return torch.tensor([VOCAB_TO_INDEX[w] for w in words])


def encode_babyai_images(images):
    """ One-hot encode the image sequence's channels separately and concat them """
    channels = list(torch.split(images, 1, dim=-1))
    for i, c in enumerate(channels):
        channels[i] = F.one_hot(c.squeeze().long(), NUM_VIEW_FEATURES_LIST[i])
    encoded_images = torch.cat(channels, dim=-1)
    # HWC to CHW (torch nn format)
    return torch.permute(encoded_images, (0, 3, 1, 2))


class BabyaiSequenceDataset(TaskCompletitionDataset):
    """ Dataset for valid babyai task sequence classification """
    # used to specify classifier embedding dim
    EMBEDDING_DIM = NUM_ACTIONS + NUM_DIRECTIONS
    # length of suffix of actions to corrupt for negative sample
    NEGATIVE_SAMPLE_SUFFIX_LEN = 1

    @classmethod
    def encode(cls, sequence: BabyaiTaskSequence) -> TaskSequenceDict:
        encoded = {}
        encoded['taskname'] = sequence.task.name
        encoded['task'] = encode_babyai_task(sequence.task.description)
        encoded['task_len'] = len(encoded['task']) 
        encoded['sequence_len'] = len(sequence)

        images, directions, actions = [], [], []
        for frame in sequence:
            images.append(frame.image)
            directions.append(frame.direction)
            actions.append(ACTIONS_TO_INDEX[frame.action.name])
        
        # one hot encode features
        encoded['directions'] = F.one_hot(torch.tensor(directions), NUM_DIRECTIONS)
        encoded['actions'] = F.one_hot(torch.tensor(actions), NUM_ACTIONS)
        encoded['images'] = encode_babyai_images(torch.stack(images))

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