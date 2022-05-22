import copy
import random
from typing import List, Dict
from dataclasses import asdict

import torch
import torch.nn.functional as F

from babyai.common import *

from datasets.task_sequence_dataset import (
    TaskCompletitionDataset, 
    TaskCompletitionDict
)
from babyai_task_sequence import BabyaiTaskSequence


def encode_babyai_task(task):
    """ Numerically encode the words of task str """
    words = task.replace(',', '').split(' ')
    words = torch.tensor([VOCAB_TO_INDEX[w] for w in words])
    return F.one_hot(words, VOCAB_SIZE)


def encode_babyai_images(images):
    """ One-hot encode the image sequence's channels separately and concat them """
    channels = list(torch.split(images, 1, dim=3))
    for i, c in enumerate(channels):
        channels[i] = F.one_hot(c.squeeze(dim=3).long(), NUM_VIEW_FEATURES_LIST[i])
    encoded_images = torch.cat(channels, dim=3)
    # HWC to CHW (torch nn format)
    return torch.permute(encoded_images, (0, 3, 1, 2))


class BabyaiSequenceDataset(TaskCompletitionDataset):
    """ Dataset for valid babyai task sequence classification """
    # used to specify classifier embedding dim
    EMBEDDING_DIM = NUM_ACTIONS + NUM_DIRECTIONS
    # length of suffix of actions to corrupt for negative sample
    NEGATIVE_SAMPLE_SUFFIX_LEN = 1

    @classmethod
    def encode(cls, sequence: BabyaiTaskSequence) -> TaskCompletitionDict:
        encoded = {}
        encoded['taskname'] = sequence.task.name
        encoded['task'] = encode_babyai_task(sequence.task.description)
        encoded['task_len'] = len(encoded['task']) 
        encoded['sequence_len'] = len(sequence)

        # aggregate features
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

        # valid task sequence
        encoded['label'] = 1

        return TaskCompletitionDict(encoded)

    @classmethod
    def negative_sample(cls, encoded: TaskCompletitionDict) -> TaskCompletitionDict:
        n_resampled = cls.NEGATIVE_SAMPLE_SUFFIX_LEN
        sequence_len = encoded['sequence_len'] 
        suffix_begin = sequence_len - n_resampled
        resampled_encoded = copy.deepcopy(encoded)
        resampled_actions = torch.randint(0, NUM_ACTIONS, (n_resampled,))

        # guarantee at least one different action in resampled actions
        idx = random.randint(0, n_resampled - 1)
        while resampled_encoded['actions'][suffix_begin + idx, resampled_actions[idx].item()] == 1:
            resampled_actions[idx] = random.randint(0, NUM_ACTIONS - 1)

        # one hot encode resampled actions
        resampled_actions_encoded = F.one_hot(resampled_actions, num_classes=NUM_ACTIONS)
        # set actions component of babyai action tensor to resampled actions
        resampled_encoded['actions'][suffix_begin:, :NUM_ACTIONS] = resampled_actions_encoded

        # not valid sequence anymore
        resampled_encoded['label'] = 0

        return resampled_encoded


if __name__ == '__main__':
    from babyai_task_sequence import load_sequences
    sequences = load_sequences('data/babyai/task_sequence_chunked/GoTo_000.pkl')
    dataset = BabyaiSequenceDataset(sequences)

    x = dataset[0]
    print(x['actions'].shape)