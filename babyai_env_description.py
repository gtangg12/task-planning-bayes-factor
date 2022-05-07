import random
import inspect
import itertools
from typing import List, Tuple
from collections import Counter, defaultdict

import torch

from babyai.common import *

from datasets.formats.task_sequence import TaskSequence


# TODO align with babyai infra i.e. minigrid env to refactor and simplify

EGO_ROW = VIEW_SHAPE[0] - 1
EGO_COL = VIEW_SHAPE[1] // 2
EGO = 'E'

EGO_VIEW_ENCODINGS = {
    'cardinal_diagonal': (
        """
        6663777
        6663777
        6663777
        6663777
        6663777
        6660777
        441E255
        """,
        {
            '0': 'Directly front',
            '1': 'Directly left',
            '2': 'Directly right',
            '3': 'Farther front',
            '4': 'Farther left',
            '5': 'Farther right',
            '6': 'Farther front left',
            '7': 'Farther front right'
        }
    )
}


def make_view_partition(view_encoding: str) -> List[Tuple[int, int]]:
    """ Generate view encoding from the perspective of the ego agent.
        Resulting dict: { statement : [(r, c)...] }
    """
    partition_dict = defaultdict(list)

    encoding_labels, label_to_statement = EGO_VIEW_ENCODINGS[view_encoding]

    encoding_labels = inspect.cleandoc(encoding_labels).split('\n')
    n, m = len(encoding_labels), len(encoding_labels[0])
    assert (n, m) == VIEW_SHAPE, 'Region pattern must be same as ego view shape.'

    for (r, c) in itertools.product(range(n), range(m)):
        label = encoding_labels[r][c]
        if label == EGO:
            continue
        statement = label_to_statement[label]
        partition_dict[statement].append((r, c))

    partitions = []
    for _, statement in sorted(list(label_to_statement.items())):
        partitions.append((statement, partition_dict[statement]))
    return partitions


def location_descriptors(image: torch.Tensor, r: int, c: int):
    """ Return object, color, and door descriptors of frame[i, j] """
    x, y, z = image[r, c]
    return OBJECTS[x], COLORS[y], DOOR_STATES[z]


def location_string(image: torch.tensor, r: int, c: int):
    """ Generate description phrase of frame[i, j] """
    object, color, door = location_descriptors(image, r, c)
    if object == 'unseen':
        return 'unseen'
    elif object in ['empty', 'floor']:
        return 'empty'
    elif object == 'door':
        return f'{door} {color} {object}'
    return f'{color} {object}'


def region_description(image: torch.Tensor, region: List[Tuple[int, int]]) -> str:
    """ Helper function to generate a grammatically correct description of a region, 
        a list of coordinate tuples """
    description = []

    # tally entities
    entities = Counter()
    for i, j in region:
        entry = location_string(image, i, j)
        entities[entry] += 1
    if set(entities.keys()).issubset({'unseen', 'empty'}):
        return 'the space is empty.' if 'empty' in entities else 'the space is not visible.'
        
    # generate description, putting walls at the end of the description
    entities_walls_last = sorted(list(entities.items()), key=lambda x: 1 if 'wall' in x else 0)
    for i, (entity, count) in enumerate(entities_walls_last):
        if entity == 'unseen' or entity == 'empty':
            continue
        if count > 1:
            plural = 'es' if 'box' in entity else 's'
            description.append(f'{count} {entity}{plural}')
        else:
            description.append(f'a {entity}')

    # proper grammar
    if len(description) > 1:
        tail = ', '.join(description[:-1]) + ' and ' + description[-1]
    else:
        tail = description[0]
    linking_verb = 'is' if tail[0] == 'a' else 'are'
    return f'{linking_verb} {tail}.'
    

def image_description(image: torch.tensor, view_partition) -> str:
    """ Generate a verbal description of the actor's current state based on view encoding """
    description = ['You are in a room.']

    for statement, region in view_partition:
        description.append(statement)
        description.append(region_description(image, region))
    return ' '.join(description)


class TaskSequencePromptBuilder():
    def __init__(self, sequence: TaskSequence, view_encoding: str) -> None:
        self.sequence = sequence
        self.view_partition = make_view_partition(view_encoding)
        self.inventory_history = self.build_inventory_history(sequence)

    @classmethod
    def build_inventory_history(cls, sequence: TaskSequence):
        """ Build a list of inventory items over time (None denotes empty inventory) """
        inventory_history = []
        item = None
        for frame in sequence.frames:
            action = frame.action.name
            if action == 'pickup':
                item = location_string(frame.image, EGO_ROW - 1, EGO_COL)
            elif action == 'drop':
                item = None
            inventory_history.append(item)
        return inventory_history
  
    def generate_env_description(self, timestamp: int) -> str:
        """ Generate a verbal description of the actor's current state """
        image = self.sequence.frames[timestamp].image
        description = [image_description(image, self.view_partition)]
        inventory_item = self.inventory_history[timestamp]
        if inventory_item:
            description.append(f'You are carrying a {inventory_item}.')
        else:
            description.append('You are not carrying anything.')
        return ' '.join(description)
    
    def get_action_taken(self, timestamp: int) -> str:
        """ Return the action taken by the actor given the current state """
        return self.sequence.frames[timestamp].action.name


def generate_env_description(sequence: TaskSequence, view_encoding='cardinal_diagonal'):
    """ Generate a sample of textual descriptions from a sequence """    
    timestamp = random.randint(0, len(sequence) - 1)
    
    generator = TaskSequencePromptBuilder(sequence, view_encoding)

    prompt = generator.generate_env_description(timestamp)
    action = generator.get_action_taken(timestamp)
    return prompt, action