import json
from typing import List, Dict, Iterable

import torch

TORCH_PRIMATIVES = (bool, int, float)


def object_to_json(obj: object) -> dict:
    """ Convert an object to a json-serializable dict.
    """
    return json.loads(
        json.dumps(obj, default = lambda x: x.__dict__)
    )


def collate_list_of_dict(batch: List[Dict], keys: Iterable, map_list_as_tensor: bool = False) -> Dict:
    """ Collate a list of dicts into a single dict.
    Args:
        batch: list of dicts
        keys: iterable of keys to collate
        map_list_as_tensor: if True, map each empty list, list of primativies (bool, int, float), or 
            list of tensors to a tensor 
    Returns:
        Collated dict
    """
    collated = {}
    for key in keys:
        collated[key] = [elm[key] for elm in batch]
        if map_list_as_tensor:
            if not len(collated[key]) or isinstance(collated[key][0], TORCH_PRIMATIVES):
                collated[key] = torch.tensor(collated[key])
            if isinstance(collated[key][0], torch.Tensor):
                collated[key] = torch.stack(collated[key])
    return collated