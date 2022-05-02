import json
from typing import Type, List, Dict, Set
import torch


def object_to_json(obj: object) -> dict:
    return json.loads(
        json.dumps(obj, default = lambda x: x.__dict__)
    )


def collate_list_of_dict(batch: List[Type[Dict]], keys: Set, map_list_as_tensor: bool = False) -> Dict:
    collated = {}
    for key in keys:
        collated[key] = [elm[key] for elm in batch]
        if map_list_as_tensor:
            collated[key] = torch.tensor(collated[key])
    return collated