from __future__ import annotations 
from dataclasses import dataclass
from typing import Type, List

from PIL import Image


@dataclass
class Action:
    name: str


@dataclass
class Task:
    name: str
    description: str
        

@dataclass
class TaskSequenceFrame:
    image: Image
    action: Type[Action]
    
    def __repr__(self) -> str:
        return f"""TaskSequenceFrame(img: {self.image}, action: {self.action.name})"""


@dataclass
class TaskSequence:
    task: Type[Task]
    frames: List[Type[TaskSequenceFrame]]
    
    def __len__(self) -> int:
        return len(self.frames)
    
    def __repr__(self) -> str:
        return f'TaskSequence(task: {self.task.name}, len: {len(self)})'

    def __getitem__(self, idx) -> Type[TaskSequenceFrame]:
        return self.frames[idx]
    
    def subsequence(self, start: int, end: int) -> TaskSequence:
        return TaskSequence(self.task, self.frames[start : end])

