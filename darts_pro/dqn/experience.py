from dataclasses import dataclass
from typing import Union
import random

import torch


@dataclass
class Experience:
    initial_state: torch.Tensor
    action: Union[int, torch.Tensor]
    reward: float
    next_state: torch.Tensor


class ReplayMemory:
    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._memory: list[Experience] = []
        self._push_count = 0

    def push(self, experience: Experience):
        if len(self._memory) < self._capacity:
            self._memory.append(experience)
        else:
            self._memory[self._push_count] = experience
        self._push_count += 1
        self._push_count = self._push_count % self._capacity

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self._memory, k=batch_size)

    def can_provide_sample(self, batch_size: int) -> bool:
        return len(self._memory) >= batch_size
