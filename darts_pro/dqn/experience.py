from dataclasses import dataclass
from typing import Union
import random
from abc import ABC, abstractmethod

import torch


@dataclass
class Experience:
    initial_state: torch.Tensor
    action: Union[int, torch.Tensor]
    reward: float
    next_state: torch.Tensor
    next_state_is_final: bool


class AbstractReplayMemory(ABC):
    def __init__(self, capacity: int, initial_memory: list[Experience]) -> None:
        self._capacity = capacity
        self._memory = initial_memory
        self._push_count = 0
    
    @abstractmethod
    def push(self, experience: Experience):
        pass

    def sample(self, batch_size: int) -> list[Experience]:
        return random.sample(self._memory, k=batch_size)

    def can_provide_sample(self, batch_size: int) -> bool:
        return len(self._memory) >= batch_size


class ReplayMemory(AbstractReplayMemory):
    def __init__(self, capacity: int):
        super().__init__(capacity, [])

    def push(self, experience: Experience):
        if len(self._memory) < self._capacity:
            self._memory.append(experience)
        else:
            self._memory[self._push_count] = experience
        self._push_count += 1
        self._push_count = self._push_count % self._capacity


class SeededReplayMemory(AbstractReplayMemory):
    def __init__(self, seeded_experiences: list[Experience], non_seeded_capacity: int):
        total_capacity = len(seeded_experiences) + non_seeded_capacity
        self._seeded_capacity = len(seeded_experiences)
        self._non_seeded_capacity=  non_seeded_capacity
        super().__init__(total_capacity, seeded_experiences)
    
    def push(self, experience: Experience):
        if len(self._memory) < self._capacity:
            self._memory.append(experience)
        else:
            self._memory[self._push_count + self._seeded_capacity] = experience
        self._push_count += 1
        self._push_count = self._push_count % self._non_seeded_capacity
