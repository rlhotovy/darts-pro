from abc import ABC, abstractmethod
from typing import TypeVar
from dataclasses import dataclass

import torch

from .board import Target


@dataclass
class ThrowResult:
    target_hit: Target
    throw_number: int
    ended_turn: bool
    ended_game: bool


class AbstractGameState(ABC):
    def __init__(
        self,
        turn_number: int,
        current_throwing_team: int,
        current_throwing_player: str,
        current_thrower_dart_number: int,
    ):
        self.turn_number = turn_number
        self.current_throwing_team = current_throwing_team
        self.current_throwing_player = current_throwing_player
        self.current_thrower_dart_number = current_thrower_dart_number

    @abstractmethod
    def to_tensor(self) -> torch.Tensor:
        pass


TState = TypeVar("TState", bound=AbstractGameState)
