from abc import ABC
from dataclasses import dataclass
from typing import TypeVar


@dataclass
class AbstractGameState(ABC):
    turn_number: int
    current_thrower_dart_number: int


TState = TypeVar("TState", bound=AbstractGameState)
