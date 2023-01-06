from abc import ABC, abstractmethod
from typing import Generic

from ..board import DartBoard, Target
from ..state import TState


class AbstractPlayer(ABC, Generic[TState]):
    def __init__(self, team_index: int, name: str):
        self._team_index = team_index
        self._name = name

    @abstractmethod
    def compute_intended_target(self, board: DartBoard, game_state: TState) -> Target:
        pass

    @abstractmethod
    def get_outcome_probabilities(
        self, board: DartBoard, intended_target: Target
    ) -> list[tuple[Target, float]]:
        pass

    @property
    def player_id(self) -> str:
        return f"{self._team_index}__{self._name}"
