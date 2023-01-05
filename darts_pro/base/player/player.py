from abc import ABC, abstractmethod
from typing import Generic

from typing import Callable, Optional

from ..board import DartBoard, Target
from ..state import TState, ThrowResult


ThrowCallback = Callable[[TState, ThrowResult], None]


class AbstractPlayer(ABC, Generic[TState]):
    def __init__(
        self,
        team_index: int,
        name: str,
        on_throw: Optional[ThrowCallback],
    ):
        self._team_index = team_index
        self._name = name
        self._on_throw = on_throw

    @abstractmethod
    def compute_intended_target(self, board: DartBoard, game_state: TState) -> Target:
        pass

    @abstractmethod
    def get_outcome_probabilities(
        self, board: DartBoard, intended_target: Target
    ) -> list[tuple[Target, float]]:
        pass

    @property
    def on_throw(self) -> Optional[ThrowCallback]:
        return self._on_throw

    @property
    def player_id(self) -> str:
        return f"{self._team_index}__{self._name}"
