from dataclasses import dataclass
from ...base import AbstractGameState


@dataclass
class X01GameState(AbstractGameState):
    team_scores: dict[int, int]
    target_score: int
