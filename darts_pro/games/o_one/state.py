import torch

from ...base import AbstractGameState


class X01GameState(AbstractGameState):
    def __init__(
        self,
        team_scores: dict[int, int],
        target_score: int,
        turn_number: int,
        current_throwing_team: int,
        current_throwing_player: str,
        current_thrower_dart_number: int,
    ):
        self.team_scores = team_scores
        self.target_score = target_score
        super().__init__(
            turn_number,
            current_throwing_team,
            current_throwing_player,
            current_thrower_dart_number,
        )

    def to_tensor(self) -> torch.Tensor:
        # TODO: implement this
        return torch.zeros(10, 10)
