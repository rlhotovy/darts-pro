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

    def __str__(self) -> str:
        state_dict = dict(
            scores=self.team_scores,
            target_score=self.target_score,
            turn_number=self.turn_number,
            current_throwing_team=self.current_throwing_team,
            current_throwing_player=self.current_throwing_player,
            current_thrower_dart_number=self.current_thrower_dart_number
        )
        return repr(state_dict)

    def to_tensor(self) -> torch.Tensor:
        args = [
            self.current_throwing_team,
            self.target_score,
            self.current_thrower_dart_number,
            self.turn_number,
        ]

        # at some point... probably something like from_tensor where we can track
        # which team score is at which index? Not sure yet if we'll need it
        for team_score in self.team_scores.values():
            args.append(team_score)

        return torch.Tensor(args)
