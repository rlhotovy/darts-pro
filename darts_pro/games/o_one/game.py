from collections import OrderedDict
from typing import Optional

from ...base import AbstractDartsGame, AbstractPlayer, DartBoard, Target
from .state import X01GameState


class X01Game(AbstractDartsGame[X01GameState]):
    def __init__(
        self,
        teams: OrderedDict[int, list[AbstractPlayer]],
        board: DartBoard,
        target_score: int = 301,
        max_turns: int = 50,
        starting_scores: Optional[dict[int, int]] = None,
        starting_turn: int = 0,
        darts_per_player: int = 3,
    ):
        if starting_scores is not None:
            scores = starting_scores
        else:
            scores = {}
            for team in teams.keys():
                scores[team] = target_score

        self._target_score = target_score
        self._scores = scores
        self._max_turns = max_turns
        self._darts_per_player = darts_per_player
        super().__init__(teams, board, starting_turn)

    def _add_throw_result(
        self,
        team: int,
        player: AbstractPlayer[X01GameState],
        target: Target,
        current_player_dart_number: int,
    ) -> bool:
        current_team_score = self._scores[team]
        throw_total = target.value * target.multiplier
        if throw_total <= current_team_score:
            self._scores[team] -= throw_total
        return (
            throw_total <= current_team_score
            and current_player_dart_number < self._darts_per_player
        )

    def _game_is_complete(self) -> tuple[bool, Optional[int]]:
        min_score, teams_with_score = self._get_min_score_and_teams_with_score()
        if self._turn_number >= self._max_turns:
            if len(self._teams) == 1:
                return True, None
            elif len(teams_with_score) > 1:
                return True, None
            else:
                return True, teams_with_score[0]
        elif min_score == 0:
            # Assumption here both teams won't have zero... that _should_ never happen
            return True, teams_with_score[0]
        else:
            return False, None

    def _get_min_score_and_teams_with_score(self) -> tuple[int, list[int]]:
        min_score = self._target_score
        teams_with_min = []
        for team, score in self._scores.items():
            if score == min_score:
                teams_with_min.append(team)
            elif score < min_score:
                min_score = score
                teams_with_min = [team]

        return min_score, teams_with_min

    def state(self) -> X01GameState:
        return X01GameState(
            dict(self._scores),
            self._target_score,
            self._turn_number,
            self._current_throwing_team,
            self._current_throwing_player,
            self._current_player_throw_number,
        )
