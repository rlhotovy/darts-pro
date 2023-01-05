from abc import ABC, abstractmethod
from typing import Optional, Generic
import random

from collections import OrderedDict

from .player import AbstractPlayer
from .board import DartBoard, Target
from .state import TState, ThrowResult


class AbstractDartsGame(ABC, Generic[TState]):
    def __init__(
        self,
        teams: OrderedDict[int, list[AbstractPlayer[TState]]],
        board: DartBoard,
        starting_turn_number: int = 0,
    ):
        # Assume we have at least one "team zero" and that we have at least one team
        assert 0 in teams.keys()
        assert all([len(t) == len(teams[0]) for t in teams.values()])

        self._teams = teams
        self._board = board
        self._turn_number = starting_turn_number
        self._players_per_team = len(teams[0])
        self._current_throwing_team = 0
        self._current_throwing_player = teams[0][0].player_id
        self._current_player_throw_number = 0

    def _end_player_turn(self):
        current_player_id = self._current_throwing_player
        current_team = self._teams[self._current_throwing_team]
        current_player_idx = [p.player_id for p in current_team].index(
            current_player_id
        )

        self._current_player_throw_number = 0
        if current_player_idx < len(current_team):
            self._current_throwing_player = current_team[
                current_player_idx + 1
            ].player_id
        else:
            next_team_idx = (current_team + 1) % len(self._teams)
            next_team = self._teams[next_team_idx]
            self._current_throwing_team = next_team_idx
            self._current_throwing_player = next_team[0].player_id

    def play_next_turn(self) -> tuple[bool, Optional[int]]:
        # For now, assume we rotate among teams... I can't think of any exmaples
        # where an entire team throws first
        team = self._teams[self._current_throwing_team]
        player = [p for p in team if p.player_id == self._current_throwing_player][0]
        on_throw = player.on_throw or (lambda _, __: None)

        can_throw_again = True
        while can_throw_again:
            intended_target = player.compute_intended_target(self._board, self.state())
            target_probs = player.get_outcome_probabilities(
                self._board, intended_target
            )

            probs = [prob for _, prob in target_probs]
            idxs = list(range(len(probs)))
            targets = [target for target, _ in target_probs]
            hit_index = random.choices(idxs, probs)[0]
            hit = targets[hit_index]

            can_throw_again = self._add_throw_result(
                self._current_throwing_team,
                player,
                hit,
                self._current_player_throw_number,
            )
            done, winner = self._game_is_complete()
            if done:
                on_throw(
                    self.state(),
                    ThrowResult(
                        hit, self._current_player_throw_number, can_throw_again, True
                    ),
                )
                return True, winner
            else:
                on_throw(
                    self.state(),
                    ThrowResult(
                        hit, self._current_player_throw_number, can_throw_again, False
                    ),
                )
                self._current_player_throw_number += 1

        self._end_player_turn()
        return False, None

    def play_to_completion(self) -> Optional[int]:
        done = False
        winner = None
        while True:
            done, winner = self.play_next_turn()
            if done:
                return winner

    @abstractmethod
    def _add_throw_result(
        self,
        team: int,
        player: AbstractPlayer[TState],
        target: Target,
        current_player_dart_number: int,
    ) -> bool:
        """
        Adds the result of an individual throw to the game. Returns a boolean indicating
        whether or not the player's turn continues.
        """
        pass

    @abstractmethod
    def _game_is_complete(self) -> tuple[bool, Optional[int]]:
        pass

    @abstractmethod
    def state(self) -> TState:
        pass
