from abc import ABC, abstractmethod
from typing import Optional, Generic
import random

from collections import OrderedDict

from .player import AbstractPlayer
from .board import DartBoard, Target
from .state import TState


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
        self._current_player_throw_number = 0

    def play_turn(self) -> tuple[bool, Optional[int]]:
        # For now, assume we rotate among teams... I can't think of any exmaples
        # where an entire team throws first
        for player_idx in range(self._players_per_team):
            for team_idx in self._teams.keys():
                player = self._teams[team_idx][player_idx]

                self._current_player_throw_number = 0
                can_throw_again = True
                while can_throw_again:
                    intended_target = player.compute_intended_target(
                        self._board, self.state()
                    )
                    target_probs = player.get_outcome_probabilities(
                        self._board, intended_target
                    )

                    probs = [prob for _, prob in target_probs]
                    idxs = list(range(len(probs)))
                    targets = [target for target, _ in target_probs]
                    hit_index = random.choices(idxs, probs)[0]
                    hit = targets[hit_index]

                    can_throw_again = self._add_throw_result(
                        team_idx, player, hit, self._current_player_throw_number
                    )
                    self._current_player_throw_number += 1
                    done, winner = self._game_is_complete()
                    if done:
                        return True, winner
        self._turn_number += 1

        # Checking for hitting turn limits essentially
        done, winner = self._game_is_complete()
        if done:
            return True, winner

        return False, None

    def play_to_completion(self) -> Optional[int]:
        done = False
        winner = None
        while True:
            done, winner = self.play_turn()
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
