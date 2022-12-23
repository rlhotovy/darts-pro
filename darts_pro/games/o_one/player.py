from collections import defaultdict

from ...base import AbstractRandomAccuracyPlayer, DartBoard, Target
from .state import X01GameState


class RandomX01Player(AbstractRandomAccuracyPlayer[X01GameState]):
    def compute_intended_target(
        self, board: DartBoard, game_state: X01GameState
    ) -> Target:
        current_team_score = game_state.team_scores[self._team_index]

        numbers_we_can_score: dict[int, list[Target]] = defaultdict(list)
        largest_bull = sorted(
            board.bullseye_targets, key=lambda t: t.value, reverse=True
        )[0]
        for bull in board.bullseye_targets:
            total = bull.value * bull.multiplier
            numbers_we_can_score[total].append(bull)
        for radial_target in board.radial_targets.values():
            for target in radial_target:
                total = target.value * target.multiplier
                numbers_we_can_score[total].append(target)

        largest_we_can_score = max(numbers_we_can_score.keys())

        # for now, don't worry about going out with multiple darts.
        if current_team_score > largest_we_can_score:
            return largest_bull

        if current_team_score in set(numbers_we_can_score.keys()):
            return numbers_we_can_score[current_team_score][0]

        next_best = max(
            [v for v in numbers_we_can_score.keys() if v < current_team_score]
        )
        return numbers_we_can_score[next_best][0]
