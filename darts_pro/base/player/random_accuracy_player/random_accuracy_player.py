from typing import Generic

from ...state import TState
from ...board import DartBoard, Target
from ..player import AbstractPlayer

from .target_probabilities import ProbabilityComputationResult, AimPoints


class AbstractRandomAccuracyPlayer(AbstractPlayer, Generic[TState]):
    def __init__(
        self,
        team_index: int,
        name: str,
        probability_lookups: dict[AimPoints, ProbabilityComputationResult],
    ):
        self._lookups = probability_lookups
        super().__init__(team_index, name)

    def get_outcome_probabilities(
        self, board: DartBoard, intended_target: Target
    ) -> list[tuple[Target, float]]:
        aim_point = AimPoints.SINGLE
        if intended_target.is_bullseye:
            aim_point = AimPoints.BULLSEYE
        elif intended_target.multiplier == 2:
            aim_point = AimPoints.DOUBLE
        elif intended_target.multiplier == 3:
            aim_point = AimPoints.TRIPLE

        lookup = self._lookups[aim_point]
        result = [
            (Target(0, 1, False), lookup.miss_percentage),
        ]

        if len(board.bullseye_targets) == 2:
            double_target = [t for t in board.bullseye_targets if t.multiplier == 2][0]
            single_target = [t for t in board.bullseye_targets if t.multiplier == 1][0]

            bullseye_targets = [
                (single_target, lookup.single_bull_percentage),
                (double_target, lookup.double_bull_percentage),
            ]
        else:
            total_bull_pct = (
                lookup.double_bull_percentage + lookup.single_bull_percentage
            )
            bullseye_targets = [(board.bullseye_targets[0], total_bull_pct)]

        result.extend(bullseye_targets)

        try:
            origin_index_on_board = board.radial_values_order.index(
                intended_target.value
            )
        except ValueError:
            # likely throwing at a bullseye, don't need to shift
            origin_index_on_board = 0
        shifted_order = (
            board.radial_values_order[origin_index_on_board:]
            + board.radial_values_order[:origin_index_on_board]
        )
        for idx, val in enumerate(shifted_order):
            probs = lookup.radial_target_percentages[idx]
            total_single_pct = (
                probs.inner_single_percentage + probs.outer_single_percentage
            )
            result.append((Target(val, 1), total_single_pct))
            result.append((Target(val, 2), probs.double_percentage))
            result.append((Target(val, 3), probs.triple_percentage))
        return result
