from typing import Optional

from ..base import (
    AbstractRandomAccuracyPlayer,
    TState,
    AimPoints,
    ProbabilityComputationResult,
    ThrowCallback,
    DartBoard,
    Target,
)
from .agent import Agent
from .network import DQN


class AgentPlayer(AbstractRandomAccuracyPlayer[TState]):
    def __init__(
        self,
        agent: Agent,
        policy_network: DQN,
        team_index: int,
        name,
        probability_lookups: dict[AimPoints, ProbabilityComputationResult],
        on_throw: Optional[ThrowCallback],
    ):
        self._agent = agent
        self._network = policy_network
        super().__init__(team_index, name, probability_lookups, on_throw)

    def compute_intended_target(self, board: DartBoard, game_state: TState) -> Target:
        action = self._agent.select_action(game_state.to_tensor(), self._network)
        return board.indexed_targets[action]
