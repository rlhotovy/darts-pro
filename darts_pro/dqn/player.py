from ..base import (
    AbstractRandomAccuracyPlayer,
    TState,
    AimPoints,
    ProbabilityComputationResult,
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
    ):
        self._agent = agent
        self._network = policy_network
        super().__init__(team_index, name, probability_lookups)

    def compute_intended_target(self, board: DartBoard, game_state: TState) -> Target:
        state = game_state.to_tensor()[None]
        action = self._agent.select_action(state, self._network)
        return board.indexed_targets[action]
