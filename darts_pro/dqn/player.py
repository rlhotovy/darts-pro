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


# In training, the training loop handles getting the intended action
# so we can pass that to the player and remove the randomness from the
# game.play_next_throw call.
class TrainingPlayer(AbstractRandomAccuracyPlayer[TState]):
    def __init__(
        self,
        team_index: int,
        name: str,
        probability_lookups: dict[AimPoints, ProbabilityComputationResult],
    ):
        super().__init__(team_index, name, probability_lookups)
        self._action_index = 0

    def compute_intended_target(self, board: DartBoard, _: TState) -> Target:
        return board.indexed_targets[self._action_index]

    def set_action_index(self, action_index: int):
        self._action_index = action_index


# Now meant for inference time; basically a wrapper around Agent
# but handles mapping from action index to Target objects
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
