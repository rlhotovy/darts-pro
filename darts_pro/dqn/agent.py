from typing import Optional
import random
import torch

from .strategy import ExplorationExploitationStrategy
from .network import DQN


class Agent:
    def __init__(
        self,
        strategy: ExplorationExploitationStrategy,
        actions: list[int],
    ):
        self._strategy = strategy
        self._actions = actions
        self._last_action_taken: Optional[int] = None
        self._strategy_decay_step = 0

    def select_action(self, state: torch.Tensor, policy_network: DQN) -> int:
        rate = self._strategy.get_exploration_rate(self._strategy_decay_step)
        if rate > random.random():
            action = random.choice(self._actions)
        else:
            with torch.no_grad():
                action = policy_network(state).argmax(dim=1).item()

        self._last_action_taken = action
        return action

    @property
    def last_action_taken(self) -> Optional[int]:
        return self._last_action_taken

    def increment_strategy_step(self):
        self._strategy_decay_step += 1
