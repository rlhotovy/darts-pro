from typing import Union
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
        self._current_step = 0
        self._actions = actions

    def select_action(self, state: torch.Tensor, policy_network: DQN) -> int:
        rate = self._strategy.get_exploration_rate(self._current_step)
        if rate > random.random():
            return random.choice(self._actions)

        with torch.no_grad():
            return policy_network(state).argmax(dim=1).item()
