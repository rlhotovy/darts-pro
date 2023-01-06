import math

from .strategy import ExplorationExploitationStrategy


class EpsilonGreedyStrategy(ExplorationExploitationStrategy):
    def __init__(self, start: float, end: float, decay: float):
        self._start = start
        self._end = end
        self._decay = decay

    def get_exploration_rate(self, current_time_step: int):
        return self._end - (self._end - self._start) * math.exp(
            -1.0 * current_time_step * self._decay
        )
