from abc import ABC, abstractmethod


class ExplorationExploitationStrategy(ABC):
    @abstractmethod
    def get_exploration_rate(self, current_time_step: int) -> float:
        pass
