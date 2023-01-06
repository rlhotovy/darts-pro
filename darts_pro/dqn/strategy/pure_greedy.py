from .strategy import ExplorationExploitationStrategy


class PureGreedyStrategy(ExplorationExploitationStrategy):
    def get_exploration_rate(self, current_time_step: int):
        return 0
