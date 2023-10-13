from typing import Optional, Callable
from dataclasses import dataclass, field

import torch
import numpy as np

from ...base import DartBoard, compute_probability_lookup
from ...dqn import (
    DQN,
    AbstractReplayMemory,
    DQNTrainingLoop,
    TrainingLoopConfig,
    StepResult,
    TrainingPlayer,
    EpisodeEndPayload,
)
from .game import X01Game


X01GameInitializer = Callable[[int], tuple[dict[int, int], int]]


@dataclass
class X01TrainingConfig:
    accuracy_sigma_x: float = 0.25
    accuracy_sigma_y: float = 0.25
    max_target_score: int = 301
    max_turns: int = 30
    min_win_turns: int = 3
    darts_per_player: int = 3
    agent_throws_between_update: int = 10

    loop_config: TrainingLoopConfig = field(default_factory=TrainingLoopConfig)


class X01TrainingLoop(DQNTrainingLoop):
    def __init__(
        self,
        n_episodes: int,
        policy_network: DQN,
        target_network: DQN,
        config: X01TrainingConfig,
        memory: Optional[AbstractReplayMemory] = None,
        board: Optional[DartBoard] = None,
        initialize_game: Optional[X01GameInitializer] = None,
    ):
        self._board = board or DartBoard.get_default_dartboard(False)
        self._init_game = initialize_game or self._get_default_start
        self._total_agent_throws = 0
        all_actions = list(board.indexed_targets.keys())
        super().__init__(
            n_episodes,
            all_actions,
            policy_network,
            target_network,
            memory,
            config.loop_config,
        )
        self.add_on_episode_end_callback(self._on_episode_end)
        self._config = config

        prob_lookup = compute_probability_lookup(
            config.accuracy_sigma_x, config.accuracy_sigma_y, len(board.radial_targets)
        )

        self._player = TrainingPlayer(0, "agent", prob_lookup)
        team_one = [self._player]
        self._teams = {0: team_one}

        self._reset_game(0)

    def _get_default_start(self, _: int) -> tuple[dict[int, int], int]:
        return {0: self._config.max_target_score}, 0

    def _reset_game(self, episode_number: int):
        starting_scores, turn_number = self._init_game(episode_number)
        self._game = X01Game(
            self._teams,
            self._board,
            self._config.max_target_score,
            self._config.max_turns,
            starting_scores,
            turn_number,
            self._config.darts_per_player,
        )
        self._total_agent_throws = 0

    def _current_state(self) -> torch.Tensor:
        return self._game.state().to_tensor()

    def _should_increment_agent_strategy(self) -> bool:
        if (
            self._total_agent_throws > 0
            and self._total_agent_throws % self._config.agent_throws_between_update == 0
        ):
            self._total_agent_throws = 0
            return True
        return False

    def _step(self, action: int) -> StepResult:
        reward = 0
        pre_throw_state = self._game.state()
        self._player.set_action_index(action)
        throw_result, winner = self._game.play_next_throw()
        self._total_agent_throws += 1
        done = throw_result.ended_game
        post_throw_state = self._game.state()

        if throw_result.ended_game:
            reward = (
                self._compute_winner_reward(
                    pre_throw_state.turn_number,
                    self._config.max_turns,
                    self._config.min_win_turns,
                )
                if winner is not None
                else -1.0
            )

        return [post_throw_state.to_tensor(), reward, done, {}]

    def _on_episode_end(self, payload: EpisodeEndPayload):
        # print(
        #     f"Finished game {payload.episode_number}. Final score {payload.final_state_tensor[0]}. Total Reward {payload.reward}"
        # )
        self._reset_game(payload.episode_number)

    # def _compute_winner_reward(self, turn_number, max_turns, min_win_turns) -> float:
    #     return 1 - ((turn_number - min_win_turns) / (max_turns - min_win_turns))

    def _compute_winner_reward(self, turn_number, max_turns, min_win_turns) -> float:
        dist = turn_number - min_win_turns
        return np.power(0.95, dist)
