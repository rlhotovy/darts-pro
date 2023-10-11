from abc import ABC, abstractmethod
from typing import Callable, Optional
from dataclasses import dataclass
from functools import partial

import torch
import numpy as np

from .agent import Agent
from .network import DQN
from .experience import ReplayMemory, Experience
from .q_values import QValues
from .strategy import EpsilonGreedyStrategy


# [new_state, reward, done, info(?)]
StepResult = tuple[torch.Tensor, float, bool, dict]


@dataclass
class EpisodeEndPayload:
    episode_number: int
    final_state_tensor: torch.Tensor
    total_epsisode_reward: float
    n_steps: int


@dataclass
class CheckpointConfig:
    episodes_between_checkpoints: int = 10
    checkpoint_directory: Optional[str] = None


@dataclass
class TrainingLoopConfig:
    train_batch_size: int = 256
    gamma: float = 0.999
    target_net_update: int = 10
    learning_rate: float = 1e-3
    # Note: Not used if memory is provided to loop constructor
    memory_capacity: int = 100000

    # agent config
    start_exploration_rate: float = 1.0
    end_exploration_rate: float = 0.1
    exploration_rate_decay: float = 0.001

    checkpoint_config: Optional[CheckpointConfig] = None


class DQNTrainingLoop(ABC):
    def __init__(
        self,
        n_episodes: int,
        agent_actions: list[int],
        policy_network: DQN,
        target_network: DQN,
        memory: Optional[ReplayMemory],
        config: TrainingLoopConfig,
    ):
        self._n_episodes = n_episodes
        strategy = EpsilonGreedyStrategy(
            config.start_exploration_rate,
            config.end_exploration_rate,
            config.exploration_rate_decay,
        )
        self._agent = Agent(strategy, agent_actions)
        self._policy_network = policy_network
        self._target_network = target_network
        self._target_net_update = config.target_net_update

        if memory is not None:
            self._memory = memory
        else:
            self._memory = ReplayMemory(config.memory_capacity)

        self._on_episode_end_callacks: list[Callable[[EpisodeEndPayload], None]] = []
        self._train_batch_size = config.train_batch_size
        self._gamma = config.gamma
        self._optimizer = torch.optim.Adam(
            params=self._policy_network.parameters(), lr=config.learning_rate
        )

        if config.checkpoint_config is not None:
            self.add_on_episode_end_callback(
                partial(
                    self._checkpoint,
                    config.checkpoint_config.episodes_between_checkpoints,
                    config.checkpoint_config.checkpoint_directory,
                )
            )

    def run(self):
        for episode in range(self._n_episodes):
            self._play_episode(episode)

    def add_on_episode_end_callback(
        self, callback: Callable[[EpisodeEndPayload], None]
    ):
        self._on_episode_end_callacks.append(callback)

    @abstractmethod
    def _current_state(self) -> torch.Tensor:
        pass

    @abstractmethod
    def _step(self, action: int) -> StepResult:
        pass

    @abstractmethod
    def _should_increment_agent_strategy(self) -> bool:
        pass

    def _play_episode(self, epsiode_number: int):
        done = False
        episode_reward = 0.0
        steps = 0
        while not done:
            pre_step_state_tensor = self._current_state()
            action = self._agent.select_action(
                pre_step_state_tensor, self._policy_network
            )
            next_state, reward, done, _ = self._step(action)
            steps += 1
            episode_reward += reward
            if self._should_increment_agent_strategy():
                self._agent.increment_strategy_step()

            experience = Experience(
                pre_step_state_tensor, action, reward, next_state, done
            )
            self._memory.push(experience)

            if self._memory.can_provide_sample(self._train_batch_size):
                experiences = self._memory.sample(self._train_batch_size)
                (
                    states,
                    actions,
                    rewards,
                    next_states,
                    next_is_final,
                ) = self._extract_tensors(experiences)

                current_q_values = QValues.get_current(
                    self._policy_network, states, actions
                )
                next_q_values = QValues.get_next(
                    self._target_network, next_states, next_is_final
                )
                target_q_values = (next_q_values * self._gamma) + rewards

                # Um... check this... I had next_q_values.unsqueeze here... why?
                loss = torch.nn.functional.mse_loss(
                    current_q_values, target_q_values.unsqueeze(1)
                )
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

        if epsiode_number % self._target_net_update == 0:
            self._target_network.load_state_dict(self._policy_network.state_dict())

        for callback in self._on_episode_end_callacks:
            callback(
                EpisodeEndPayload(
                    epsiode_number, self._current_state(), episode_reward, steps
                )
            )

    def _extract_tensors(
        self, experiences: list[Experience]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.stack([e.initial_state for e in experiences])
        actions = torch.from_numpy(
            np.array([e.action for e in experiences], dtype=np.int64)
        )
        rewards = torch.Tensor([e.reward for e in experiences])
        next_states = torch.stack([e.next_state for e in experiences])
        next_state_is_final = torch.from_numpy(
            np.array([e.next_state_is_final for e in experiences], dtype=np.bool_)
        )

        return states, actions, rewards, next_states, next_state_is_final

    def _checkpoint(
        self,
        checkpoint_frequency: int,
        checkpoint_dir: Optional[str],
        payload: EpisodeEndPayload,
    ):
        if (
            payload.episode_number > 0
            and payload.episode_number % checkpoint_frequency == 0
        ):
            prefix = checkpoint_dir or ""
            ckpt_path = f"{prefix}/episode_{payload.episode_number}.pt"
            torch.save(self._policy_network.state_dict(), ckpt_path)
