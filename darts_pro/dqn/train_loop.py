from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Callable

import torch
import numpy as np

from .agent import Agent
from .network import DQN
from .experience import ReplayMemory, Experience
from .q_values import QValues

TLoopState = TypeVar("TLoopState")

# [new_state, reward, done, info(?)]
StepResult = tuple[TLoopState, float, bool, dict]
# [episode_number, final_state]
Callback = Callable[[int, TLoopState], None]


class DQNTrainingLoop(ABC, Generic[TLoopState]):
    def __init__(
        self,
        n_episodes: int,
        agent: Agent,
        policy_network: DQN,
        target_network: DQN,
        memory: ReplayMemory,
        initial_state: TLoopState,
        optimizer,
        train_batch_size: int = 16,
        gamma: float = 0.9999,
    ):
        self._n_episodes = n_episodes
        self._current_state = initial_state
        self._agent = agent
        self._policy_network = policy_network
        self._target_network = target_network
        self._memory = memory
        self._optimizer = optimizer

        self._on_episode_end_callacks: list[Callback] = []
        self._train_batch_size = train_batch_size
        self._gamma = gamma

    def run(self):
        for episode in range(self._n_episodes):
            self._play_episode(episode)

    def add_on_episode_end_callback(self, callback: Callback):
        self._on_episode_end_callacks.append(callback)

    @abstractmethod
    def _step(self) -> StepResult:
        pass

    def _play_episode(self, epsiode_number: int):
        done = False
        while not done:
            current_state_tensor = self._current_state.to_tensor()
            action = self._agent.select_action(
                current_state_tensor, self._policy_network
            )
            next_state, reward, done, _ = self._step()
            next_state_tensor = next_state.to_tensor()

            experience = Experience(
                current_state_tensor, action, reward, next_state_tensor
            )
            self._memory.push(experience)
            self._current_state = next_state

            if self._memory.can_provide_sample(self._train_batch_size):
                experiences = self._memory.sample(self._train_batch_size)
                states, actions, rewards, next_states = self._extract_tensors(
                    experiences
                )

                current_q_values = QValues.get_current(
                    self._policy_network, states, actions
                )
                next_q_values = QValues.get_next(self._target_network, next_states)
                target_q_values = (next_q_values * self._gamma) + rewards

                # Um... check this... I had next_q_values.unsqueeze here... why?
                loss = torch.nn.functional.mse_loss(
                    current_q_values, target_q_values.unsqueeze(1)
                )
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

        for callback in self._on_episode_end_callacks:
            callback(epsiode_number, self._current_state)

    def _extract_tensors(
        self, experiences: list[Experience]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states = torch.stack([e.initial_state for e in experiences])
        actions = torch.from_numpy(
            np.array([e.action for e in experiences], dtype=np.int64)
        )
        rewards = torch.Tensor([e.reward for e in experiences])
        next_states = torch.stack([e.next_state for e in experiences])

        return states, actions, rewards, next_states
