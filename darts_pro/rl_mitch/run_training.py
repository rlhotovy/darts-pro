from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from darts_pro.rl_mitch.dartboards.base_dartboard import DartBoard
from darts_pro.rl_mitch.dartboards.grid_dartboard import GridDartBoard, ScoreGrid

from darts_pro.rl_mitch.geometry import Point2d, Size2d
from darts_pro.rl_mitch.networks.config.multi_layer_perceptron_config import (
    MultiLayerPerceptronConfig,
)
from darts_pro.rl_mitch.networks.config.perceptron_layer_config import (
    PerceptronLayerConfig,
)
from darts_pro.rl_mitch.networks.multilayer_perceptron import MultiLayerPerceptron
from random import random

def get_dartboard(board_size_mm: Size2d) -> DartBoard:
    score_grid = ScoreGrid(np.array([[1, 2], [3, 5]]))
    return GridDartBoard(board_size_mm, score_grid)


class ActionSpace:
    """The action is to choose a point to throw the dart at. This is measured from the
    top left corner of the smallest enclosing rectangle.
    """

    def __init__(self, size_mm: Size2d):
        self.size_mm = size_mm

    def sample(self) -> Point2d:
        """Sample a random action. This is measured from the top left corner of the
        smallest enclosing rectangle.

        Returns
        -------
        Point2d
            A random point to throw the dart at. This is measured from the top left
            corner of the smallest enclosing rectangle.
        """
        return Point2d(
            np.random.uniform(0, self.size_mm.width_mm),
            np.random.uniform(0, self.size_mm.height_mm),
        )

    @property
    def dimensions(self) -> int:
        return 2


@dataclass
class State:
    player_accuracy_std_mm: float
    current_score: int
    darts_thrown: int = 0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.player_accuracy_std_mm, self.current_score])


class StateSpace:
    dimensions = 2


def is_end_episode(state: State) -> bool:
    """Check if the episode has ended.

    Parameters
    ----------
    state : State
        The current state.

    Returns
    -------
    bool
        True if the episode has ended, False otherwise.
    """
    return state.current_score == 0


@dataclass
class Experience:
    state: State
    action: Point2d
    reward: float
    new_state: State


# class StateNormalisation:
#     def __init__(self, dartboard: DartBoard):
#         self.dartboard = dartboard
    
#     def normalise(self, state: State) -> State:
#         self.dartboard.

class Network(nn.Module):
    def __init__(self, state_space_dimensions: int, action_space_dimensions: int):
        super().__init__()
        self.input_size = state_space_dimensions
        self.output_size = action_space_dimensions
        self.model = self.build_model_layers()


    def forward(self, state: State) -> torch.Tensor:
        return self.model(state.to_tensor().float())


    def build_model_layers(self) -> nn.Module:
        hidden_size = 8
        activation = nn.ReLU()

        layer_1_config = PerceptronLayerConfig(
            input_size=self.input_size,
            output_size=hidden_size,
            activation=activation,
        )

        layer_2_config = PerceptronLayerConfig(
            input_size=hidden_size, output_size=self.output_size,dropout=0.1,
        )

        config = MultiLayerPerceptronConfig(
            layer_configs=(layer_1_config, layer_2_config)
        )

        return MultiLayerPerceptron(config=config)


@dataclass()
class ExplorationConfig:
    """Configuration for exploration for Q-learning."""

    epsilon: float = 0.5  # balance between exploration and exploitation
    rate: float = 1
    max_rate: float = 1
    min_rate: float = 0.01
    decay_rate: float = 0.01

    def update(self, episode: int) -> None:
        """Update the exploration rate.

        Parameters
        ----------
        episode : int
            The current episode.
        """
        self.epsilon = self.min_rate + (self.max_rate - self.min_rate) * np.exp(
            -self.decay_rate * episode
        )

class DeepLearningConfig:
    """Configuration for deep learning."""
    learning_rate_alpha: float = 0.001
    batch_size: int = 10


@dataclass()
class QLearningConfig:
    """Configuration for Q-learning."""

    num_episodes = 1000
    max_steps_per_episode = 20
    deep_learning: DeepLearningConfig = DeepLearningConfig()
    gamma = 0.6  # discount factor (value of future rewards)
    exploration: ExplorationConfig = ExplorationConfig()


class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.position = 0

    def push(self, experience: Experience) -> None:
        """Push an experience to the memory.

        Parameters
        ----------
        experience : Experience
            The experience to push to the memory.
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def can_provide_sample(self, batch_size: int) -> bool:
        """Check if the memory can provide a batch of experiences.

        Parameters
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns
        -------
        bool
            True if the memory can provide a batch of experiences, False otherwise.
        """
        return len(self.memory) >= batch_size

    def sample(self, batch_size: int) -> Experience:
        """Sample a batch of experiences from the memory.

        Parameters
        ----------
        batch_size : int
            The number of experiences to sample.

        Returns
        -------
        Experience
            The batch of experiences.
        """
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


class RewardConfig:
    miss_the_board: float = -1
    bust: float = -2
    score_is_zero: float = 10

class RewardCalculator:
    def __init__(self, reward_config: RewardConfig):
        self.reward_config = reward_config

    def calculate_reward(self, shot_score:int, state: State) -> Tuple[float, State]:
        
        if shot_score == 0: # miss the board
            new_state = State(
                player_accuracy_std_mm=state.player_accuracy_std_mm,
                current_score=state.current_score,
                darts_thrown=state.darts_thrown + 1,
            )
            reward = self.reward_config.miss_the_board
        
        elif self.player_has_busted(state, shot_score):
            new_state = State(
                player_accuracy_std_mm=state.player_accuracy_std_mm,
                current_score=state.current_score,
                darts_thrown=state.darts_thrown + 1,
            )
            reward = self.reward_config.bust
        elif self.player_has_checked_out(state, shot_score):
            new_state = State(
                player_accuracy_std_mm=state.player_accuracy_std_mm,
                current_score=0,
                darts_thrown=state.darts_thrown + 1,
            )
            reward = self.reward_config.score_is_zero

        else:
            new_state = State(
                player_accuracy_std_mm=state.player_accuracy_std_mm,
                current_score=state.current_score - shot_score,
                darts_thrown=state.darts_thrown + 1,
            )
            reward = shot_score

        return reward, new_state

    def player_has_checked_out(self, state: State, shot_score: int) -> bool:
        """Check if the player has checked out.

        Parameters
        ----------
        state : State
            The current state.
        shot_score : int
            The score of the dart shot.

        Returns
        -------
        bool
            True if the player has checked out, False otherwise.
        """
        return state.current_score == shot_score


    def player_has_busted(self, state: State, shot_score: int) -> bool:
        """Check if the player has busted.

        Parameters
        ----------
        state : State
            The current state.
        score : int
            The score of the dart shot.

        Returns
        -------
        bool
            True if the player has busted, False otherwise.
        """
        return shot_score > state.current_score



class DartsGame:
    def __init__(self, dartboard: DartBoard, reward_calculator: RewardCalculator):
        self.dartboard = dartboard
        self.reward_calculator = reward_calculator

    def take_action(self, state:State,  action: Point2d) -> Tuple[State, float]:
        """Take an action.

        Parameters
        ----------
        action : Point2d
            The action to take.

        Returns
        -------
        int
            The reward for the action.
        """
        shot_score = self.dartboard.score(action)
        reward, new_state = self.reward_calculator.calculate_reward(shot_score, state)
        return new_state, reward


def main() -> None:

    board_size_mm = Size2d(500, 500)
    dartboard = get_dartboard(board_size_mm=board_size_mm)

    action_space = ActionSpace(board_size_mm)
    state_space = StateSpace()

    player_accuracy_std_mm = 0
    current_score = 10
    initial_state = State(
        player_accuracy_std_mm=player_accuracy_std_mm,
        current_score=current_score,
    )

    q_learning_config = QLearningConfig()

    reward_config = RewardConfig()
    reward_calculator = RewardCalculator(reward_config=reward_config)

    darts_game = DartsGame(dartboard=dartboard, reward_calculator=reward_calculator)

    replay_memory = ReplayMemory(capacity=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy_network = Network(
        state_space_dimensions=state_space.dimensions,
        action_space_dimensions=action_space.dimensions,
    ).to(device)

    target_network = Network(
        state_space_dimensions=state_space.dimensions,
        action_space_dimensions=action_space.dimensions,
    ).to(device)

    target_network.eval()

    optimizer = torch.optim.Adam(policy_network.parameters(), lr=q_learning_config.deep_learning.learning_rate_alpha)
    

    target_network.load_state_dict(policy_network.state_dict())


    rewards_all_episodes = []

    for episode in range(q_learning_config.num_episodes):
        state = initial_state
        reward_current_episode = 0

        for step in range(q_learning_config.max_steps_per_episode):

            # exploration-exploitation trade-off
            exploration_rate_threshold = np.random.uniform(0, 1)

            if exploration_rate_threshold > q_learning_config.exploration.epsilon:
                # TODO - this should output q values, not an action. This approach
                # is not going to be possible though, as the action space is
                # continuous. Need to use a policy gradient approach.
                action = policy_network(state)
            else:
                action = action_space.sample()

            new_state, reward = darts_game.take_action(state, action)  

            replay_memory.push(
                Experience(
                    state=state,
                    action=action,
                    reward=reward,
                    new_state=new_state,
                )
            )      

            state = new_state

        
            if replay_memory.can_provide_sample(batch_size=q_learning_config.deep_learning.batch_size):
                
                experiences = replay_memory.sample(batch_size=q_learning_config.deep_learning.batch_size)
                
                
                current_q_values = policy_network(experiences.state)
                next_q_values = target_network(experiences.new_state)

                target_q_values = (experiences.reward + q_learning_config.gamma * torch.max(next_q_values, dim=1).values)
                
                loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


            if is_end_episode(state):
                rewards_all_episodes.append(reward_current_episode)
                break

        # exploration rate decay
        q_learning_config.exploration.update(episode)

        if episode % 100 == 0:
            target_network.load_state_dict(policy_network.state_dict())
            print(f"Episode: {episode}")


        rewards_all_episodes.append(reward_current_episode)

    # Calculate and print the average reward per thousand episodes
    rewards_per_thousand_episodes = np.split(
        np.array(rewards_all_episodes), q_learning_config.num_episodes / 1000
    )

    count = 1000
    print("********Average reward per thousand episodes********\n")
    for r in rewards_per_thousand_episodes:
        print(count, ": ", str(sum(r / 1000)))
        count += 1000

    # Print updated Q-table
    print("\n\n********Q-table********\n")



if __name__ == "__main__":
    main()
