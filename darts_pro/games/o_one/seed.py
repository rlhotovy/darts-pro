import torch
import random

from ...dqn import SeededReplayMemory, TrainingPlayer, Experience
from ...base import DartBoard

def create_seeded_x01_memory(
        player: TrainingPlayer,
        board: DartBoard,
        max_score: int,
        n_turns: int,
        non_seeded_capacity: int,
        samples_per_target: int = 100
    ) -> SeededReplayMemory:
    """
    Seeds closeouts from single darts
    """
    experiences = []
    turn_numbers = list(range(n_turns))
    for action, target in board.indexed_targets.items():
        starting_value = target.multiplier * target.value
        probs = player.get_outcome_probabilities(board, target)
        hits = random.choices([p[0] for p in probs], weights=[p[1] for p in probs], k=samples_per_target)
        for hit in hits:
            turn_number = random.choice(turn_numbers)
            throw_number = random.choice(list(range(3))) # Could pass this in later on
            hit_value = hit.multiplier * hit.value

            reward = 0
            new_value = starting_value - hit_value
            done = False
            if hit_value == starting_value:
                reward = 1
                done = True

            if hit_value > starting_value:
                if turn_number == n_turns - 1:
                    reward = -1
                    new_value = 0
                    done = True
                else:
                    new_value = starting_value
            
            starting_tensor = torch.tensor([0, max_score, throw_number, turn_number, starting_value])
            end_tensor = torch.tensor([0, max_score, throw_number + 1 % 3, turn_number + 1 % n_turns, new_value])
        
            experiences.append(Experience(starting_tensor, action, reward, end_tensor, done))
    
    return SeededReplayMemory(experiences, non_seeded_capacity)
