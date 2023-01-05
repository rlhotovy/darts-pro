import torch
from abc import abstractmethod


class DQN(torch.nn.Module):
    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        pass
