from dataclasses import dataclass
from typing import Optional


from torch import nn


@dataclass(frozen=True)
class PerceptronLayerConfig:
    input_size: int
    output_size: int
    activation: Optional[nn.Module] = None
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.input_size <= 0:
            raise ValueError("input_size must be greater than 0")
        if self.output_size <= 0:
            raise ValueError("output_size must be greater than 0")
        if self.dropout < 0.0 or self.dropout >= 1.0:
            raise ValueError("dropout must be between 0.0 and 1.0")