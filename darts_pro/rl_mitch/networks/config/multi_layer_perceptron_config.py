from typing import Sequence
from dataclasses import dataclass

from darts_pro.rl_mitch.networks.config.perceptron_layer_config import (
    PerceptronLayerConfig,
)


@dataclass
class MultiLayerPerceptronConfig:
    layer_configs: Sequence[PerceptronLayerConfig]
