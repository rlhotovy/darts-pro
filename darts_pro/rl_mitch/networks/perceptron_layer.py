from typing import List

from torch import Tensor, nn

from darts_pro.rl_mitch.networks.config.perceptron_layer_config import (
    PerceptronLayerConfig,
)


class PerceptronLayer(nn.Module):
    """A perceptron layer to be use as part of a multi-layer perceptron."""

    def __init__(self, config: PerceptronLayerConfig):
        """Initialise a perceptron layer

        Parameters
        ----------
        input_size
            The size of the input to the layer
        output_size
            The size of the output of the layer
        dropout
            The dropout rate, between 0.0 and 1.0
        """
        super().__init__()
        self.config = config
        self.model = self.build()

    def forward(self, x_data: Tensor) -> Tensor:
        """Pass the input data through the layer.

        Parameters
        ----------
        x_data
            The input data, should be of shape (batch_size, input_size)

        Returns
        -------
        Tensor
            The output of the layer
        """
        return self.model(x_data)

    def build(self) -> nn.Module:
        """Build the perceptron layer

        Returns
        -------
        The perceptron layer
        """
        layers: List[nn.Module] = []

        if self.config.dropout != 0.0:
            layers.append(nn.Dropout(self.config.dropout))

        layers.append(nn.Linear(self.config.input_size, self.config.output_size))

        if self.config.activation is not None:
            layers.append(self.config.activation)

        return nn.Sequential(*layers)
