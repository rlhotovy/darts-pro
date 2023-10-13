from collections import OrderedDict

from torch import Tensor, nn

from darts_pro.rl_mitch.networks.config.multi_layer_perceptron_config import MultiLayerPerceptronConfig
from darts_pro.rl_mitch.networks.perceptron_layer import PerceptronLayer


class MultiLayerPerceptron(nn.Module):
    """A multi-layer perceptron."""

    def __init__(self, config: MultiLayerPerceptronConfig):
        """Initialise a multi-layer perceptron.

        Parameters
        ----------
        config
            The configuration for the multi-layer perceptron
        """

        super().__init__()
        self.config = config
        self.model = self.build()

    def forward(self, x_data: Tensor) -> Tensor:
        """Forward pass for the multi-layer perceptron.

        Parameters
        ----------
        x_data
            The input data, should be of shape (batch_size, input_size), where
            input_size is the size of the input to the first layer

        Returns
        -------
        Tensor
            The output of the multi-layer perceptron. Should be of shape
            (batch_size, output_size), where output_size is the size of the
            output of the final layer
        """

        return self.model(x_data)

    def build(self) -> nn.Module:
        """Build the multi-layer perceptron

        Returns
        -------
        The multi-layer perceptron
        """
        return nn.Sequential(
            OrderedDict(
                {
                    f"mlp_layer_{ii}": PerceptronLayer(config=layer_config)
                    for ii, layer_config in enumerate(self.config.layer_configs)
                }
            )
        )
