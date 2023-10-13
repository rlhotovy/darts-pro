import pytest
from torch import nn

from darts_pro.rl_mitch.networks.config.perceptron_layer_config import (
    PerceptronLayerConfig,
)


class TestPerceptronLayerConfig:
    """Tests for the PerceptronLayerConfig class"""

    @staticmethod
    def test_init() -> None:
        """GIVEN a PerceptronLayerConfig
        WHEN the config is initialised
        THEN the config has the correct internal structure
        """

        input_size = 10
        output_size = 5
        activation = nn.ReLU()
        dropout = 0.3

        config = PerceptronLayerConfig(
            input_size=input_size,
            output_size=output_size,
            activation=activation,
            dropout=dropout,
        )

        assert config.input_size == input_size
        assert config.output_size == output_size
        assert config.activation == activation
        assert config.dropout == dropout

    @staticmethod
    def test_init_negative_input_size() -> None:
        """GIVEN a PerceptronLayerConfig
        WHEN the config is initialised with a negative input size
        THEN a ValueError is raised
        """

        input_size = -10
        output_size = 5
        activation = nn.ReLU()
        dropout = 0.3

        with pytest.raises(ValueError):
            PerceptronLayerConfig(
                input_size=input_size,
                output_size=output_size,
                activation=activation,
                dropout=dropout,
            )

    @staticmethod
    def test_init_negative_output_size() -> None:
        """GIVEN a PerceptronLayerConfig
        WHEN the config is initialised with a negative output size
        THEN a ValueError is raised
        """

        input_size = 10
        output_size = -5
        activation = nn.ReLU()
        dropout = 0.3

        with pytest.raises(ValueError):
            PerceptronLayerConfig(
                input_size=input_size,
                output_size=output_size,
                activation=activation,
                dropout=dropout,
            )

    @staticmethod
    def test_init_negative_dropout() -> None:
        """GIVEN a PerceptronLayerConfig
        WHEN the config is initialised with a negative dropout
        THEN a ValueError is raised
        """
        input_size = 10
        output_size = 5
        activation = nn.ReLU()
        dropout = -0.3

        with pytest.raises(ValueError):
            PerceptronLayerConfig(
                input_size=input_size,
                output_size=output_size,
                activation=activation,
                dropout=dropout,
            )

    @staticmethod
    def test_init_dropout_too_large():
        """GIVEN a PerceptronLayerConfig
        WHEN the config is initialised with a dropout greater than or equal to 1.0
        THEN a ValueError is raised
        """

        input_size = 10
        output_size = 5
        activation = nn.ReLU()

        for dropout in (1.0, 1.5):
            with pytest.raises(ValueError):
                PerceptronLayerConfig(
                    input_size=input_size,
                    output_size=output_size,
                    activation=activation,
                    dropout=dropout,
                )
