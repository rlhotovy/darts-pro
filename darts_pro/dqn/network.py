import torch


DQN = torch.nn.Module


class LinearNetwork(DQN):
    def __init__(self, input_size: int, output_size: int, inner_layer_sizes: list[int]):
        super().__init__()
        layers = []
        last_layer_size = input_size
        for size in inner_layer_sizes:
            layers.append(torch.nn.Linear(last_layer_size, size))
            last_layer_size = size
        layers.append(torch.nn.Linear(size, output_size))
        self._layers = torch.nn.ModuleList(layers)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        for layer in self._layers:
            t = layer(t)
            t = torch.nn.functional.relu(t)
        return t
