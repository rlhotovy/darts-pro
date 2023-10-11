import torch


class QValues:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def get_current(
        policy_net: torch.nn.Module, states: torch.Tensor, actions: torch.Tensor
    ) -> float:
        return policy_net(states).gather(dim=1, index=actions.unsqueeze(1))

    @staticmethod
    def get_next(
        target_net: torch.nn.Module,
        next_states: torch.Tensor,
        next_is_final: torch.Tensor,
    ) -> torch.Tensor:
        non_final_states = next_states[~next_is_final]
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).to(QValues.device)
        values[~next_is_final] = target_net(non_final_states).max(dim=1)[0].detach()
        return values
