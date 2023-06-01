from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from utilities.impCarv_points import PulsarLayer

torch.manual_seed(13)


class SimpleMLP(nn.Module):
    def __init__(self, input_dim: int, net_type: str = "painter") -> None:
        super(SimpleMLP, self).__init__()
        num_layers = 6
        net_layers = []
        if net_type == "painter":
            layer_dims = [input_dim, 1024, 512, 256, 128, 64, 32]
        else:
            layer_dims = [input_dim, 256, 512, 256, 128, 64, 1]
            # layer_dims = [input_dim, 1024, 512, 256, 128, 64, 1]

        for i in range(num_layers - 1):
            net_layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            net_layers.append(nn.ReLU())
        net_layers.append(nn.Linear(layer_dims[-2], layer_dims[-1]))

        if net_type == "painter":
            net_layers.append(nn.Tanh())
        else:
            net_layers.append(nn.Sigmoid())

        self.my_net = nn.Sequential(*net_layers)

    def forward(self, x: torch.tensor) -> torch.tensor:
        out = self.my_net(x)
        return out


if __name__ == "__main__":
    in_dim = 1096
    x = torch.randn((1, 300000, in_dim))
    # x = x.reshape(-1,7)
    print(f"x.size(0):{x.size(0)}")
    painter_net = SimpleMLP(input_dim=1096, net_type="painter")
    print(painter_net)
    # simple_net = CheckNN()
    out = painter_net(x)
    print(f"out.shape: {out.shape}")
    print(f"min(out):{torch.min(out)}, max(out): {torch.max(out)}")

    exit()
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    print("#######   Debugging   ##########")
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    for name, val in painter_net.named_parameters():
        print(f"name: {name}")
    print(f"simple_net[0] weights :{painter_net.painter_net[0].weight.shape}")
