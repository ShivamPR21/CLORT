import torch
import torch.nn as nn


class FeatureMixer(nn.Module):

    def __init__(self,
                 vis_size: int = 512,
                 pcl_size: int = 40,
                 embed_dim: int = 64) -> None:
        super().__init__()

        self.activation = nn.SELU()

        # Visual dimension reduction
        self.v_linear_1 = nn.Linear(vis_size, 128, bias=False)
        self.v_bn_1 = nn.BatchNorm1d(128)

        self.v_linear_2 = nn.Linear(128, 64, bias=True)

        # Fusion layers
        fused_size = 64+pcl_size
        self.f_linear_1 = nn.Linear(fused_size, 128)
        self.f_linear_2 = nn.Linear(128, embed_dim)

    def forward(self, x_vis: torch.Tensor, x_pcl: torch.Tensor) -> torch.Tensor:
        x_vis = self.activation(self.v_bn_1(self.v_linear_1(x_vis)))
        x_vis = self.activation(self.v_linear_2(x_vis))

        x = torch.cat((x_vis, x_pcl), dim=1)

        x = self.activation(self.f_linear_1(x))
        x = self.activation(self.f_linear_2(x))

        return x
