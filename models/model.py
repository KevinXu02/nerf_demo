import torch
import torch.nn as nn


class NeRF(nn.Module):
    def __init__(
        self,
        L_position=10,
        L_direction=4,
        hidden_dim=256,
        include_input_xyz=True,
        include_input_dir=True,
    ):
        super(NeRF, self).__init__()
        self.L_pos = L_position
        self.L_dir = L_direction
        self.input_dim_pos = (
            3 * (2 * self.L_pos + 1) if include_input_xyz else 6 * self.L_pos
        )
        self.input_dim_dir = (
            3 * (2 * self.L_dir + 1) if include_input_dir else 6 * self.L_dir
        )
        self.hidden_dim = hidden_dim
        self.output_dim = 4
        self.net_1 = nn.Sequential(
            nn.Linear(self.input_dim_pos, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.net_2 = nn.Sequential(
            nn.Linear(self.input_dim_pos + self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
        )
        self.density = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.Linear(self.hidden_dim // 2, 1),
            nn.ReLU(),
        )
        self.linear1 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.color = nn.Sequential(
            nn.Linear(self.hidden_dim + self.input_dim_dir, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        xyz, rd = x[..., : self.input_dim_pos], x[..., self.input_dim_pos :]
        out1 = self.net_1(xyz)
        out1 = torch.cat([out1, xyz], dim=-1)
        out2 = self.net_2(out1)
        alpha = self.density(out2)
        out3 = torch.cat([out2, rd], dim=-1)
        rgb = self.color(out3)
        return alpha, rgb
