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


class ReplicateNeRFModel(torch.nn.Module):
    r"""NeRF model that follows the figure (from the supp. material of NeRF) to
    every last detail. (ofc, with some flexibility)
    """

    def __init__(
        self,
        hidden_size=256,
        num_layers=4,
        num_encoding_fn_xyz=10,
        num_encoding_fn_dir=4,
        include_input_xyz=True,
        include_input_dir=True,
    ):
        super(ReplicateNeRFModel, self).__init__()
        # xyz_encoding_dims = 3 + 3 * 2 * num_encoding_functions

        self.dim_xyz = (3 if include_input_xyz else 0) + 2 * 3 * num_encoding_fn_xyz
        self.dim_dir = (3 if include_input_dir else 0) + 2 * 3 * num_encoding_fn_dir

        self.layer1 = torch.nn.Linear(self.dim_xyz, hidden_size)
        self.layer2 = torch.nn.Linear(hidden_size, hidden_size)
        self.layer3 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc_alpha = torch.nn.Linear(hidden_size, 1)

        self.layer4 = torch.nn.Linear(hidden_size + self.dim_dir, hidden_size // 2)
        self.layer5 = torch.nn.Linear(hidden_size // 2, hidden_size // 2)
        self.fc_rgb = torch.nn.Linear(hidden_size // 2, 3)
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.functional.relu

    def forward(self, x):
        xyz, direction = x[..., : self.dim_xyz], x[..., self.dim_xyz :]
        x_ = self.relu(self.layer1(xyz))
        x_ = self.relu(self.layer2(x_))
        feat = self.layer3(x_)
        alpha = self.fc_alpha(x_)
        y_ = self.relu(self.layer4(torch.cat((feat, direction), dim=-1)))
        y_ = self.relu(self.layer5(y_))
        rgb = self.fc_rgb(y_)
        rgb = self.sigmoid(rgb)
        return alpha, rgb
