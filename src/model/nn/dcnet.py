import torch
import torch.nn.functional as F
from torch import nn


class DCNet(torch.nn.Module):
    def __init__(self):
        super(DCNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(6, 64, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 32, 3, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = torch.nn.Conv2d(16, 8, 3, padding=1)
        self.conv5 = torch.nn.Conv2d(8, 1, 3, padding=1)

    def forward(self, x):
        h1 = self.conv1(x)
        h2 = F.leaky_relu(self.conv2(h1), negative_slope=0.05)
        h3 = F.leaky_relu(self.conv3(h2), negative_slope=0.05)
        h4 = F.leaky_relu(self.conv4(h3), negative_slope=0.05)
        return F.leaky_relu(self.conv5(h4), negative_slope=0.05)

    def initialize(self, model_weight_path, cuda):
        if model_weight_path:
            self.load_state_dict(torch.load(model_weight_path))
        else:
            for layer in self.children():
                nn.init.xavier_normal_(layer.weight)
                nn.init.uniform_(layer.bias, -1, 1)
        device = torch.device("cuda" if cuda else "cpu")
        self.double().to(device)
