import torch.nn.functional as F
from configs import D
from torch import nn
import torch

class bbox_regression_FFN(nn.Module):
    def __init__(self):
        super().__init__()

        # the two hidden layers
        self.affine_transform_1 = nn.Linear(D, D)
        self.affine_transform_2 = nn.Linear(D, D)

        # the output dimension of 4 represents the 4 predictions:
        # (center_x, center_y, width, height)
        self.output_layer = nn.Linear(D, 4)

    def forward(self, x):
        x = F.relu(self.affine_transform_1(x))
        x = F.relu(self.affine_transform_2(x))

        return F.sigmoid(self.output_layer(x))