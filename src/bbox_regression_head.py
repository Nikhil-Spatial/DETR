import torch.nn.functional as F
from torch import nn
import torch

class bbox_regression_FFN(nn.Module):
    def __init__(self):
        super().__init__()

        self.affine_transform_1 = nn.Linear(D, D)
        self.affine_transform_2 = nn.Linear(D, D)

        # the output dimension of 4 represents the 4 predictions:
        # (center_x, center_y, width, height)
        self.output_layer = nn.Linear(D, 4)