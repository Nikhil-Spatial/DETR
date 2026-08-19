from configs import D, H, W, HEADS
import torch.nn.functional as F
from torch import sin, cos
from torch import nn
import torch

def positional_encoding():
    pos_encodings = torch.empty(H, W, D)

    for pos_y in range(H):  # rows
        for pos_x in range(W):  # columns
            # compute y's half and x's half of positional encoding
            for i in range(0, D // 4):
                # sinusoidal positional encoding from "Attention Is All You
                # Need" paper
                scale = torch.tensor(10_000 ** (2 * i / (D / 2)))
                argument_y = pos_y / scale
                argument_x = pos_x / scale

                pos_encodings[pos_y][pos_x][2*i] = sin(argument_y)
                pos_encodings[pos_y][pos_x][2*i+1] = cos(argument_y)

                pos_encodings[pos_y][pos_x][(2*i)+128] = sin(argument_x)
                pos_encodings[pos_y][pos_x][(2*i+1)+128] = cos(argument_x)

    return pos_encodings.reshape(H*W, D)

class SelfAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()

        # dimensions of each token in the query, key, and value tensors
        self.D_qkv = torch.tensor(D // HEADS)

        # Q, K, and V transformations are purely linear
        self.Q_linear_transform = nn.Linear(D, self.D_qkv, bias=False)
        self.K_linear_transform = nn.Linear(D, self.D_qkv, bias=False)
        self.V_linear_transform = nn.Linear(D, self.D_qkv, bias=False)

    def forward(self, x):
        # query, key, and value tensors
        Q = self.Q_linear_transform(x)
        K = self.K_linear_transform(x)
        V = self.V_linear_transform(x)

        # compute attention scores:
        # softmax(matmul(Q, transpose(K)) / sqrt(D_qkv))
        attention_scores = F.softmax(
            (Q @ K.transpose(-2, -1)) / self.D_qkv.sqrt(),
            dim=-1
        )

        # compute head output
        return attention_scores @ V