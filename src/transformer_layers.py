from configs import D, HEADS, dropout_p
import torch.nn.functional as F
from torch import nn
import torch

class AttentionHead(nn.Module):
    def __init__(self):
        super().__init__()

        # dimensions of each token in the query, key, and value tensors
        self.D_qkv = torch.tensor(D // HEADS)

        # Q, K, and V transformations are purely linear
        self.Q_linear_transform = nn.Linear(D, self.D_qkv, bias=False)
        self.K_linear_transform = nn.Linear(D, self.D_qkv, bias=False)
        self.V_linear_transform = nn.Linear(D, self.D_qkv, bias=False)

    def forward(self, x_q, x_k, x_v):
        # query, key, and value tensors
        Q = self.Q_linear_transform(x_q)
        K = self.K_linear_transform(x_k)
        V = self.V_linear_transform(x_v)

        # compute attention scores:
        # softmax(matmul(Q, transpose(K)) / sqrt(D_qkv))
        attention_scores = F.softmax(
            (Q @ K.transpose(-2, -1)) / self.D_qkv.sqrt(),
            dim=-1
        )

        # compute head output
        return attention_scores @ V

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # list of attention heads
        self.attention_heads = nn.ModuleList(
            [AttentionHead() for _ in range(HEADS)]
        )

        # apply linear transformation to vertically concatenated head outputs
        self.output_linear_transform = nn.Linear(D, D, bias=False)

    def forward(self, x_q, x_k, x_v):
        # compute head outputs
        head_outputs = [attention_head(x_q, x_k, x_v) for attention_head
                        in self.attention_heads]

        # vertically concatenate the head outputs
        output = torch.cat(head_outputs, dim=-1)

        # apply linear transformation
        return self.output_linear_transform(output)

class FFN(nn.Module):
    def __init__(self):
        super().__init__()

        self.affine_transform_1 = nn.Linear(D, D)
        self.dropout_1 = nn.Dropout(dropout_p)

        self.affine_transform_2 = nn.Linear(D, D)
        self.dropout_2 = nn.Dropout(dropout_p)

    def forward(self, x):
        # affine transform -> dropout -> ReLU activation -> affine transform
        # -> dropout
        x = F.relu(self.dropout_1(self.affine_transform_1(x)))

        return self.dropout_2(self.affine_transform_2(x))