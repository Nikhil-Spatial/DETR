from transformer_layers import MultiHeadAttention, FFN
from configs import D, H, W, ENCODER_LAYERS
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

                pos_encodings[pos_y][pos_x][(2*i)+(D//2)] = sin(argument_x)
                pos_encodings[pos_y][pos_x][(2*i+1)+(D//2)] = cos(argument_x)

    return pos_encodings.reshape(H*W, D)

class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # layer normalization layers
        self.layer_norm_1 = nn.LayerNorm(D)
        self.layer_norm_2 = nn.LayerNorm(D)

        # multi-head attention (MHA) layer
        self.mha = MultiHeadAttention()

        # feed-forward network (FFN)
        self.ffn = FFN()

    def forward(self, x, pos_encoding):
        # 1) multi-head attention residual connection -> layer normalize
        x = x + self.mha(
            x_q=(x + pos_encoding),
            x_k=(x + pos_encoding),
            x_v =x
        )
        x = self.layer_norm_1(x)

        # 2) ffn residual connection -> layer normalize
        x = self.ffn(x) + x

        return self.layer_norm_2(x)

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # module list of transformer layers
        self.transformer_encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer() for _ in range(ENCODER_LAYERS)]
        )
    def forward(self, x, pos_encoding):
        # feed patch embeddings to the transformer layers
        for transformer_encoder_layer in self.transformer_encoder_layers:
            x = transformer_encoder_layer(x, pos_encoding)

        return x