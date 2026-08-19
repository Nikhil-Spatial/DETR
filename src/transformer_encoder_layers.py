from configs import D, H, W, HEADS, BB_CHANNELS, dropout_p, ENCODER_LAYERS
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

class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()

        # list of self-attention heads
        self.self_attention_heads = nn.ModuleList(
            [SelfAttentionHead() for _ in range(HEADS)]
        )

        # apply linear transformation to vertically concatenated head outputs
        self.output_linear_transform = nn.Linear(D, D, bias=False)

    def forward(self, x):
        # compute head outputs
        head_outputs = [self_attention_head(x) for self_attention_head
                        in self.self_attention_heads]

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

class TransformerEncoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # layer normalization layers
        self.layer_norm_1 = nn.LayerNorm(D)
        self.layer_norm_2 = nn.LayerNorm(D)

        # multi-head self-attention layer
        self.multi_head_self_attention = MultiHeadSelfAttention()

        # feed-forward network (FFN)
        self.ffn = FFN()

    def forward(self, x):
        # multi-head self-attention residual connection -> layer normalize
        x = self.multi_head_self_attention(x) + x
        x = self.layer_norm_1(x)

        # ffn residual connection -> layer normalize
        x = self.ffn(x) + x

        return self.layer_norm_2(x)

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        # module list of transformer layers
        self.transformer_encoder_layers = nn.ModuleList(
            [TransformerEncoderLayer() for _ in range(ENCODER_LAYERS)]
        )
    def forward(self, x):
        # feed patch embeddings to the transformer layers
        for transformer_encoder_layer in self.transformer_encoder_layers:
            x = transformer_encoder_layer(x)

        return x