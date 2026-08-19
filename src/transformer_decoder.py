from transformer_layers import MultiHeadAttention, FFN
from configs import D
from torch import nn
import torch

class TransformerDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()

        # layer normalization layers
        self.layer_norm_1 = nn.LayerNorm(D)
        self.layer_norm_2 = nn.LayerNorm(D)
        self.layer_norm_3 = nn.LayerNorm(D)

        # multi-head attention for object queries
        self.multi_head_attention = MultiHeadAttention()

        # cross-attention for contextualized object queries and encoder output
        self.cross_attention = MultiHeadAttention()

        # feed-forward network
        self.ffn = FFN()

    def forward(self, object_queries, encoder_output):
        # contextualize object queries
        object_queries = self.multi_head_attention(
            object_queries, object_queries
        )

        return
