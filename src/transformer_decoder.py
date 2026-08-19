from IPython.core.magic import output_can_be_silenced

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

        # multi-head attention (MHA) for object queries
        self.mha = MultiHeadAttention()

        # cross-attention for contextualized object queries and encoder output
        self.cross_attention = MultiHeadAttention()

        # feed-forward network (FFN)
        self.ffn = FFN()

    def forward(self, object_queries, query_pos, encoder_output, pos_encoding):
        # 1) contextualize object queries, add a residual connection, and apply
        # layer normalization
        object_queries = object_queries + self.mha(
            x_q=(object_queries + query_pos),
            x_k=(object_queries + query_pos),
            x_v=object_queries
        )
        object_queries = self.layer_norm_1(object_queries)

        # 2) apply cross-attention using contextualized object queries as
        # queries and the encoder's output as keys and values, add a residual
        # connection with the contextualized object queries, and apply layer
        # normalization
        object_queries = object_queries + self.cross_attention(
            x_q=(object_queries + query_pos),
            x_k=(encoder_output + pos_encoding),
            x_v=encoder_output
        )
        object_queries = self.layer_norm_2(object_queries)

        # 3) feed object queries that are contextualized with the encoder's
        # representation of the image to the feed-forward network (FFN), add a
        # residual connection, and apply layer normalization
        object_queries = object_queries + self.ffn(object_queries)

        return self.layer_norm_3(object_queries)
