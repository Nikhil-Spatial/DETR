from torchvision.models import resnet18
from transformer_encoder_layers import positional_encoding, TransformerEncoder
from configs import D, H, W, dropout_p, BB_CHANNELS, N
from torch import nn
import torch

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        # use resnet18 pretrained on ImageNet as the backbone by cutting off
        # its classification head
        self.backbone = nn.Sequential(
            *list(resnet18(weights="DEFAULT").children())[:-2]
        )

        # project channels to configured embedded dimensions
        self.projection = nn.Conv2d(BB_CHANNELS, D, kernel_size=1)

        # store positional encoding, and register it as a buffer in the
        # model's state dict
        pos_encoding = positional_encoding().unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)
        self.dropout = nn.Dropout(dropout_p)

        # transformer encoder
        self.transformer_encoder = TransformerEncoder()

        # object queries
        self.object_queries = nn.Parameter(torch.randn(N, D))

    def forward(self, x):
        batch_size = x.shape[0]

        # (B, 3, IMAGE_HEIGHT, IMAGE_WIDTH) -> (B, 512, H, W) -> (B, 256, H, W)
        x = self.backbone(x)
        x = self.projection(x)

        # (B, 256, H, W) -> (B, H*W, D)
        x = x.reshape(batch_size, D, H*W).transpose(-2, -1)

        # add positional encoding to feature maps
        x = self.dropout(x + self.pos_encoding)

        # feed feature embeddings to transformer encoder
        x = self.transformer_encoder(x)

        return x

inputs = torch.randn(32, 3, 224, 224)
model = Model()
print(model(inputs).shape)



