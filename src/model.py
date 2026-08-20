from torchvision.models import resnet18
from transformer_encoder import positional_encoding, TransformerEncoder
from transformer_decoder import TransformerDecoder
from bbox_regression_head import bbox_regression_FFN
from configs import D, H, W, BB_CHANNELS, N, C
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

        # transformer encoder
        self.transformer_encoder = TransformerEncoder()

        # learnable positional object query embeddings
        self.query_pos = nn.Parameter(torch.randn(N, D))

        # transformer decoder
        self.transformer_decoder = TransformerDecoder()

        # classification head: projects D embedded dims to C + 1 dims
        # the + 1 represents a "no-object" class
        self.class_head = nn.Linear(D, C+1)

        # bounding box regression head: refines predictions, and then projects
        # D embedded dims to 4 dims for the center coordinates, height, and
        # width.
        self.bbox_regression_head = bbox_regression_FFN()

    def forward(self, x):
        batch_size = x.shape[0]

        # (B, 3, IMAGE_HEIGHT, IMAGE_WIDTH) -> (B, 512, H, W) -> (B, 256, H, W)
        x = self.backbone(x)
        x = self.projection(x)

        # (B, 256, H, W) -> (B, H*W, D)
        x = x.reshape(batch_size, D, H*W).transpose(-2, -1)

        # feed feature embeddings to transformer encoder
        x = self.transformer_encoder(x, self.pos_encoding)

        # feed transformer encoder's image representation and positional
        # encodings/embeddings to the decoder
        x = self.transformer_decoder(self.query_pos, x, self.pos_encoding)

        # classification and bounding box regression prediction heads
        x_class = self.class_head(x)
        x_bbox_pred = self.bbox_regression_head(x)

        return x_class, x_bbox_pred

inputs = torch.randn(32, 3, 224, 224)
model = Model()
class_pred, bbox_pred = model(inputs)
print(class_pred.shape, bbox_pred.shape)