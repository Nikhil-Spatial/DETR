from torchvision.models import resnet18
from transformer_layers import positional_encodings
from configs import D, H, W, dropout_p
import torch.nn as nn
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
        self.projection = nn.Conv2d(512, D, kernel_size=1)

        # store positional encodings + add dropout after positional encodings
        self.pos_encodings = positional_encodings().unsqueeze(0)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        batch_size = x.shape[0]

        # (B, 3, IMAGE_HEIGHT, IMAGE_WIDTH) -> (B, 512, H, W) -> (B, 256, H, W)
        x = self.backbone(x)
        x = self.projection(x)

        # (B, 256, H, W) -> (B, H*W, D)
        x = x.reshape(batch_size, D, H*W).transpose(-2, -1)

        x = self.dropout(x + self.pos_encodings)

        return x

inputs = torch.randn(32, 3, 224, 224)
model = Model()
print(model(inputs).shape)



