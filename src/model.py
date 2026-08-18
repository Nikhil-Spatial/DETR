from torchvision.models import resnet18
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

    def forward(self, x):
        x = self.backbone(x)
        return x

inputs = torch.randn(32, 3, 224, 224)
model = Model()
print(model(inputs).shape)



