from torchvision.transforms import v2
import torch

# The mean and standard deviations across each channel for the normalized pixels
# of every single image in the ImageNet1k Dataset
MEANS = (0.485, 0.456, 0.406)
STDS = (0.229, 0.224, 0.225)

trainval_transforms = v2.Compose([
    v2.ColorJitter(
        brightness=[0.6, 1.4],
        contrast=[0.6, 1.4],
        saturation=[0.6, 1.4]
    ),
    v2.Normalize(mean=MEANS, std=STDS)
])

test_transforms = v2.Compose([
    v2.Normalize(mean=MEANS, std=STDS)
])

def revert_normalization(image):
    image *= 255
    return image.to(torch.uint8)

def revert_standardization(image):
    for i in range(3):
        image[i] *= STDS[i]
        image[i] += MEANS[i]