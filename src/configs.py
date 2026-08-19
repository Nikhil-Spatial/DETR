# width and height of the resized images
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

# seed for reproducibility
SEED = 7

# number of classes
C = 20

CLASS_TO_IDX = {
    "aeroplane": 0,
    "bicycle": 1,
    "bird": 2,
    "boat": 3,
    "bottle": 4,
    "bus": 5,
    "car": 6,
    "cat": 7,
    "chair": 8,
    "cow": 9,
    "diningtable": 10,
    "dog": 11,
    "horse": 12,
    "motorbike": 13,
    "person": 14,
    "pottedplant": 15,
    "sheep": 16,
    "sofa": 17,
    "train": 18,
    "tvmonitor": 19,
}

IDX_TO_CLASS = {
    idx: class_
    for class_, idx in CLASS_TO_IDX.items()
}

# size of batches in DataLoader
BATCH_SIZE = 32

# embedded dimensions
D = 256

# height/width of feature maps output by backbone
H = 7
W = 7

# probability of dropout layer
dropout_p = 0.0

# number of heads in the multi-head self-attention layer
HEADS = 4

# backbone output channels
BB_CHANNELS = 512