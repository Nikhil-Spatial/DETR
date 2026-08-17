from torch import round as rd
from src.configs import IMAGE_WIDTH, IMAGE_HEIGHT
import torch

def convert_xywh_coordinates(bbox, draw=False):
    # 1. unnormalize (center_x, center_y, width, height)
    x = bbox[0] * IMAGE_WIDTH
    y = bbox[1] * IMAGE_HEIGHT
    w = bbox[2] * IMAGE_WIDTH
    h = bbox[3] * IMAGE_HEIGHT

    # 2. convert center point to top-left and bottom-right of box coordinates
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    # 3. if the conversion is for drawing bounding boxes, then round
    if draw:
        return (int(rd(x1)), int(rd(y1)), int(rd(x2)), int(rd(y2)))

    return (x1, y1, x2, y2)