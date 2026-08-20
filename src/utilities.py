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

def area(bbox):
    w = torch.clamp(bbox[2] - bbox[0], min=0)
    h = torch.clamp(bbox[3] - bbox[1], min=0)

    return w * h

def IoU(bbox_1, bbox_2):
    # 1. find intersection box coordinates
    x1 = torch.max(bbox_1[0], bbox_2[0])
    y1 = torch.max(bbox_1[1], bbox_2[1])
    x2 = torch.min(bbox_1[2], bbox_2[2])
    y2 = torch.min(bbox_1[3], bbox_2[3])

    inter = (x1, y1, x2, y2)

    # 2. find areas of boxes
    target_area = area(bbox_1)
    pred_area = area(bbox_2)
    inter_area = area(inter)
    union_area = target_area + pred_area - inter_area

    # 3. compute IoU
    return inter_area / union_area if union_area != 0 else 0