from torch import round as rd
from src.configs import IMAGE_WIDTH, IMAGE_HEIGHT
import torch

def intersection_coords(bbox_1, bbox_2):
    """
    :param bbox_1: a tensor where the last dimension contains 4 bbox values
    :param bbox_2: a 1D tensor of 4 bbox values
    """
    bbox_1[..., 0] = torch.max(bbox_1[..., 0], bbox_2[0])
    bbox_1[..., 1] = torch.max(bbox_1[..., 1], bbox_2[1])
    bbox_1[..., 2] = torch.min(bbox_1[..., 2], bbox_2[2])
    bbox_1[..., 3] = torch.min(bbox_1[..., 3], bbox_2[3])

def enclose_coords(bbox_1, bbox_2):
    """
    :param bbox_1: a tensor where the last dimension contains 4 bbox values
    :param bbox_2: a 1D tensor of 4 bbox values
    """
    bbox_1[..., 0] = torch.min(bbox_1[..., 0], bbox_2[0])
    bbox_1[..., 1] = torch.min(bbox_1[..., 1], bbox_2[1])
    bbox_1[..., 2] = torch.max(bbox_1[..., 2], bbox_2[2])
    bbox_1[..., 3] = torch.max(bbox_1[..., 3], bbox_2[3])

def cxcywh_to_xyxy(bboxes, draw=False):
    # 1) unnormalize (center_x, center_y, width, height)
    bboxes[..., [0, 2]] *= IMAGE_WIDTH
    bboxes[..., [1, 3]] *= IMAGE_HEIGHT

    # 2) convert center point to top-left and bottom-right of box coordinates
    x, y, w, h = bboxes.unbind(-1)

    xyxy = torch.stack([
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    ], dim=-1)

    # 3) if the conversion is for drawing bounding boxes, then round
    if draw:
        return xyxy.round().to(torch.uint8)

    return xyxy

def area(bbox):
    # compute area for one bbox or a tensor full of them
    w = torch.clamp(bbox[..., 2] - bbox[..., 0], min=0)
    h = torch.clamp(bbox[..., 3] - bbox[..., 1], min=0)

    return w * h

def iou(bbox_1, bbox_2):
    # 1) find coordinates of box that intersects both boxes
    intersect_coords = intersection_coords(bbox_1, bbox_2)

    # 2) compute union
    union_area = area(bbox_1) + area(bbox_2) - area(intersect_coords)

    # 3) compute IoU
    return area(intersect_coords) / union_area if union_area != 0 else 0

def giou(bbox_1, bbox_2):
    # 1) compute IoU
    bbox_IoU = iou(bbox_1, bbox_2)

    # 2) find coordinates of box that encloses both boxes
    rectangle_coords = enclose_coords(bbox_1, bbox_2)

    # 3) find coordinates of box that intersects both boxes
    intersect_coords = intersection_coords(bbox_1, bbox_2)

    # 4) compute union
    union_area = area(bbox_1) + area(bbox_2) - area(intersect_coords)

    # 5) compute GIoU
    rectangle_area = area(rectangle_coords)

    return bbox_IoU - ((rectangle_area - union_area) / rectangle_area)