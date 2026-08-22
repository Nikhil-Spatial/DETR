from torch import round as rd
from src.configs import IMAGE_WIDTH, IMAGE_HEIGHT
import torch

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

def compute_area(bbox):
    w = (bbox[..., 2] - bbox[..., 0]).clamp(min=0)
    h = (bbox[..., 3] - bbox[..., 1]).clamp(min=0)

    return w * h

def compute_giou(bbox_1, bbox_2):
    # 1) find intersection coords, and compute area of intersection box
    inter_x1 = torch.max(bbox_1[:, :, None, 0], bbox_2[:, None, :, 0])
    inter_y1 = torch.max(bbox_1[:, :, None, 1], bbox_2[:, None, :, 1])
    inter_x2 = torch.min(bbox_1[:, :, None, 2], bbox_2[:, None, :, 2])
    inter_y2 = torch.min(bbox_1[:, :, None, 3], bbox_2[:, None, :, 3])

    intersection_coords = torch.cat([
        inter_x1[:, :, :, None],
        inter_y1[:, :, :, None],
        inter_x2[:, :, :, None],
        inter_y2[:, :, :, None]
    ], dim=-1)

    intersection = compute_area(intersection_coords)

    # 2) compute union
    union_ = compute_area(bbox_1) + compute_area(bbox_2) - intersection

    # 3) compute area of rectangle that encloses both bboxes
    rect_x1 = torch.min(bbox_1[:, :, None, 0], bbox_2[:, None, :, 0])
    rect_y1 = torch.min(bbox_1[:, :, None, 1], bbox_2[:, None, :, 1])
    rect_x2 = torch.max(bbox_1[:, :, None, 2], bbox_2[:, None, :, 2])
    rect_y2 = torch.max(bbox_1[:, :, None, 3], bbox_2[:, None, :, 3])

    rectangle_coords = torch.cat([
        rect_x1[:, :, :, None],
        rect_y1[:, :, :, None],
        rect_x2[:, :, :, None],
        rect_y2[:, :, :, None]
    ], dim=-1)

    rectangle_area = compute_area(rectangle_coords)

    # compute GIoU and return
    return (intersection / union_) - ((rectangle_area - union_) /
                                      rectangle_area)