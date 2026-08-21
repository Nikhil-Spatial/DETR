from torch import round as rd
from src.configs import IMAGE_WIDTH, IMAGE_HEIGHT
import torch

def compute_intersection_coords(bbox_1, bbox_2):
    """
    :param bbox_1: a tensor where the last dimension contains 4 bbox values
    :param bbox_2: a 1D tensor of 4 bbox values
    """
    intersect_boxes = torch.empty(bbox_1.shape[0], 4)

    intersect_boxes[..., 0] = torch.max(bbox_1[..., 0], bbox_2[0])
    intersect_boxes[..., 1] = torch.max(bbox_1[..., 1], bbox_2[1])
    intersect_boxes[..., 2] = torch.min(bbox_1[..., 2], bbox_2[2])
    intersect_boxes[..., 3] = torch.min(bbox_1[..., 3], bbox_2[3])

    return intersect_boxes

def compute_enclosed_coords(bbox_1, bbox_2):
    """
    :param bbox_1: a tensor where the last dimension contains 4 bbox values
    :param bbox_2: a 1D tensor of 4 bbox values
    """
    enclosed_boxes = torch.empty(bbox_1.shape[0], 4)

    enclosed_boxes[..., 0] = torch.min(bbox_1[..., 0], bbox_2[0])
    enclosed_boxes[..., 1] = torch.min(bbox_1[..., 1], bbox_2[1])
    enclosed_boxes[..., 2] = torch.max(bbox_1[..., 2], bbox_2[2])
    enclosed_boxes[..., 3] = torch.max(bbox_1[..., 3], bbox_2[3])

    return enclosed_boxes

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

def compute_iou(bbox_1, bbox_2):
    # 1) find coordinates of box that intersects both boxes
    intersection_coords(bbox_1, bbox_2)

    # 2) compute area of intersecting box
    intersection_area = area(bbox)

    # 2) compute union
    union_area = area(bbox_1) + area(bbox_2) - area(intersect_coords)

    # 3) compute IoU
    return area(intersect_coords) / union_area if union_area != 0 else 0

def compute_giou(bbox_1, bbox_2):
    """
    :param bbox_1: tensor of shape (100, 4), the last dim's 4 values are bbox
    predictions
    :param bbox_2: tensor of shape (?, 4), the last dim's 4 values are bbox
    ground truth labels
    """
    gious = []

    for i in range(bbox_2.shape[0]):
        preds_clone = bbox_1.clone().detach()

        # 1) compute IoU
        iou = compute_iou(preds_clone, bbox_2[i])


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