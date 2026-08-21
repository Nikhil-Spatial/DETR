from torch import round as rd
from src.configs import IMAGE_WIDTH, IMAGE_HEIGHT
import torch

def compute_intersection_coords(bbox_1, bbox_2):
    intersect_boxes = torch.empty(bbox_1.shape[0], 4)

    intersect_boxes[..., 0] = torch.max(bbox_1[..., 0], bbox_2[0])
    intersect_boxes[..., 1] = torch.max(bbox_1[..., 1], bbox_2[1])
    intersect_boxes[..., 2] = torch.min(bbox_1[..., 2], bbox_2[2])
    intersect_boxes[..., 3] = torch.min(bbox_1[..., 3], bbox_2[3])

    return intersect_boxes

def compute_enclosed_coords(bbox_1, bbox_2):
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

def compute_area(bbox):
    # compute area for one bbox or a tensor full of them
    w = torch.clamp(bbox[..., 2] - bbox[..., 0], min=0)
    h = torch.clamp(bbox[..., 3] - bbox[..., 1], min=0)

    return w * h

def compute_giou(bbox_1, bbox_2):
    giou_costs = []

    for i in range(bbox_2.shape[0]):
        # 1) find coordinates of box that encloses both boxes, and compute its
        # area
        rectangle_coords = compute_enclosed_coords(bbox_1, bbox_2[i])
        rectangle_area = compute_area(rectangle_coords)

        # 2) find coordinates of box that intersects both boxes, and compute
        # its area
        intersection_coords = compute_intersection_coords(
            bbox_1, bbox_2[i]
        )
        intersection_area = compute_area(intersection_coords)

        # 3) compute union
        union_area = (compute_area(bbox_1) + compute_area(bbox_2[i])
                      - compute_area(intersection_coords))

        # 4) compute IoU
        iou = intersection_area / union_area

        # 5) compute GIoU
        giou_costs.append(
            iou - ((rectangle_area - union_area) / rectangle_area)
        )

    return torch.cat(giou_costs, dim=-1)