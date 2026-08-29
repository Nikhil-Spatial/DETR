import torch

def cxcywh_to_xyxy(bboxes, draw=False):
    # convert center point to top-left and bottom-right of box coordinates
    x, y, w, h = bboxes.unbind(-1)

    xyxy = torch.stack([
        x - w / 2,
        y - h / 2,
        x + w / 2,
        y + h / 2
    ], dim=-1)

    # if the conversion is for drawing bounding boxes, then round
    if draw:
        return xyxy.round().to(torch.uint8)

    return xyxy

def compute_area(bbox):
    w = (bbox[..., 2] - bbox[..., 0]).clamp(min=0)
    h = (bbox[..., 3] - bbox[..., 1]).clamp(min=0)

    return w * h

def pairwise_giou_cxcywh(boxes_1, boxes_2):
    # 1) convert (cx, cy, w, h) to (x1, y1, x2, y2)
    boxes_1 = cxcywh_to_xyxy(boxes_1)
    boxes_2 = cxcywh_to_xyxy(boxes_2)

    # 2) find intersection coords, and compute area of intersection box
    inter_x1 = torch.max(boxes_1[:, None, 0], boxes_2[None, :, 0])
    inter_y1 = torch.max(boxes_1[:, None, 1], boxes_2[None, :, 1])
    inter_x2 = torch.min(boxes_1[:, None, 2], boxes_2[None, :, 2])
    inter_y2 = torch.min(boxes_1[:, None, 3], boxes_2[None, :, 3])

    intersection_coords = torch.stack([
        inter_x1,
        inter_y1,
        inter_x2,
        inter_y2
    ], dim=-1)

    intersection = compute_area(intersection_coords)

    # 3) compute union
    union_ = (compute_area(boxes_1)[:, None] +
              compute_area(boxes_2)[None, :] -
              intersection)

    # 4) compute area of rectangle that encloses both bboxes
    rect_x1 = torch.min(boxes_1[:, None, 0], boxes_2[None, :, 0])
    rect_y1 = torch.min(boxes_1[:, None, 1], boxes_2[None, :, 1])
    rect_x2 = torch.max(boxes_1[:, None, 2], boxes_2[None, :, 2])
    rect_y2 = torch.max(boxes_1[:, None, 3], boxes_2[None, :, 3])

    rectangle_coords = torch.stack([
        rect_x1,
        rect_y1,
        rect_x2,
        rect_y2
    ], dim=-1)

    rectangle_area = compute_area(rectangle_coords)

    # 5) compute GIoU and return
    return ((intersection / union_.clamp(min=1e-7)) -
            ((rectangle_area - union_) / rectangle_area.clamp(min=1e-7)))