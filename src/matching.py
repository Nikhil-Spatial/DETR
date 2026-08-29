import torch.nn.functional as F
from src.utilities import pairwise_giou_cxcywh
from src.configs import LAMBDA_CLS, LAMBDA_L1,  LAMBDA_GIOU
import torch

def compute_cls_cost(class_preds, truth_classes):
    # negate the correct class probabilities
    class_costs = [
        -F.softmax(class_preds, dim=-1).select(-1, int(idx)).unsqueeze(-1)
        for idx in truth_classes
    ]

    return torch.cat(class_costs, dim=-1)

def compute_l1_cost(bbox_preds, truth_boxes):
    # compute l1 norms of the vectors containing the distance between center
    # points
    l1_costs = [(bbox_preds - truth_boxes[i]).abs().sum(-1, True)
                for i in range(truth_boxes.shape[0])]

    return torch.cat(l1_costs, dim=-1)

def compute_giou_cost(bbox_preds, truth_boxes):
    # negate the giou
    return -pairwise_giou_cxcywh(bbox_preds, truth_boxes)

def hungarian_match_costs(class_preds, bbox_preds, truth_labels):
    # last dimension is only ground truth class labels
    truth_classes = truth_labels[..., 0]

    # last dimension is only ground truth boxes of (cx, cy, w, h)
    truth_boxes = truth_labels[..., -4:]

    # compute costs
    cls_cost = compute_cls_cost(class_preds, truth_classes)
    l1_cost = compute_l1_cost(bbox_preds, truth_boxes)
    giou_cost = compute_giou_cost(bbox_preds, truth_boxes)

    # combine costs scaled by their respective hyperparameters
    return LAMBDA_CLS*cls_cost + LAMBDA_L1*l1_cost + LAMBDA_GIOU*giou_cost