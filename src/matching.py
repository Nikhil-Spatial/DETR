import torch.nn.functional as F
from src.utilities import pairwise_giou_cxcywh
from src.configs import LAMBDA_CLS, LAMBDA_L1,  LAMBDA_GIOU, N, C
from scipy.optimize import linear_sum_assignment
import torch

def compute_cls_cost(class_preds, gt_classes):
    # negate the correct class probabilities
    class_costs = [
        -F.softmax(class_preds, dim=-1).select(-1, int(idx)).unsqueeze(-1)
        for idx in gt_classes
    ]

    return torch.cat(class_costs, dim=-1)

def compute_l1_cost(bbox_preds, gt_boxes):
    # compute l1 norms of the vectors containing the distance between center
    # points
    l1_costs = [(bbox_preds - gt_boxes[i]).abs().sum(-1, True)
                for i in range(gt_boxes.shape[0])]

    return torch.cat(l1_costs, dim=-1)

def compute_giou_cost(bbox_preds, gt_boxes):
    # negate the giou
    return -pairwise_giou_cxcywh(bbox_preds, gt_boxes)

def hungarian_match_costs(class_preds, bbox_preds, gt_labels):
    # last dimension is only ground truth class labels
    gt_classes = gt_labels[..., 0]

    # last dimension is only ground truth boxes of (cx, cy, w, h)
    gt_boxes = gt_labels[..., -4:]

    # compute costs
    cls_cost = compute_cls_cost(class_preds, gt_classes)
    l1_cost = compute_l1_cost(bbox_preds, gt_boxes)
    giou_cost = compute_giou_cost(bbox_preds, gt_boxes)

    # combine costs scaled by their respective hyperparameters
    return LAMBDA_CLS*cls_cost + LAMBDA_L1*l1_cost + LAMBDA_GIOU*giou_cost

def hungarian_matching(class_preds, bbox_preds, truth_labels):
    # each position represents the cost of an object query's prediction with
    # a ground truth label
    cost_matrix = hungarian_match_costs(class_preds, bbox_preds, truth_labels)

    # one-to-one matching with minimum total cost
    pred_idxs, gt_idxs = linear_sum_assignment(cost)

    # create N-dim vector to represent all target matches, initializing each
    # object query's match with no_object class (idx C)
    target_matches = torch.full((N,), C)

    # match object query's with their class idx
    target_matches[pred_idxs] = gt_labels[gt_idxs, 0]

    return target_matches