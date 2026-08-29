import torch.nn.functional as F
from src.utilities import pairwise_giou_cxcywh
from src.configs import N
import torch

def classification_cost(class_preds, truth_labels):
    # negate the correct class probabilities
    class_costs = [
        -F.softmax(class_preds, dim=-1).select(-1, int(idx)).unsqueeze(-1)
        for idx in truth_labels
    ]

    return torch.cat(class_costs, dim=-1)

def l1_cost(bbox_preds, truth_boxes):
    # compute l1 norms of the vectors containing the distance between center
    # points
    l1_costs = [(bbox_preds - truth_boxes[i][:]).abs().sum(-1, True)
                for i in range(truth_boxes.shape[0])]

    return torch.cat(l1_costs, dim=-1)

def giou_cost(bbox_preds, truth_boxes):
    # negate the giou
    return -pairwise_giou_cxcywh(bbox_preds, truth_boxes)

def hungarian_match_cost(class_preds, bbox_preds, truth_labels):

    cls_cost = classification_cost(class_preds, truth_labels[:, 0])


