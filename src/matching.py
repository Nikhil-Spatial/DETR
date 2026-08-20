import torch.nn.functional as F
from utilities import cxcywh_to_xyxy
from configs import N
import torch

def classification_cost(class_preds, truth_labels):
    class_costs = [
        -F.softmax(class_preds, dim=-1).select(-1, idx).unsqueeze(-1)
        for idx in range(truth_labels.shape[0])
    ]

    return torch.cat(class_costs, dim=-1)

def l1_cost(bbox_preds, truth_labels):
    # convert truth labels from (cx, cy, w, h) to (x1, y1, x2, y2)
    truth_labels[..., [1, 3]] *= 224
    truth_labels[..., [2, 4]] *= 224


    # compute l1 norms of the vectors containing the distance between
    l1_costs = [(bbox_preds - truth_labels[i][1:]).abs().sum(-1, True)
                for i in range(truth_labels.shape[0])]

def giou_cost():
    pass

def hungarian_match_cost(class_preds, bbox_preds, truth_labels):
    pass


