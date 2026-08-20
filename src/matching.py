import torch.nn.functional as F
from configs import N
import torch

def classification_cost(class_preds, truth_labels):
    class_costs = [
        -F.softmax(class_preds, dim=-1).select(-1, idx).unsqueeze(-1)
        for idx in range(truth_labels.shape[0])
    ]

    return torch.cat(class_costs, dim=-1)

def l1_cost():
    pass

def giou_cost():
    pass

def hungarian_match_cost(class_preds, bbox_preds, truth_labels):
    pass


