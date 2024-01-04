from constants import *
from yolo_utils import *
import torch
import torch.nn.functional as F

def yolo_loss_sq(pred, target, lambd_coord = 5, lambd_noobj = 0.5):
    """
    Squared loss for YOLO
    Not vectorized. the speed is acceptable.
    lambds are the weight for different losses
    """

    loss = 0
    coord_loss = 0
    obj_loss = 0
    noobj_loss = 0
    class_loss = 0
    batch_size = pred.size()[0]
    n_grid_side = N_GRID_SIDE
    # loop over the batch, grid, and bounding box
    for i in range(batch_size):
        for j in range(n_grid_side):
            for k in range(n_grid_side):
                if target[i, j, k, 4] == 1:
                    # calculate iou for both bounding boxes
                    bbox1_pred = {
                        "xmin": (pred[i, j, k, 0] + j) / n_grid_side - pred[i, j, k, 2] / 2,
                        "ymin": (pred[i, j, k, 1] + k) / n_grid_side - pred[i, j, k, 3] / 2,
                        "xmax": (pred[i, j, k, 0] + j) / n_grid_side + pred[i, j, k, 2] / 2,
                        "ymax": (pred[i, j, k, 1] + k) / n_grid_side + pred[i, j, k, 3] / 2
                    }
                    bbox2_pred = {
                        "xmin": (pred[i, j, k, 5] + j) / n_grid_side - pred[i, j, k, 7] / 2,
                        "ymin": (pred[i, j, k, 6] + k) / n_grid_side - pred[i, j, k, 8] / 2,
                        "xmax": (pred[i, j, k, 5] + j) / n_grid_side + pred[i, j, k, 7] / 2,
                        "ymax": (pred[i, j, k, 6] + k) / n_grid_side + pred[i, j, k, 8] / 2
                    }
                    bbox_target = {
                        "xmin": (target[i, j, k, 0] + j) / n_grid_side - target[i, j, k, 2] / 2,
                        "ymin": (target[i, j, k, 1] + k) / n_grid_side - target[i, j, k, 3] / 2,
                        "xmax": (target[i, j, k, 0] + j) / n_grid_side + target[i, j, k, 2] / 2,
                        "ymax": (target[i, j, k, 1] + k) / n_grid_side + target[i, j, k, 3] / 2
                    }
                    iou1 = iou(bbox1_pred, bbox_target)
                    iou2 = iou(bbox2_pred, bbox_target)
                    # use the bounding box with higher iou to calculate object loss and coordinate loss
                    # the lower iou is used to calculate no-object loss
                    if iou1 > iou2:
                        coord_loss = coord_loss + lambd_coord * (\
                            torch.sum((pred[i, j, k, 0:2] - target[i, j, k, 0:2]) ** 2) +\
                            torch.sum((pred[i, j, k, 2:4].sqrt() - target[i, j, k, 2:4].sqrt()) ** 2))
                        obj_loss = obj_loss + (pred[i, j, k, 4] - iou1) ** 2
                        noobj_loss = noobj_loss + lambd_noobj * (pred[i, j, k, 9] - iou2) ** 2
                    else:
                        coord_loss = coord_loss + lambd_coord * (\
                            torch.sum((pred[i, j, k, 5:7] - target[i, j, k, 5:7]) ** 2) +\
                            torch.sum((pred[i, j, k, 7:9].sqrt() - target[i, j, k, 7:9].sqrt()) ** 2))
                        obj_loss = obj_loss + (pred[i, j, k, 9] - iou2) ** 2
                        noobj_loss = noobj_loss + lambd_noobj * (pred[i, j, k, 4] - iou1) ** 2
                    # class loss
                    class_loss = class_loss + torch.sum((pred[i, j, k, 10:] - target[i, j, k, 10:]) ** 2)
                else:
                    # if gt is not present, calculate no-object loss
                    noobj_loss = noobj_loss + lambd_noobj * (pred[i, j, k, 4] - target[i, j, k, 4]) ** 2
                    noobj_loss = noobj_loss + lambd_noobj * (pred[i, j, k, 9] - target[i, j, k, 9]) ** 2

    loss = coord_loss + obj_loss + noobj_loss + class_loss
    return loss / batch_size


def yolo_loss_sqvec(pred, target, lambd_coord=5, lambd_noobj=0.5):
    """
    vectorization attempt but failed
    """
    batch_size, n_grid_side = pred.size(0), pred.size(1)
    
    # Extract coordinates and sizes from prediction and target tensors
    pred_x = pred[:, :, :, 0]
    pred_y = pred[:, :, :, 1]
    pred_w = pred[:, :, :, 2]
    pred_h = pred[:, :, :, 3]
    
    pred_x2 = pred[:, :, :, 5]
    pred_y2 = pred[:, :, :, 6]
    pred_w2 = pred[:, :, :, 7]
    pred_h2 = pred[:, :, :, 8]
    
    target_x = target[:, :, :, 0]
    target_y = target[:, :, :, 1]
    target_w = target[:, :, :, 2]
    target_h = target[:, :, :, 3]
    
    # Calculate bounding box coordinates
    bbox1_pred_xmin = (pred_x + torch.arange(n_grid_side).view(1, 1, n_grid_side).float()) / n_grid_side - pred_w / 2
    bbox1_pred_ymin = (pred_y + torch.arange(n_grid_side).view(1, n_grid_side, 1).float()) / n_grid_side - pred_h / 2
    bbox1_pred_xmax = (pred_x + torch.arange(n_grid_side).view(1, 1, n_grid_side).float()) / n_grid_side + pred_w / 2
    bbox1_pred_ymax = (pred_y + torch.arange(n_grid_side).view(1, n_grid_side, 1).float()) / n_grid_side + pred_h / 2
    
    bbox2_pred_xmin = (pred_x2 + torch.arange(n_grid_side).view(1, 1, n_grid_side).float()) / n_grid_side - pred_w2 / 2
    bbox2_pred_ymin = (pred_y2 + torch.arange(n_grid_side).view(1, n_grid_side, 1).float()) / n_grid_side - pred_h2 / 2
    bbox2_pred_xmax = (pred_x2 + torch.arange(n_grid_side).view(1, 1, n_grid_side).float()) / n_grid_side + pred_w2 / 2
    bbox2_pred_ymax = (pred_y2 + torch.arange(n_grid_side).view(1, n_grid_side, 1).float()) / n_grid_side + pred_h2 / 2
    
    bbox_target_xmin = (target_x + torch.arange(n_grid_side).view(1, 1, n_grid_side).float()) / n_grid_side - target_w / 2
    bbox_target_ymin = (target_y + torch.arange(n_grid_side).view(1, n_grid_side, 1).float()) / n_grid_side - target_h / 2
    bbox_target_xmax = (target_x + torch.arange(n_grid_side).view(1, 1, n_grid_side).float()) / n_grid_side + target_w / 2
    bbox_target_ymax = (target_y + torch.arange(n_grid_side).view(1, n_grid_side, 1).float()) / n_grid_side + target_h / 2
    
    # Calculate IoU for both predicted bounding boxes
    iou1 = iou_vectorized(bbox1_pred_xmin, bbox1_pred_ymin, bbox1_pred_xmax, bbox1_pred_ymax,
                          bbox_target_xmin, bbox_target_ymin, bbox_target_xmax, bbox_target_ymax)
    
    iou2 = iou_vectorized(bbox2_pred_xmin, bbox2_pred_ymin, bbox2_pred_xmax, bbox2_pred_ymax,
                          bbox_target_xmin, bbox_target_ymin, bbox_target_xmax, bbox_target_ymax)
    
    iou_mask = iou1 > iou2
    
    # Mask for objects present in the target
    obj_mask = target[:, :, :, 4] == 1
    
    # Mask for objects not present in the target
    noobj_mask = ~obj_mask
    
    # Calculate coordinate loss, object loss, no-object loss, and class loss
    coord_loss = lambd_coord * (\
        torch.sum((pred[:, :, :, 0:2] - target[:, :, :, 0:2]) ** 2 * obj_mask.unsqueeze(-1) * iou_mask.unsqueeze(-1)) +\
        torch.sum((pred[:, :, :, 2:4].sqrt() - target[:, :, :, 2:4].sqrt()) ** 2 * obj_mask.unsqueeze(-1) * iou_mask.unsqueeze(-1)) +\
        torch.sum((pred[:, :, :, 5:7] - target[:, :, :, 5:7]) ** 2 * obj_mask.unsqueeze(-1) * ~iou_mask.unsqueeze(-1)) +\
        torch.sum((pred[:, :, :, 7:9].sqrt() - target[:, :, :, 7:9].sqrt()) ** 2 * obj_mask.unsqueeze(-1) * ~iou_mask.unsqueeze(-1)))

    obj_loss = torch.sum((pred[:, :, :, 4] - iou1) ** 2 * obj_mask.unsqueeze(-1) * iou_mask.unsqueeze(-1)) +\
                torch.sum((pred[:, :, :, 9] - iou2) ** 2 * obj_mask.unsqueeze(-1) * ~iou_mask.unsqueeze(-1))
    
    noobj_loss = lambd_noobj * (\
                torch.sum((pred[:, :, :, 4] - iou1) ** 2 * noobj_mask.unsqueeze(-1) * iou_mask.unsqueeze(-1)) +\
                torch.sum((pred[:, :, :, 9] - iou2) ** 2 * noobj_mask.unsqueeze(-1) * ~iou_mask.unsqueeze(-1)))
    
    noobj_loss2 = lambd_noobj * (\
                torch.sum((pred[:, :, :, 4] - target[:, :, :, 4]) ** 2 * noobj_mask.unsqueeze(-1) * ~iou_mask.unsqueeze(-1)) +\
                torch.sum((pred[:, :, :, 9] - target[:, :, :, 9]) ** 2 * noobj_mask.unsqueeze(-1) * iou_mask.unsqueeze(-1)))

    class_loss = torch.sum((pred[:, :, :, 10:] - target[:, :, :, 10:]) ** 2 * obj_mask.unsqueeze(-1))
        
    # Sum all the losses
    loss = coord_loss + obj_loss + noobj_loss + noobj_loss2 + class_loss
    
    # Average the loss over the batch size
    return loss / batch_size

def iou_vectorized(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    """
    vectorized version of iou
    """

    # Intersection
    xmin_inter = torch.maximum(xmin1, xmin2)
    ymin_inter = torch.maximum(ymin1, ymin2)
    xmax_inter = torch.minimum(xmax1, xmax2)
    ymax_inter = torch.minimum(ymax1, ymax2)
    
    width_inter = torch.maximum(xmax_inter - xmin_inter, torch.zeros_like(xmax_inter))
    height_inter = torch.maximum(ymax_inter - ymin_inter, torch.zeros_like(ymax_inter))
    
    area_inter = width_inter * height_inter
    
    # Union
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    
    area_union = area1 + area2 - area_inter
    
    # IoU
    iou = area_inter / area_union
    
    return iou

def yolo_loss_be(pred, target, lambd_coord = 5, lambd_noobj = 0.5):
    """
    A binary cross entropy plus iou version of yolo loss
    pred: predicted label
    target: ground truth label
    lambds: weights for different losses
    It's not vectorized. The speed is acceptable.
    """

    loss = 0
    coord_loss = 0
    obj_loss = 0
    noobj_loss = 0
    class_loss = 0
    batch_size = pred.size()[0]
    n_grid_side = N_GRID_SIDE
    # loop over the batch, grid, and bounding box
    for i in range(batch_size):
        for j in range(n_grid_side):
            for k in range(n_grid_side):
                if target[i, j, k, 4] == 1:
                    # calculate iou for both bounding boxes
                    bbox1_pred = {
                        "xmin": (pred[i, j, k, 0] + j) / n_grid_side - pred[i, j, k, 2] / 2,
                        "ymin": (pred[i, j, k, 1] + k) / n_grid_side - pred[i, j, k, 3] / 2,
                        "xmax": (pred[i, j, k, 0] + j) / n_grid_side + pred[i, j, k, 2] / 2,
                        "ymax": (pred[i, j, k, 1] + k) / n_grid_side + pred[i, j, k, 3] / 2
                    }
                    bbox2_pred = {
                        "xmin": (pred[i, j, k, 5] + j) / n_grid_side - pred[i, j, k, 7] / 2,
                        "ymin": (pred[i, j, k, 6] + k) / n_grid_side - pred[i, j, k, 8] / 2,
                        "xmax": (pred[i, j, k, 5] + j) / n_grid_side + pred[i, j, k, 7] / 2,
                        "ymax": (pred[i, j, k, 6] + k) / n_grid_side + pred[i, j, k, 8] / 2
                    }
                    bbox_target = {
                        "xmin": (target[i, j, k, 0] + j) / n_grid_side - target[i, j, k, 2] / 2,
                        "ymin": (target[i, j, k, 1] + k) / n_grid_side - target[i, j, k, 3] / 2,
                        "xmax": (target[i, j, k, 0] + j) / n_grid_side + target[i, j, k, 2] / 2,
                        "ymax": (target[i, j, k, 1] + k) / n_grid_side + target[i, j, k, 3] / 2
                    }
                    iou1 = iou(bbox1_pred, bbox_target)
                    iou2 = iou(bbox2_pred, bbox_target)
                    # use the bounding box with higher iou to calculate object loss and coordinate loss
                    # NOTE: did not penalize the confidence of the other bounding box
                    if iou1 > iou2:
                        coord_loss += 1 - iou1
                        obj_loss += F.binary_cross_entropy(pred[i, j, k, 4], target[i, j, k, 4])
                    else:
                        coord_loss += 1 - iou2
                        obj_loss += F.binary_cross_entropy(pred[i, j, k, 9], target[i, j, k, 9])
                    # class loss
                    class_loss += F.binary_cross_entropy(pred[i, j, k, 10:], target[i, j, k, 10:])
                else:
                    # if gt is not present, calculate no-object loss
                    noobj_loss += F.binary_cross_entropy(pred[i, j, k, 4], target[i, j, k, 4])
                    noobj_loss += F.binary_cross_entropy(pred[i, j, k, 9], target[i, j, k, 9])
    
    loss = lambd_coord * coord_loss + obj_loss + lambd_noobj * noobj_loss + class_loss
    return loss / batch_size


def dummy_loss(pred, target):
    return torch.sum(pred - target) / pred.size()[0]

