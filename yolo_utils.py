from constants import *
from dataloader import *
from data_augmentation import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
import cv2
import copy

def bbox_to_center(box):

    # box: box dictionary
    # return: augmented box dictionary
    bbox = box
    xmin = bbox["xmin"]
    ymin = bbox["ymin"]
    xmax = bbox["xmax"]
    ymax = bbox["ymax"]
    width = bbox["width"]
    height = bbox["height"]
    # convert
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    # update
    bbox["cx"] = cx
    bbox["cy"] = cy
    bbox["w"] = w
    bbox["h"] = h
    return bbox

def bbox_to_corners(box):
    # box: dictionary with keys (cx, cy, w, h)
    #   but keys (xmin, ymin, xmax, ymax) are 0 currently
    
    bbox = box.copy()
    cx, cy, w, h = bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]
    width = bbox["width"]
    height = bbox["height"]
    xmin = int((cx - w / 2) * width)
    ymin = int((cy - h / 2) * height)
    xmax = int((cx + w / 2) * width)
    ymax = int((cy + h / 2) * height)
    bbox["xmin"] = xmin
    bbox["ymin"] = ymin
    bbox["xmax"] = xmax
    bbox["ymax"] = ymax
    return bbox

# convert bounding boxes to labels for 
# yolo training
def box2label(boxes):
    
    grid_side = 1.0 / N_GRID_SIDE
    # currently without anchor boxes
    labels = np.zeros((N_GRID_SIDE, N_GRID_SIDE, 5 * N_BBOX + N_CLASSES), dtype=np.float32)
    for i in range(len(boxes)):
        box = boxes[i]
        cx = box["cx"]
        cy = box["cy"]
        w = box["w"]
        h = box["h"]
        cls = box["class"]
        # find grid cell, NOTE: cx and cy are normalized
        grid_x = int(cx / grid_side)
        grid_y = int(cy / grid_side)
        if grid_x == N_GRID_SIDE or grid_y == N_GRID_SIDE:
            grid_x = N_GRID_SIDE - 1
            grid_y = N_GRID_SIDE - 1
        # find offset
        offset_x = cx / grid_side - grid_x
        offset_y = cy / grid_side - grid_y
        # find label
        label = np.zeros(5 * N_BBOX + N_CLASSES, dtype=np.float32)
        for j in range(N_BBOX):
            label[5 * j: 5 * j + 5] = [offset_x, offset_y, w, h, 1]
        label[5 * N_BBOX + CLASSES.index(cls)] = 1
        # update labels
        labels[grid_y, grid_x, :] = label[:]
    
    labels = labels.reshape(1, -1)
    return labels

def label2box(label, threshold = 0.2):
    # label: numpy array of shape (N_GRID_SIDE, N_GRID_SIDE, 5 * N_BBOX + N_CLASSES)
    boxes = []
    label_confidence = np.zeros((N_GRID_SIDE, N_GRID_SIDE, N_BBOX), dtype=np.float32)
    label_confidence[:, :, :] = label[:, :, 4:4 + 5 * N_BBOX:5]
    label_class_prob = label[:, :, 5 * N_BBOX:]
    label_score = label_confidence * np.max(label_class_prob, axis=2)[..., np.newaxis]
    max_score = np.max(label_score, axis=2)
    max_idx = np.argmax(label_score, axis=2)
    filtering_mask = max_score > threshold
    for i in range(N_GRID_SIDE):
        for j in range(N_GRID_SIDE):
            if filtering_mask[i, j]:
                b = max_idx[i, j]
                cx = (j + label[i, j, 5 * b]) / N_GRID_SIDE
                cy = (i + label[i, j, 5 * b + 1]) / N_GRID_SIDE
                w = label[i, j, 5 * b + 2]
                h = label[i, j, 5 * b + 3]
                cls = np.argmax(label_class_prob[i, j])
                confidence = label_confidence[i, j, b]
                box = {"cx": cx, "cy": cy, "w": w, "h": h,\
                        "class": CLASSES[cls], "confidence": confidence,\
                        "width": INPUT_SIDE, "height": INPUT_SIDE}
                box = bbox_to_corners(box)
                boxes.append(box)

    return boxes


def get_train_batches(train_images, train_labels, batch_size = BATCH_SIZE):
    indices = np.arange(len(train_images))
    np.random.shuffle(indices)
    train_images_numpy = np.array(train_images)
    train_labels_numpy = np.array(train_labels)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i: i + batch_size]
        batch_images = train_images_numpy[batch_indices]
        batch_labels = train_labels_numpy[batch_indices]
        yield batch_images, batch_labels

def iou(box1, box2):
    # box: dictionary with keys (xmin, ymin, xmax, ymax)
    
    x1min, y1min, x1max, y1max = box1["xmin"], box1["ymin"], box1["xmax"], box1["ymax"]
    x2min, y2min, x2max, y2max = box2["xmin"], box2["ymin"], box2["xmax"], box2["ymax"]
    x1 = max(x1min, x2min)
    y1 = max(y1min, y2min)
    x2 = min(x1max, x2max)
    y2 = min(y1max, y2max)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (x1max - x1min) * (y1max - y1min) + (x2max - x2min) * (y2max - y2min) - inter_area
    return inter_area / union_area

def non_max_suppression(boxes, iou_threshold = 0.5, max_boxes = 10):
    
    # copy the list
    original_boxes = boxes.copy()
    keep_boxes = []
    for cls in CLASSES:
        boxes_of_class = [box for box in original_boxes if box["class"] == cls]
        while boxes_of_class and len(keep_boxes) < max_boxes:
            max_confidence_box = max(boxes_of_class, key=lambda x: x["confidence"])
            keep_boxes.append(max_confidence_box)
            boxes_of_class.remove(max_confidence_box)
            for box in boxes_of_class:
                if iou(max_confidence_box, box) > iou_threshold:
                    boxes_of_class.remove(box)
    
    return keep_boxes



def plot_bbox_on_image(image, boxes):
    # image: ndarray of shape (height, width, 3)
    # boxes: list of box dictionary
    # copy the image
    image = image.copy()
    for box in boxes:
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]
        # mark bounding box and class name
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(image, box["class"] + " " + str(round(box["confidence"], 2)), (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image
