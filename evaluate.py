from dataloader import DataLoader
from model import *
from dataloader_utils import *
from yolo_utils import *
from constants import *
import numpy as np
import copy
import os
import cv2
import torch

def compute_mAP(pred_boxes, GT_boxes, iou_threshold = 0.5):
    # pred_boxes: list of box list of a single image
    # GT_boxes: list of box list of a single image

    # rearrange the pred_boxes and GT_boxes
    pred_boxes_class = {cls: [] for cls in CLASSES}
    GT_boxes_class = {cls: [] for cls in CLASSES}
    pred = copy.deepcopy(pred_boxes)
    GT = copy.deepcopy(GT_boxes)
    for i in range(len(pred)):
        boxes = pred[i]
        for box in boxes:
            box["idx"] = i
            pred_boxes_class[box["class"]].append(box)
    for i in range(len(GT)):
        boxes = GT[i]
        for box in boxes:
            box["idx"] = i
            GT_boxes_class[box["class"]].append(box)

    APs = {cls: None for cls in CLASSES}
    for cls in CLASSES:
        if len(GT_boxes_class[cls]) == 0:
            continue
        if len(pred_boxes_class[cls]) == 0:
            APs[cls] = 0
            continue
        TP, FP = 0, 0
        total_gt_boxes = len(GT_boxes_class[cls])
        precision, recall = [1], [0]

        pred_boxes_class[cls].sort(key=lambda x: x["confidence"], reverse=True)

        for box in pred_boxes_class[cls]:
            iou_max = 0
            for GT_box in GT[box["idx"]]:
                iou_score = iou(box, GT_box)
                if iou_score > iou_max:
                    iou_max = iou_score
                    matched_GT_box = GT_box

            if iou_max > iou_threshold:
                TP += 1
                GT[box["idx"]].remove(matched_GT_box)
            else:
                FP += 1

            precision.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
            recall.append(TP / total_gt_boxes if total_gt_boxes > 0 else 0)

        # Calculate AP using trapezoidal rule
        precision = np.array(precision)
        recall = np.array(recall)
        AP = np.sum((recall[1:] - recall[:-1]) * precision[:-1])
        APs[cls] = AP
        print("AP for class {}: {}".format(cls, AP))
        print("precision: {}, recall: {}".format(precision[-1], recall[-1]))

    APs = {k: v for k, v in APs.items() if v is not None}
    mAP = np.mean(np.array(list(APs.values())))
    return mAP, APs

def plot_on_test(test_images, test_boxes, pred_boxes, image_names, plot_dir):

    pred_dir = os.path.join(plot_dir, "pred")
    gt_dir = os.path.join(plot_dir, "gt")
    # check if the directory exists
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    
    for i in range(len(test_images)):
        # image with predicted bounding boxes
        image_wpb = plot_bbox_on_image(test_images[i], pred_boxes[i])
        image_name = image_names[i]
        image_path = os.path.join(pred_dir, image_name + ".jpg")
        cv2.imwrite(image_path, image_wpb)
        # image with ground truth bounding boxes
        image_wgb = plot_bbox_on_image(test_images[i], test_boxes[i])
        image_path = os.path.join(gt_dir, image_name + ".jpg")
        cv2.imwrite(image_path, image_wgb)

def plot_on_train(train_images, train_boxes, pred_boxes, image_names, plot_dir, max_n = 100):
    
    pred_dir = os.path.join(plot_dir, "pred")
    gt_dir = os.path.join(plot_dir, "gt")
    # check if the directory exists
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir)
    if not os.path.exists(gt_dir):
        os.makedirs(gt_dir)
    
    for i in range(len(train_images)):
        # image with predicted bounding boxes
        image_wpb = plot_bbox_on_image(train_images[i], pred_boxes[i])
        image_name = image_names[i]
        image_path = os.path.join(pred_dir, image_name + ".jpg")
        cv2.imwrite(image_path, image_wpb)
        # image with ground truth bounding boxes
        image_wgb = plot_bbox_on_image(train_images[i], train_boxes[i])
        image_path = os.path.join(gt_dir, image_name + ".jpg")
        cv2.imwrite(image_path, image_wgb)
        if i == max_n:
            break


def main(is_plot=True, is_mAP=True, print_plot=100, max_train = 300):
    """
    Evaluate the model on the test set
    """
    dl = DataLoader()
    dl.load_data()

    net = Net()
    net.load_weights("weights225-1226.pth")

    train_images = dl.train_images[:max_train]
    train_boxes = [msg["boxes"] for msg in dl.train_msg[:max_train]]
    train_names = [msg["image_name"] for msg in dl.train_msg[:max_train]]
    test_images = dl.test_images
    test_boxes = [msg["boxes"] for msg in dl.test_msg]
    test_names = [msg["image_name"] for msg in dl.test_msg]
    
    # predict on train set
    pred_boxes_list_train = net.predict(train_images)
    # predict on test set
    pred_boxes_list_test = net.predict(test_images)
    
    if is_plot:
        print("Plotting on train set...")
        plot_on_train(train_images, train_boxes, pred_boxes_list_train, train_names, TRAIN_PLOT_DIR, max_n=max_train)
        print("Plotting on test set...")
        plot_on_test(test_images, test_boxes, pred_boxes_list_test, test_names, TEST_PLOT_DIR)
    
    if is_mAP:
        print("Computing mAP...")
        mAP_train, APs_train = compute_mAP(pred_boxes_list_train, train_boxes)
        mAP_test, APs_test = compute_mAP(pred_boxes_list_test, test_boxes)
        print("mAP on train set: {}".format(mAP_train))
        print("mAP on test set: {}".format(mAP_test))
        print("APs on train set: {}".format(APs_train))
        print("APs on test set: {}".format(APs_test))
        
if __name__ == "__main__":
    main(is_plot=False, is_mAP=True, print_plot=100, max_train=10000)

