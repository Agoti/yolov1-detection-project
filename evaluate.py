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
import matplotlib.pyplot as plt

def compute_mAP(pred_boxes, GT_boxes, iou_threshold = 0.5, use_07_metric = True, print_pr = False, plot_pr = False):
    """
    Compute mAP
    pred_boxes: list of boxes(list) of a single image
    GT_boxes: list of boxes(list) of a single image
    """

    # split into classes
    pred_boxes_class = {cls: [] for cls in CLASSES}
    GT_boxes_class = {cls: [] for cls in CLASSES}

    # deep copy
    pred = copy.deepcopy(pred_boxes)
    GT = copy.deepcopy(GT_boxes)

    # add index of image to each box
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

    # calculate AP for each class
    APs = {cls: None for cls in CLASSES}
    for cls in CLASSES:

        # skip if no ground truth boxes
        if len(GT_boxes_class[cls]) == 0:
            continue
        # skip if no predicted boxes
        if len(pred_boxes_class[cls]) == 0:
            APs[cls] = 0
            continue

        # Initialize
        TP, FP = 0, 0
        total_gt_boxes = len(GT_boxes_class[cls])
        precision, recall = [1], [0]
        # sort by confidence
        pred_boxes_class[cls].sort(key=lambda x: x["confidence"], reverse=True)

        # loop over predicted boxes
        for box in pred_boxes_class[cls]:
            iou_max = 0

            # find the ground truth box with the highest iou
            for GT_box in GT[box["idx"]]:
                iou_score = iou(box, GT_box)
                if iou_score > iou_max:
                    iou_max = iou_score
                    matched_GT_box = GT_box
            # if iou is greater than threshold, it is a true positive
            if iou_max > iou_threshold:
                TP += 1
                # remove the matched ground truth box
                GT[box["idx"]].remove(matched_GT_box)
            else:
                # false positive
                FP += 1

            # calculate precision and recall
            precision.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
            recall.append(TP / total_gt_boxes if total_gt_boxes > 0 else 0)

        # Calculate AP
        precision = np.array(precision)
        recall = np.array(recall)
        if use_07_metric:
            # 11 point interpolation
            AP = 0
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(recall >= t) == 0:
                    p = 0
                else:
                    p = np.max(precision[recall >= t])
                AP += p / 11
        else:
            # every point interpolation
            AP = np.sum((recall[1:] - recall[:-1]) * precision[:-1])

        APs[cls] = AP
        # print precision and recall
        if print_pr:
            print("AP for class {}: {:.4f}".format(cls, AP))
            print("precision: {:.4f}, recall: {:.4f}".format(precision[-1], recall[-1]))
        # plot precision and recall
        if plot_pr:
            plt.plot(recall, precision)
            plt.xlabel("recall")
            plt.ylabel("precision")
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.title("PR curve")
            plt.show()

    APs = {k: v for k, v in APs.items() if v is not None}
    mAP = np.mean(np.array(list(APs.values())))

    return mAP, APs

def plot_on_test(test_images, test_boxes, pred_boxes, image_names, plot_dir):
    """
    Plot boxes on test images
    """

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
    """
    Plot boxes on train images
    """
    
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

if __name__ == "__main__":

    def main(is_plot=True, is_mAP=True, is_train=False, is_test=True, max_train=300):
        """
        Evaluate the model on the test set
        """
        dl = DataLoader()
        dl.load_data()

        net = Net()
        net.load_weights("checkpoints/weights220.pth")

        if is_train:
            train_images = [label["resized_image"] for label in dl.train_labels]
            train_boxes = [label["resized_boxes"] for label in dl.train_labels]
            train_names = [label["image_name"] for label in dl.train_labels]

            # predict on train set
            pred_boxes_list_train = net.predict(train_images)

            if is_plot:
                print("Plotting on train set...")
                plot_on_train(train_images, train_boxes, pred_boxes_list_train, train_names, TRAIN_PLOT_DIR, max_n=max_train)
            
            if is_mAP:
                print("Computing train mAP...")
                mAP_train, APs_train = compute_mAP(pred_boxes_list_train, train_boxes, print_pr=True, plot_pr=True)
                print("mAP on train set: {}".format(mAP_train))
                print("APs on train set: {}".format(APs_train))

        if is_test:
            test_images = dl.test_images
            test_boxes = [label["boxes"] for label in dl.test_labels]
            test_names = [label["image_name"] for label in dl.test_labels]

            # predict on test set
            pred_boxes_list_test = net.predict(test_images)
        
            if is_plot:
                print("Plotting on test set...")
                plot_on_test(test_images, test_boxes, pred_boxes_list_test, test_names, TEST_PLOT_DIR)
        
            if is_mAP:
                print("Computing test mAP...")
                mAP_test, APs_test = compute_mAP(pred_boxes_list_test, test_boxes, print_pr=True, plot_pr=True)
                print("mAP on test set: {}".format(mAP_test))
                print("APs on test set: {}".format(APs_test))
        
    main(is_plot=False, is_mAP=True, max_train=100)

