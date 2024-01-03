import os
import cv2
import xml.etree.ElementTree as ET
import numpy as np
from constants import *
from yolo_utils import *


def get_classes():
    # loop through annotation files
    classes = []
    annotation_files = os.listdir(ANNOTATION_DIR)
    for annotation_file in annotation_files:
        annotation_path = os.path.join(ANNOTATION_DIR, annotation_file)
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        for obj in root.iter('object'):
            cls = obj.find('name').text
            if cls not in classes:
                classes.append(cls)
    print("Classes: ", classes)
    return classes
                

def get_boxes(name, annotation_files, more_data = False):

    annotation_file = name + ".xml"
    boxes = []
    if annotation_file in annotation_files:
        # read annotation file
        if more_data:
            annotation_path = os.path.join(MORE_ANNOTATION_DIR, annotation_file)
        else:
            annotation_path = os.path.join(ANNOTATION_DIR, annotation_file)
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # get class name and bounding box
        for obj in root.iter('object'):
            # bounding box
            xmlbox = obj.find('bndbox')
            xmin = int(xmlbox.find('xmin').text)
            ymin = int(xmlbox.find('ymin').text)
            xmax = int(xmlbox.find('xmax').text)
            ymax = int(xmlbox.find('ymax').text)
            cls = obj.find('name').text
            box = {
                "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                "cx" : 0, "cy": 0, "w": 0, "h": 0,
                "width": width, "height": height,
                "class": cls, "confidence": 1
            }
            box = bbox_to_center(box)
            # convert bounding box
            boxes.append(box)
    else:
        # no annotation file
        return None
    
    return boxes

def image_resize2sq(image, boxes, input_side = INPUT_SIDE):
    # pad image to square
    # resize image to input_side
    # resize bounding box
    
    # pad image to square
    is_padx = False
    is_pady = False
    if image.shape[0] > image.shape[1]:
        pad = (image.shape[0] - image.shape[1]) // 2
        is_odd = (image.shape[0] - image.shape[1]) % 2
        image_pad = cv2.copyMakeBorder(image, 0, 0, pad, pad + is_odd, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        is_padx = True
    else:
        pad = (image.shape[1] - image.shape[0]) // 2
        is_odd = (image.shape[1] - image.shape[0]) % 2
        image_pad = cv2.copyMakeBorder(image, pad, pad + is_odd, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0)) 
        is_pady = True
    # resize image
    assert image_pad.shape[0] == image_pad.shape[1]
    original_side = image_pad.shape[0]
    resized_image = cv2.resize(image_pad, (input_side, input_side))
    # resize bounding box
    resized_boxes = []
    for j in range(len(boxes)):
        box = boxes[j].copy()
        # resize xmin, ymin, xmax, ymax
        xmin = box["xmin"] + pad if is_padx else box["xmin"]
        ymin = box["ymin"] + pad if is_pady else box["ymin"]
        xmax = box["xmax"] + pad if is_padx else box["xmax"]
        ymax = box["ymax"] + pad if is_pady else box["ymax"]
        xmin = xmin * input_side // original_side
        ymin = ymin * input_side // original_side
        xmax = xmax * input_side // original_side
        ymax = ymax * input_side // original_side
        # update
        box["xmin"] = xmin
        box["ymin"] = ymin
        box["xmax"] = xmax
        box["ymax"] = ymax
        box["width"] = input_side
        box["height"] = input_side
        # calculate cx, cy, w, h
        box = bbox_to_center(box)
        # update
        resized_boxes.append(box)
    
    return resized_image, resized_boxes


