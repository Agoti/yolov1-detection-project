# data_augmentation.py 
# Description: data augmentation functions

from yolo_utils import *
import cv2
import numpy as np
import copy
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


def random_flip(image, boxes):
    """
    Randomly horizantally flip the image and adjust the bounding boxes accordingly
    """
    # a 50% chance to flip the image
    if np.random.randint(2):
        # flip the image
        image = cv2.flip(image, 1)
        # adjust the bounding boxes accordingly
        for box in boxes:
            xmin = box["xmin"]
            xmax = box["xmax"]
            box["xmin"] = image.shape[1] - xmax
            box["xmax"] = image.shape[1] - xmin
    return image, boxes

def random_hue(image, boxes, delta=18.0):
    """
    Randomly change the hue of the image
    Input:
        image: HSV float32 image
        boxes: list of dictionaries containing bounding box coordinates and class
        delta: the hue change range
    """
    if np.random.randint(2):
        # hue += random(-delta, delta)
        random_hue = np.random.uniform(-delta, delta)
        image[:, :, 0] = (image[:, :, 0] + random_hue)
        image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
        image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
    return image, boxes

def random_saturation(image, boxes, lower=0.8, upper=1.2):
    """
    Randomly change the saturation of the image
    Input:
        image: HSV float32 image
        boxes: list of dictionaries containing bounding box coordinates and class
        lower: the lower bound of the saturation change range
        upper: the upper bound of the saturation change range
    """
    if np.random.randint(2):
        random_saturation = np.random.uniform(lower, upper)
        image[:, :, 1] *= random_saturation
        image[:, :, 1][image[:, :, 1] > 1.0] = 1.0
    return image, boxes

def random_value(image, boxes, lower=0.5, upper=1.5):
    """
    Randomly change the value of the image
    Input:
        image: HSV float32 image
        boxes: list of dictionaries containing bounding box coordinates and class
        lower: the lower bound of the value change range
        upper: the upper bound of the value change range
    """
    if np.random.randint(2):
        random_value = np.random.uniform(lower, upper)
        image[:, :, 2] *= random_value
        image[:, :, 2][image[:, :, 2] > 255.0] = 255.0
    return image, boxes

def random_brightness(image, boxes, delta=32):
    """
    Randomly change the brightness of the image
    Input:
        image: RGB float32 image
        boxes: list of dictionaries containing bounding box coordinates and class
        delta: the brightness change range
    """
    if np.random.randint(2):
        random_brightness = np.random.uniform(-delta, delta)
        image += random_brightness
        image[image > 255.0] = 255.0
        image[image < 0.0] = 0.0
    return image, boxes

def random_expand(image, boxes):
    """
    Randomly expand the image and adjust the bounding boxes accordingly
    Input:
        image: RGB float32 image
        boxes: list of dictionaries containing bounding box coordinates and class
    """
    if np.random.randint(2):
        height, width, depth = image.shape
        # random expansion ratio
        ratio = np.random.uniform(1, 1.2)
        # random left and top
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)
        # expand the image and fill in the new pixels with zeros
        expand_image = np.zeros((int(height * ratio), int(width * ratio), depth), dtype=image.dtype)
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image
        for box in boxes:
            box["xmin"] += int(left)
            box["xmax"] += int(left)
            box["ymin"] += int(top)
            box["ymax"] += int(top)
            box["width"] = int(width * ratio)
            box["height"] = int(height * ratio)
    return image, boxes

def random_sample_crop(image, boxes):
    """
    Randomly crop the image and adjust the bounding boxes accordingly
    Input:
        image: RGB float32 image
        boxes: list of dictionaries containing bounding box coordinates and class
    """
    
    # the crop mode, (min_iou, max_iou)
    modes = (None, None, 
             (0.7, 1.1), (0.9, 1.1), 
             (-0.1, 1.1))
    # modes = (None, None, None,  (-0.1, 1.1),
    #         (0.3, 1.1), (0.5, 1.1),
    #          (0.7, 1.1), (0.9, 1.1))
    # modes = (None, (-0.1, 1.1))
    if len(boxes) == 0:
        return image, boxes
    
    width, height, _ = image.shape

    while True:
        # randomly choose a mode
        mode = modes[np.random.randint(0, len(modes))]
        # if mode is None, don't crop
        if mode is None:
            return image, boxes
        min_iou, max_iou = mode

        # try 50 times
        for _ in range(50):
            current_image = image.copy()
            # randomly choose a width and height from 0.3 to 1.0
            width_crop = np.random.uniform(0.3 * width, width)
            height_crop = np.random.uniform(0.3 * height, height)
            # if the aspect ratio is too large or too small, continue
            if height_crop / width_crop < 0.5 or height_crop / width_crop > 2:
                continue
            # randomly choose a left and top from 0 to width - width_crop and 0 to height - height_crop
            left = np.random.uniform(width - width_crop)
            top = np.random.uniform(height - height_crop)
            width_crop, height_crop, left, top = int(width_crop), int(height_crop), int(left), int(top)
            # the crop box
            box_crop = {"xmin": int(left), "ymin": int(top), "xmax": int(left + width_crop), "ymax": int(top + height_crop)}
            iou_ = [overlap(box, box_crop) for box in boxes]
            # all boxes must have an iou(actually OVERLAP ratio) between min_iou and max_iou
            if min_iou > min(iou_) or max_iou < max(iou_):
                continue
            # calculate the new bounding boxes and crop the image
            current_image = current_image[int(top):int(top + height_crop), int(left):int(left + width_crop), :]
            new_boxes = []
            for box in boxes:
                center_x = 0.5 * (box["xmin"] + box["xmax"])
                center_y = 0.5 * (box["ymin"] + box["ymax"])
                if center_x >= left and center_x <= left + width_crop and center_y >= top and center_y <= top + height_crop:
                    new_boxes.append(box)
            if len(new_boxes) == 0:
                continue
            for box in new_boxes:
                box["xmin"] -= int(left)
                box["xmax"] -= int(left)
                box["ymin"] -= int(top)
                box["ymax"] -= int(top)
                box["xmin"] = max(0, box["xmin"])
                box["xmax"] = min(width_crop - 1, box["xmax"])
                box["ymin"] = max(0, box["ymin"])
                box["ymax"] = min(height_crop - 1, box["ymax"])
                box["width"] = width_crop
                box["height"] = height_crop
            return current_image, new_boxes
        
def random_sample_crop2(image, boxes):
    """
    Randomly crop the image and adjust the bounding boxes accordingly
    Input:
        image: RGB float32 image
        boxes: list of dictionaries containing bounding box coordinates and class
    """

    # the crop mode, (min_iou, max_iou)
    modes = (None, None, 
             (0.7, 1.1), (0.9, 1.1))
    # modes = (None, None, None,  (-0.1, 1.1),
    #         (0.3, 1.1), (0.5, 1.1),
    #          (0.7, 1.1), (0.9, 1.1))
    # modes = (None, (-0.1, 1.1))
    if len(boxes) == 0:
        return image, boxes
    
    width, height, _ = image.shape

    while True:
        # randomly choose a mode
        mode = modes[np.random.randint(0, len(modes))]
        # if mode is None, don't crop
        if mode is None:
            return image, boxes
        min_iou, max_iou = mode

        # try 50 times
        for _ in range(50):
            current_image = image.copy()
            # randomly choose a width and height from 0.3 to 1.0
            width_crop = np.random.uniform(0.3 * width, width)
            height_crop = np.random.uniform(0.3 * height, height)
            # if the aspect ratio is too large or too small, continue
            if height_crop / width_crop < 0.5 or height_crop / width_crop > 2:
                continue
            # randomly choose a left and top from 0 to width - width_crop and 0 to height - height_crop
            left = np.random.uniform(width - width_crop)
            top = np.random.uniform(height - height_crop)
            width_crop, height_crop, left, top = int(width_crop), int(height_crop), int(left), int(top)
            # the crop box
            box_crop = {"xmin": int(left), "ymin": int(top), "xmax": int(left + width_crop), "ymax": int(top + height_crop)}
            iou_ = [overlap(box, box_crop) for box in boxes]
            # remove zeros
            iou_ = [iou for iou in iou_ if iou > 0]
            if len(iou_) == 0:
                continue
            # all boxes must have an iou(actually OVERLAP ratio) between min_iou and max_iou
            if min_iou > min(iou_) or max_iou < max(iou_):
                continue
            # calculate the new bounding boxes and crop the image
            current_image = current_image[int(top):int(top + height_crop), int(left):int(left + width_crop), :]
            new_boxes = []
            for box in boxes:
                center_x = 0.5 * (box["xmin"] + box["xmax"])
                center_y = 0.5 * (box["ymin"] + box["ymax"])
                if center_x >= left and center_x <= left + width_crop and center_y >= top and center_y <= top + height_crop:
                    new_boxes.append(box)
            if len(new_boxes) == 0:
                continue
            for box in new_boxes:
                box["xmin"] -= int(left)
                box["xmax"] -= int(left)
                box["ymin"] -= int(top)
                box["ymax"] -= int(top)
                box["xmin"] = max(0, box["xmin"])
                box["xmax"] = min(width_crop - 1, box["xmax"])
                box["ymin"] = max(0, box["ymin"])
                box["ymax"] = min(height_crop - 1, box["ymax"])
                box["width"] = width_crop
                box["height"] = height_crop
            return current_image, new_boxes

def photometric_distort(image, boxes):
    """
    Apply photometric distortions to the image
    (brightness, (value), saturation, hue, (value))
    Input:
        image: RGB float32 image
        boxes: list of dictionaries containing bounding box coordinates and class
    """
    image, boxes = random_brightness(image, boxes)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    before = np.random.randint(2)
    if before:
        image, boxes = random_value(image, boxes)
    image, boxes = random_saturation(image, boxes)
    image, boxes = random_hue(image, boxes)
    if not before:
        image, boxes = random_value(image, boxes)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2RGB)
    return image, boxes

def augmentation(image, boxes):
    """
    Apply augmentation to the image
    (photometric distortions, expand, crop, flip)
    Input:
        image: RGB float32 image
        boxes: list of dictionaries containing bounding box coordinates and class
    """
    image = image.astype(np.float32)
    image, boxes = photometric_distort(image, boxes)
    image, boxes = random_expand(image, boxes)
    image, boxes = random_sample_crop(image, boxes)
    image, boxes = random_flip(image, boxes)
    image = image.astype(np.uint8)
    for box in boxes:
        box = bbox_to_center(box)
    return image, boxes

def overlap(box, crop):
    """
    Calculate the overlap ratio of box in crop
    |box intersect crop| / |box|
    """
    intersect_xmin = max(box["xmin"], crop["xmin"])
    intersect_ymin = max(box["ymin"], crop["ymin"])
    intersect_xmax = min(box["xmax"], crop["xmax"])
    intersect_ymax = min(box["ymax"], crop["ymax"])
    inter_area = max(0, intersect_xmax - intersect_xmin + 1) * max(0, intersect_ymax - intersect_ymin + 1)
    box_area = (box["xmax"] - box["xmin"] + 1) * (box["ymax"] - box["ymin"] + 1)
    return inter_area / box_area

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from dataloader import DataLoader
    np.random.seed(42)
    dataloader = DataLoader()
    dataloader.load_data(max_file=100)
    # for i in range(100):
    #     image = dataloader.train_images[i].copy()
    #     boxes = dataloader.train_labels[i]["boxes"]
    while True:
        # i = np.random.randint(len(dataloader.train_images))
        i = 0
        image = dataloader.train_images[i].copy()
        boxes = dataloader.train_labels[i]["boxes"]
        boxes = copy.deepcopy(boxes)
        image, boxes = augmentation(image, boxes)
        plt.imshow(image)
        for box in boxes:
            xmin = box["xmin"]
            ymin = box["ymin"]
            xmax = box["xmax"]
            ymax = box["ymax"]
            # plot bounding box, a rectangle
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        plt.imshow(image)
        plt.show()

