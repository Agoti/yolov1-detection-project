import cv2
import numpy as np
from yolo_utils import *

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def random_flip(image, boxes):
    """
    Randomly flip the image and adjust the bounding boxes accordingly
    """
    if np.random.randint(2):
        image = cv2.flip(image, 1)
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
        random_hue = np.random.uniform(-delta, delta)
        image[:, :, 0] = (image[:, :, 0] + random_hue) % 180
        image[:, :, 0][image[:, :, 0] > 360.0] -= 360.0
        image[:, :, 0][image[:, :, 0] < 0.0] += 360.0
    return image, boxes

def random_saturation(image, boxes, lower=0.5, upper=1.5):
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
    if 1:
        height, width, depth = image.shape
        ratio = np.random.uniform(1, 1.2)
        left = np.random.uniform(0, width * ratio - width)
        top = np.random.uniform(0, height * ratio - height)
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
    modes = (None, (0.3, 1.1), (0.7, 1.1), (0.9, 1.1), (-0.1, 1.1))
    # modes = (None, (-0.1, 1.1))
    if len(boxes) == 0:
        return image, boxes
    
    width, height, _ = image.shape

    while True:
        mode = modes[np.random.randint(0, len(modes))]
        if mode is None:
            return image, boxes
        min_iou, max_iou = mode

        for _ in range(50):
            current_image = image.copy()
            width_crop = np.random.uniform(0.3 * width, width)
            height_crop = np.random.uniform(0.3 * height, height)
            if height_crop / width_crop < 0.5 or height_crop / width_crop > 2:
                continue
            left = np.random.uniform(width - width_crop)
            top = np.random.uniform(height - height_crop)
            width_crop, height_crop, left, top = int(width_crop), int(height_crop), int(left), int(top)
            box_crop = {"xmin": int(left), "ymin": int(top), "xmax": int(left + width_crop), "ymax": int(top + height_crop)}
            iou_ = [overlap(box_crop, box) for box in boxes]
            if min_iou > max(iou_) or max_iou < min(iou_):
                continue
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
    return image, boxes

def overlap(box, crop):
    intersect_xmin = max(box["xmin"], crop["xmin"])
    intersect_ymin = max(box["ymin"], crop["ymin"])
    intersect_xmax = min(box["xmax"], crop["xmax"])
    intersect_ymax = min(box["ymax"], crop["ymax"])
    inter_area = max(0, intersect_xmax - intersect_xmin + 1) * max(0, intersect_ymax - intersect_ymin + 1)
    box_area = (box["xmax"] - box["xmin"] + 1) * (box["ymax"] - box["ymin"] + 1)
    return inter_area / box_area

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    np.random.seed(40)
    image = cv2.imread("test.jpg")
    boxes = [
                {"xmin": 10, "ymin": 10, "xmax": 100, "ymax": 100,\
               "class": "person", "confidence": 1, "width": image.shape[1], "height": image.shape[0]},\
                {"xmin": 200, "ymin": 200, "xmax": 300, "ymax": 300,\
                 "class": "person", "confidence": 1, "width": image.shape[1], "height": image.shape[0]}
            ]
    image_plot = plot_bbox_on_image(image, boxes)
    # plot the original image
    plt.figure()
    plt.imshow(image_plot)
    plt.show()
    original_image = image.copy()
    original_boxes = copy.deepcopy(boxes)
    while True:
        try:
            image = original_image.copy()
            boxes = copy.deepcopy(original_boxes)
            image, boxes = augmentation(image, boxes)
            image = plot_bbox_on_image(image, boxes)
            # plot the augmented image
            plt.figure()
            plt.imshow(image)
            plt.show()
        except Exception as e:
            if str(e) == "KeyboardInterrupt":
                break
            else:
                print(e)
                print(boxes)
                continue
