# dataloader.py: Data loader for the dataset

# dataset: PASCAL VOC 2012
# training set: 2007_000559 to 2012_001051
# test set: else

# 2 directory: JPEGImages and Annotations
# Annotations: xml file, format:
# <annotation>
# 	<folder>VOC2012</folder>
# 	<filename>2007_000559.jpg</filename>
# 	<source>
# 		<database>The VOC2007 Database</database>
# 		<annotation>PASCAL VOC2007</annotation>
# 		<image>flickr</image>
# 	</source>
# 	<size>
# 		<width>500</width>
# 		<height>370</height>
# 		<depth>3</depth>
# 	</size>
# 	<segmented>1</segmented>
# 	<object>
# 		<name>bottle</name>
# 		<pose>Unspecified</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<bndbox>
# 			<xmin>36</xmin>
# 			<ymin>250</ymin>
# 			<xmax>79</xmax>
# 			<ymax>354</ymax>
# 		</bndbox>
# 	</object>
# 	<object>
# 		<name>tvmonitor</name>
# 		<pose>Frontal</pose>
# 		<truncated>0</truncated>
# 		<difficult>0</difficult>
# 		<bndbox>
# 			<xmin>160</xmin>
# 			<ymin>26</ymin>
# 			<xmax>371</xmax>
# 			<ymax>241</ymax>
# 		</bndbox>
# 	</object>
# </annotation>

# I only need class name and bounding box for each object

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import h5py
from constants import *

class DataLoader(object):
    def __init__(self):
        self.classes = get_classes()
        self.train_images = []
        self.test_images = []
        self.train_labels = []
        self.test_labels = []
        # used for saving and visualization
        self.train_msg = []
        self.test_msg = []
        self.batch_size = BATCH_SIZE
        np.random.seed(SEED)

    def load_data(self):
        print("Loading data...")
        image_files = os.listdir(IMAGE_DIR)
        annotation_files = os.listdir(ANNOTATION_DIR)
        for image_file in image_files:
            image_name = image_file.split(".")[0]
            image = cv2.imread(os.path.join(IMAGE_DIR, image_file))
            # get class name and bounding box
            boxes = get_boxes(image_name, annotation_files)
            if boxes is None:
                continue
            # resize image
            resized_image, resized_boxes = image_resize(image, boxes, INPUT_SIDE)
            label = box2label(resized_boxes, self.classes)
            # split into train and test set
            if image_name < TRAIN_TEST_SPLIT:
                self.train_images.append(resized_image)
                self.train_labels.append(label)
                self.train_msg.append({"image_name": image_name, "boxes": resized_boxes,\
                                        "original_image": image, "original_boxes": boxes})
            else:
                self.test_images.append(resized_image)
                self.test_labels.append(label)
                self.test_msg.append({"image_name": image_name, "boxes": resized_boxes,\
                                        "original_image": image, "original_boxes": boxes})
                
        print("Loading data done.")
        print("Train set size: ", len(self.train_images))
        print("Test set size: ", len(self.test_images))
    
    def save_data(self):
        # save bounding box as txt file
        # save label as npy file
        # path: data/train/box/*.txt
        # path: data/train/label/*.npy
        # path: data/test/...
        # plot bounding box on image and save
        print("Saving data...")
        for i in range(len(self.train_images)):
            image = self.train_msg[i]["original_image"]
            image_name = self.train_msg[i]["image_name"]
            boxes = self.train_msg[i]["original_boxes"] 
            label = self.train_labels[i]
            # save bounding box
            box_path = os.path.join(TRAIN_BOX_DIR, image_name + ".txt")
            # create txt file and write bounding box, do not use np.savetxt, it will raise error
            with open(box_path, "w") as f:
                for box in boxes:
                    f.write(str(box["xmin"]) + " " + str(box["ymin"]) + " " + str(box["xmax"]) + " " + str(box["ymax"]) + " " + box["class"] + "\n")
            # save label as h5 file
            label_path = os.path.join(TRAIN_LABEL_DIR, image_name + ".h5")
            with h5py.File(label_path, "w") as f:
                f.create_dataset("label", data=label)
            # plot bounding box on image
            for box in boxes:
                xmin = box["xmin"]
                ymin = box["ymin"]
                xmax = box["xmax"]
                ymax = box["ymax"]
                # mark bounding box and class name
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, box["class"], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # save image
                plot_path = os.path.join(TRAIN_PLOT_DIR, image_name + ".jpg")
                cv2.imwrite(plot_path, image)
        
        print("Saving train data done.")
        for i in range(len(self.test_images)):
            image = self.test_msg[i]["original_image"]
            image_name = self.test_msg[i]["image_name"]
            boxes = self.test_msg[i]["original_boxes"]
            label = self.test_labels[i]
            # save bounding box
            box_path = os.path.join(TEST_BOX_DIR, image_name + ".txt")
            with open(box_path, "w") as f:
                for box in boxes:
                    f.write(str(box["xmin"]) + " " + str(box["ymin"]) + " " + str(box["xmax"]) + " " + str(box["ymax"]) + " " + box["class"] + "\n")
            # save label as hdf5 file
            label_path = os.path.join(TEST_LABEL_DIR, image_name + ".h5")
            with h5py.File(label_path, "w") as f:
                f.create_dataset("label", data=label)
            # plot bounding box on image
            for box in boxes:
                xmin = box["xmin"]
                ymin = box["ymin"]
                xmax = box["xmax"]
                ymax = box["ymax"]
                # mark bounding box and class name
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(image, box["class"], (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # save image
                plot_path = os.path.join(TEST_PLOT_DIR, image_name + ".jpg")
                cv2.imwrite(plot_path, image)
        
        print("Saving test data done.")
    
    # def get_train_batches(self):
    #     # return: list of batch images, list of batch labels
    #     # image: (batch_size, 3, 448, 448)
    #     # label: (batch_size, N_GRID_SIDE, N_GRID_SIDE, 5 + N_CLASSES)
        
    #     # shuffle
    #     indices = np.arange(len(self.train_images))
    #     np.random.shuffle(indices)
    #     shuffled_images = []
    #     shuffled_labels = []
    #     for i in indices:
    #         shuffled_images.append(self.train_images[i])
    #         shuffled_labels.append(self.train_labels[i])
    #     # split into batches
    #     batches_images = []
    #     batches_labels = []
    #     for i in range(0, len(self.train_images), self.batch_size):
    #         batch_images = shuffled_images[i:i + self.batch_size]
    #         batch_labels = shuffled_labels[i:i + self.batch_size]
    #         # convert label to np array
    #         batch_labels = np.array(batch_labels)
    #         # update
    #         batches_images.append(batch_images)
    #         batches_labels.append(batch_labels)
    #     return batches
        
## Utility functions

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
                

def bbox_to_center(box):

    # box: box dictionary
    # return: augmented box dictionary
    xmin = box["xmin"]
    ymin = box["ymin"]
    xmax = box["xmax"]
    ymax = box["ymax"]
    width = box["width"]
    height = box["height"]
    # convert
    cx = (xmin + xmax) / 2.0 / width
    cy = (ymin + ymax) / 2.0 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    # update
    box["cx"] = cx
    box["cy"] = cy
    box["w"] = w
    box["h"] = h
    return box

def get_boxes(name, annotation_files):

    annotation_file = name + ".xml"
    boxes = []
    if annotation_file in annotation_files:
        # read annotation file
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

def image_resize(image, boxes, input_side):
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

# convert bounding boxes to labels for 
# yolo training
def box2label(boxes, classes):
    
    grid_side = 1.0 / N_GRID_SIDE
    # currently without anchor boxes
    labels = np.zeros((N_GRID_SIDE, N_GRID_SIDE, 5 + len(classes)), dtype=np.float32)
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
        label = np.zeros(5 + len(classes), dtype=np.float32)
        label[0] = offset_x
        label[1] = offset_y
        label[2] = w
        label[3] = h
        label[4] = 1.0
        label[5 + classes.index(cls)] = 1.0
        # update labels
        labels[grid_x, grid_y, :] = label[:]

    
    return labels

if __name__ == "__main__":
    dataset = DataLoader()

    dataset.load_data()
    # dataset.save_data()
    image = dataset.train_images[0]
    label = dataset.train_labels[0]
    boxes = dataset.train_msg[0]["boxes"]

    # show one of the resized images and its bounding box
    import matplotlib.pyplot as plt
    plt.imshow(dataset.train_images[0])
    for box in boxes:
        xmin = box["xmin"]
        ymin = box["ymin"]
        xmax = box["xmax"]
        ymax = box["ymax"]
        # plot bounding box, a rectangle
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="green", linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(xmin, ymin, box["class"], color="green", fontsize=12)
    plt.show()
    # find non-zero confidence in label
    for i in range(N_GRID_SIDE):
        for j in range(N_GRID_SIDE):
            if label[i, j, 4] > 0:
                print(i, j)
                print(label[i, j, 4])
    
