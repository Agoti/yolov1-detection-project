# dataloader.py: Data loader for the dataset

# dataset: PASCAL VOC 2012
# training set: 2007_000559 to 2012_001051
# test set: else

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import h5py
from constants import *
from yolo_utils import *
from dataloader_utils import *

class DataLoader(object):
    def __init__(self):
        self.train_images = []
        self.test_images = []
        self.train_labels = []
        self.test_labels = []
        # used for saving and visualization
        self.train_msg = []
        self.test_msg = []
        self.batch_size = BATCH_SIZE
        np.random.seed(SEED)

    def load_data(self, max_file = 1000000):
        print("Loading data...")
        image_files = os.listdir(IMAGE_DIR)
        annotation_files = os.listdir(ANNOTATION_DIR)
        count = 0
        for image_file in image_files:
            count += 1
            if count > max_file:
                break
            image_name = image_file.split(".")[0]
            image = cv2.imread(os.path.join(IMAGE_DIR, image_file))
            # get class name and bounding box
            boxes = get_boxes(image_name, annotation_files)
            if boxes is None:
                continue
            # resize image
            resized_image, resized_boxes = image_resize(image, boxes, INPUT_SIDE)
            label = box2label(resized_boxes)
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
    
