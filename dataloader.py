# dataloader.py: Data loader for the dataset
# dataset: PASCAL VOC 2012
# training set: 2007_000559 to 2012_001051
# test set: else

import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import h5py
import copy
from constants import *
from yolo_utils import *
from dataloader_utils import *
from data_augmentation import *


class DataLoader(object):
    def __init__(self):
        # the list of images and labels
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.dev_images = []
        self.dev_labels = []
        self.batch_size = BATCH_SIZE

    def load_data(self, max_file = 1000000, more_data = False):
        """
        Load data from VOC dataset
        max_file: maximum number of files to load, set to a small number for debugging
        more_data: whether to load the whole VOC2007 dataset
        """

        print("Loading data...")
        # list all image files and annotation files
        image_files = os.listdir(IMAGE_DIR)
        annotation_files = os.listdir(ANNOTATION_DIR)
        # count for max_file
        count = 0
        for image_file in image_files:
            # max_file
            count += 1
            if count > max_file:
                break
            # get image
            image_name = image_file.split(".")[0]
            image = cv2.imread(os.path.join(IMAGE_DIR, image_file))
            # get class name and bounding box
            boxes = get_boxes(image_name, annotation_files)
            if boxes is None:
                continue
            # split into train and test set
            if image_name < TRAIN_TEST_SPLIT:
                resized_image, resized_boxes = image_resize2sq(image, boxes, INPUT_SIDE)
                self.train_images.append(image)
                self.train_labels.append({"image_name": image_name, "boxes": boxes,\
                                        "resized_image": resized_image, "resized_boxes": resized_boxes})
            else:
                resized_image, resized_boxes = image_resize2sq(image, boxes, INPUT_SIDE)
                self.test_images.append(resized_image)
                self.test_labels.append({"image_name": image_name, "boxes": resized_boxes,\
                                        "original_image": image, "original_boxes": boxes})
                        
        if more_data:
            print("Loading more data...")
            image_files = os.listdir(MORE_IMAGE_DIR)
            annotation_files = os.listdir(MORE_ANNOTATION_DIR)
            for image_file in image_files:
                count += 1
                if count > max_file:
                    break
                image_name = image_file.split(".")[0]
                image = cv2.imread(os.path.join(MORE_IMAGE_DIR, image_file))
                # get class name and bounding box
                boxes = get_boxes(image_name, annotation_files, more_data = True)
                if boxes is None:
                    continue
                # resize image
                resized_image, resized_boxes = image_resize2sq(image, boxes, INPUT_SIDE)
                self.train_images.append(image)
                self.train_labels.append({"image_name": image_name, "boxes": boxes,\
                                            "resized_image": resized_image, "resized_boxes": resized_boxes})

        print("Loading data done.")
        print("Train set size: ", len(self.train_images))
        print("Test set size: ", len(self.test_images))
    
    # DID NOT USE
    def split_train_dev(self, ratio = 0.1):
        # merge train and dev set
        self.train_images += self.dev_images
        self.train_labels += self.dev_labels
        # split train set into train and dev set
        n_dev = int(len(self.train_images) * ratio)
        indices = np.arange(len(self.train_images))
        np.random.shuffle(indices)
        dev_indices = indices[:n_dev]
        train_indices = indices[n_dev:]
        self.dev_images = [self.train_images[i] for i in dev_indices]
        self.dev_labels = [self.train_labels[i] for i in dev_indices]
        self.train_images = [self.train_images[i] for i in train_indices]
        self.train_labels = [self.train_labels[i] for i in train_indices]
        print("Split train set into train and dev set.")
        print("Train set size: ", len(self.train_images))
        print("Dev set size: ", len(self.dev_images))
    
    # DID NOT USE
    def save_data(self):
        # save bounding box as txt file
        # save label as npy file
        print("Saving data...")
        for i in range(len(self.train_images)):
            image = self.train_labels[i]["original_image"]
            image_name = self.train_labels[i]["image_name"]
            boxes = self.train_labels[i]["original_boxes"]
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
            image_with_box = plot_bbox_on_image(image, boxes)
            plot_path = os.path.join(TRAIN_PLOT_DIR, image_name + ".jpg")
            cv2.imwrite(plot_path, image_with_box)
        
        print("Saving train data done.")
        for i in range(len(self.test_images)):
            image = self.test_labels[i]["original_image"]
            image_name = self.test_labels[i]["image_name"]
            boxes = self.test_labels[i]["original_boxes"]
            label = self.test_labels[i]
            # save bounding box
            box_path = os.path.join(TEST_BOX_DIR, image_name + ".txt")
            # create txt file and write bounding box, do not use np.savetxt, it will raise error
            with open(box_path, "w") as f:
                for box in boxes:
                    f.write(str(box["xmin"]) + " " + str(box["ymin"]) + " " + str(box["xmax"]) + " " + str(box["ymax"]) + " " + box["class"] + "\n")
            # save label as h5 file
            label_path = os.path.join(TEST_LABEL_DIR, image_name + ".h5")
            with h5py.File(label_path, "w") as f:
                f.create_dataset("label", data=label)
            # plot bounding box on image
            image_with_box = plot_bbox_on_image(image, boxes)
            plot_path = os.path.join(TEST_PLOT_DIR, image_name + ".jpg")
            cv2.imwrite(plot_path, image_with_box)
        
        print("Saving test data done.")
    
    def pull_item(self, index, is_aug = False):
        """
        Pull an item at index from the dataset
        index: index of the item
        is_aug: whether to apply data augmentation
        return: 
            image(np.ndarray), label(np.array)
        """
        image = self.train_images[index].copy()
        boxes = copy.deepcopy(self.train_labels[index]["boxes"])
        if not is_aug:
            # no augmentation
            image, boxes = image_resize2sq(image, boxes, INPUT_SIDE)
            return image, box2label(boxes)
        # apply data augmentation
        image, boxes = augmentation(image, boxes)
        image, boxes = image_resize2sq(image, boxes, INPUT_SIDE)
        # convert to label
        label = box2label(boxes)
        
        return image, label
    
    def get_train_batch(self, is_aug = False):
        """
        Get a batch of training data
        is_aug: whether to apply data augmentation
        return:
            batch_images(list of np.ndarray)
            batch_labels(list of np.array)
        """

        # shuffle the indices
        indices = np.arange(len(self.train_images))
        np.random.shuffle(indices)
        # get a batch of data, pull items one by one
        for i in range(0, len(indices), self.batch_size):
            batch_images = []
            batch_labels = []
            batch_indices = indices[i: i + self.batch_size]
            for j in batch_indices:
                image, label = self.pull_item(j, is_aug = is_aug)
                batch_images.append(image)
                batch_labels.append(label)

            yield batch_images, batch_labels
    
            

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
    
