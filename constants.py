import os

# DataLoader
IMAGE_DIR = "JPEGImages"
ANNOTATION_DIR = "Annotations"
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = "2012_001051"
TRAIN_BOX_DIR = os.path.join("data", "train", "box")
TRAIN_LABEL_DIR = os.path.join("data", "train", "label")
TRAIN_PLOT_DIR = os.path.join("result", "train_plot")
TEST_BOX_DIR = os.path.join("data", "test", "box")
TEST_LABEL_DIR = os.path.join("data", "test", "label")
TEST_PLOT_DIR = os.path.join("result", "test_plot")
CLASSES = ['bottle', 'tvmonitor', 'train', 'person', 'sofa', 'pottedplant',\
            'chair', 'motorbike', 'boat', 'dog', 'bird', 'bicycle', 'diningtable',\
            'cat', 'horse', 'bus', 'car', 'sheep', 'aeroplane', 'cow']
N_CLASSES = len(CLASSES)

# YOLO model
N_GRID_SIDE = 7
INPUT_SIDE = 448
N_BBOX = 2

# Random seed
SEED = 1219183536 # This is my qq number :)

