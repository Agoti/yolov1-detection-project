import os

# DataLoader
# directory paths
IMAGE_DIR = "JPEGImages"
ANNOTATION_DIR = "Annotations"
# the whole VOC2007 dataset
MORE_IMAGE_DIR = "VOC2007/JPEGImages"
MORE_ANNOTATION_DIR = "VOC2007/Annotations"
# batch size: it shouldn't have been a constant
BATCH_SIZE = 32
# the border between train and test set
TRAIN_TEST_SPLIT = "2012_001051"
# the pathes to save the visualization
TRAIN_BOX_DIR = os.path.join("data", "train", "box")
TRAIN_LABEL_DIR = os.path.join("data", "train", "label")
TRAIN_PLOT_DIR = os.path.join("result", "train_plot")
TEST_BOX_DIR = os.path.join("data", "test", "box")
TEST_LABEL_DIR = os.path.join("data", "test", "label")
TEST_PLOT_DIR = os.path.join("result", "test_plot")
# pascal voc classes
CLASSES = ['bottle', 'tvmonitor', 'train', 'person', 'sofa', 'pottedplant',\
            'chair', 'motorbike', 'boat', 'dog', 'bird', 'bicycle', 'diningtable',\
            'cat', 'horse', 'bus', 'car', 'sheep', 'aeroplane', 'cow']
N_CLASSES = len(CLASSES)

# YOLO model
# the number of grid cells on each side
N_GRID_SIDE = 7
# the number of input width and height
INPUT_SIDE = 448
# the number of box detector per grid cell: YOLOv1 uses 2
N_BBOX = 2

# Random seed
SEED = 1219183536 # This is my qq number :)

