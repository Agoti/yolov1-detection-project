import os

# DataLoader
IMAGE_DIR = "JPEGImages"
ANNOTATION_DIR = "Annotations"
BATCH_SIZE = 32
TRAIN_TEST_SPLIT = "2012_001051"
TRAIN_BOX_DIR = os.path.join("data", "train", "box")
TRAIN_LABEL_DIR = os.path.join("data", "train", "label")
TRAIN_PLOT_DIR = os.path.join("data", "train", "plot")
TEST_BOX_DIR = os.path.join("data", "test", "box")
TEST_LABEL_DIR = os.path.join("data", "test", "label")
TEST_PLOT_DIR = os.path.join("data", "test", "plot")

# YOLO model
N_GRID_SIDE = 7
INPUT_SIDE = 448
N_CLASSES = 20

# Random seed
SEED = 1219183536 # This is my qq number :)

