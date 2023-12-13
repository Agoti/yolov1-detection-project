import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from constants import *
from dataloader import *

# YOLO Net
# To save time, use pre-trained ResNet34
# remove the last two layers, and connect 
# YOLONet layers(last 4 conv and 2 fc)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        resnet = models.resnet34(pretrained=True)
        # remove the last two layers
        resnet_out_channels = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])
        self.yolo_tail = nn.Sequential(
            nn.Conv2d(resnet_out_channels, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.fcs = nn.Sequential(
            nn.Linear(N_GRID_SIDE * N_GRID_SIDE * 1024, 4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, N_GRID_SIDE * N_GRID_SIDE * (5 + N_CLASSES)),
            # add sigmoid to the last layer
            # nn.Sigmoid()
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.yolo_tail(x)
        x = x.view(x.size(0), -1)
        x = self.fcs(x)
        return x

    def fit(self, train_images, train_labels, n_epoch=10, learning_rate=0.001):

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(n_epoch):
            for batch_images, batch_labels in get_train_batches(train_images, train_labels):
                batch_images = torch.Tensor(batch_images)
                batch_labels = torch.Tensor(batch_labels, dtype=torch.float32)
                pred = self.forward(batch_images)
                loss = yolo_loss(pred, batch_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
    
    def predict(self, test_images):
        test_images = torch.Tensor(test_images)
        pred = self.forward(test_images)
        return pred

def main():
    net = Net()
    dataset = DataSet()
    dataset.load_data()
    train_images, train_labels = dataset.train_images, dataset.train_labels
    test_images, test_labels = dataset.test_images, dataset.test_labels
    net.fit(train_images, train_labels)
        

# Loss function
# YOLONet loss function is composed of three parts:
# 1. Classification loss
# 2. Confidence loss
# 3. Localization loss

def yolo_loss(pred, target, lambd_coord = 5, lambd_noobj = 0.5):

    loss = 0

    batch_size = pred.size(0)
    n_grid_side = pred.size(1)

    pred = pred.view(batch_size, n_grid_side, n_grid_side, 5 + N_CLASSES)
    target = target.view(batch_size, n_grid_side, n_grid_side, 5 + N_CLASSES)
    
    # 1. Localization loss
    loss_xy = torch.sum((pred[:, :, :, 0:2] - target[:, :, :, 0:2]) ** 2)
    loss_wh = torch.sum((pred[:, :, :, 2:4] - target[:, :, :, 2:4]) ** 2)

    # 2. Confidence loss
    obj_loss = torch.sum((pred[:, :, :, 4] - target[:, :, :, 4]) ** 2)
    noobj_loss = torch.sum((pred[:, :, :, 4] - target[:, :, :, 4]) ** 2)

    # 3. Classification loss
    class_loss = torch.sum((pred[:, :, :, 5:] - target[:, :, :, 5:]) ** 2)

    loss = lambd_coord * (loss_xy + loss_wh) + obj_loss + lambd_noobj * noobj_loss + class_loss

    return loss

def get_train_batches(train_images, train_labels, batch_size = BATCH_SIZE):
    indices = np.arange(len(train_images))
    np.random.shuffle(indices)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i: i + batch_size]
        batch_images = train_images[batch_indices]
        batch_labels = train_labels[batch_indices]
        yield batch_images, batch_labels

def mAP(pred, target):
    
    # sort pred by confidence
    pass

def iou(box1, box2):
    # box1, box2: list of (xmin, ymin, xmax, ymax)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area
    

def label2box(label, threshold = 0.6):
    # label: torch.Tensor (N_GRID_SIDE, N_GRID_SIDE, 5 + N_CLASSES)
    # boxes: list of (cx, cy, w, h, cls, confidence)
    label_confidence = label[:, :, 4]
    label_class_prob = label[:, :, 5:]
    label_scores = label_confidence * label_class_prob
    # filter out the boxes with confidence < threshold
    filtering_mask = label_scores > threshold
    boxes = []
    for i in range(N_GRID_SIDE):
        for j in range(N_GRID_SIDE):
            if filtering_mask[i, j]:
                cx = (i + 0.5) / N_GRID_SIDE
                cy = (j + 0.5) / N_GRID_SIDE
                w = label[i, j, 2]
                h = label[i, j, 3]
                cls = np.argmax(label_class_prob[i, j])
                confidence = label_confidence[i, j]
                boxes.append((cx, cy, w, h, cls, confidence))
    
    return boxes
                
    
def non_max_suppression(boxes, iou_threshold = 0.5, max_boxes = 10):
    # boxes: list of (cx, cy, w, h, cls, confidence)
    
    # copy the list
    original_boxes = boxes.copy()
    keep_boxes = []
    while original_boxes and len(keep_boxes) < max_boxes:
        max_confidence_box = max(original_boxes, key=lambda x: x[5])
        keep_boxes.append(max_confidence_box)
        original_boxes.remove(max_confidence_box)
        for box in boxes:
            if iou(max_confidence_box, box) > iou_threshold:
                boxes.remove(box)
    
    return keep_boxes
    
