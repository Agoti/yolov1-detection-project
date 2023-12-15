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

    def __init__(self, classes):
        super(Net, self).__init__()
        self.classes = classes
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
            nn.Linear(4096, N_GRID_SIDE * N_GRID_SIDE * (5 + len(classes))),
        )

    def forward(self, x):
        x = self.resnet(x)
        x = self.yolo_tail(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcs(x)
        x = x.reshape(x.size(0), N_GRID_SIDE, N_GRID_SIDE, 5 + len(self.classes))
        return x

    def fit(self, train_images, train_labels, n_epoch=10, learning_rate=0.001, train_boxes = None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        for epoch in range(n_epoch):
            print("Epoch: {}".format(epoch))
            for batch_images, batch_labels in get_train_batches(train_images, train_labels):
                batch_images = torch.Tensor(batch_images).to(device)
                batch_labels = torch.Tensor(batch_labels).to(device).float()
                pred = self.forward(batch_images)
                # print(pred.shape)
                loss = yolo_loss(pred, batch_labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
            # compute mAP
            if train_boxes is not None:
                pred_boxes = self.predict(train_images, train_boxes)
                mAP = self.compute_mAP(pred_boxes, train_boxes, self.classes)
                print("Epoch: {}, mAP: {}".format(epoch, mAP))
    
    def predict(self, test_images, test_boxes):
        test_images = torch.Tensor(test_images)
        pred = self.forward(test_images)
        pred_boxes = label2box(pred)
        pred_boxes = non_max_suppression(pred_boxes)
        # compute mAP
        mAP = self.compute_mAP(pred_boxes, test_boxes, self.classes)
        return pred, mAP
    
    def compute_mAP(self, pred_boxes, GT_boxes, classes, iou_threshold = 0.5):
        # pred_boxes: list of box list of a single image
        # GT_boxes: list of box list of a single image

        # rearrange the pred_boxes and GT_boxes
        pred_boxes_class = {cls: [] for cls in classes}
        GT_boxes_class = {cls: [] for cls in classes}
        pred = pred_boxes.copy()
        GT = GT_boxes.copy()
        for i in range(len(pred)):
            boxes = pred[i]
            for box in boxes:
                box["idx"] = i
                pred_boxes_class[box["cls"]].append(box)
        for i in range(len(GT)):
            boxes = GT[i]
            for box in boxes:
                box["idx"] = i
                GT_boxes_class[box["cls"]].append(box)

        # sort the pred_boxes_class by confidence
        APs = []
        for cls in classes:
            TP, FP = 0, 0
            total_gt_boxes = len(GT_boxes_class[cls])
            precision, recall = [], []

            pred_boxes_class[cls].sort(key=lambda x: x["confidence"], reverse=True)

            for box in pred_boxes_class[cls]:
                iou_max = 0
                for GT_box in GT[box["idx"]]:
                    iou_score = iou(box, GT_box)
                    if iou_score > iou_max:
                        iou_max = iou_score
                        matched_GT_box = GT_box

                if iou_max > iou_threshold:
                    TP += 1
                    GT[box["idx"]].remove(matched_GT_box)
                else:
                    FP += 1

                precision.append(TP / (TP + FP) if (TP + FP) > 0 else 0)
                recall.append(TP / total_gt_boxes if total_gt_boxes > 0 else 0)

            # Calculate AP using trapezoidal rule
            sorted_indices = torch.argsort(torch.Tensor(recall))
            sorted_recall = torch.Tensor(recall)[sorted_indices]
            sorted_precision = torch.Tensor(precision)[sorted_indices]
            if len(sorted_recall) > 0 and len(sorted_precision) > 0:
                AP = torch.trapz(sorted_precision, sorted_recall)
            else:
                AP = 0
            APs.append(AP.item() if not torch.isnan(AP) else 0)

        mAP = sum(APs) / len(APs) if len(APs) > 0 else 0
        return mAP

def main():
    dataset = DataLoader()
    dataset.load_data()
    train_images, train_labels = dataset.train_images, dataset.train_labels
    test_images, test_labels = dataset.test_images, dataset.test_labels
    train_boxes = [msg["boxes"] for msg in dataset.train_msg]
    test_boxes = [msg["boxes"] for msg in dataset.test_msg]
    net = Net(dataset.classes)
    net.fit(train_images, train_labels, n_epoch=10, learning_rate=0.001)
    boxes, mAP = net.predict(test_images, test_boxes)
    print("mAP: {}".format(mAP))

# ---- utils ----

# Loss function
# YOLONet loss function is composed of three parts:
# 1. Classification loss
# 2. Confidence loss
# 3. Localization loss

def yolo_loss(pred, target, lambd_coord = 5, lambd_noobj = 0.5):

    loss = 0
    batch_size = BATCH_SIZE
    n_grid_side = N_GRID_SIDE
    nclasses = N_CLASSES
    pred = pred.view(batch_size, n_grid_side, n_grid_side, nclasses + 5)
    target = target.view(batch_size, n_grid_side, n_grid_side, nclasses + 5)
    # 1. Localization loss
    loss_xy = torch.sum((pred[:, :, :, 0:2] - target[:, :, :, 0:2]) ** 2 * target[:, :, :, 4].unsqueeze(-1))
    loss_wh = torch.sum((pred[:, :, :, 2:4] ** 0.5 - target[:, :, :, 2:4] ** 0.5) ** 2 * target[:, :, :, 4].unsqueeze(-1))
    # 2. Confidence loss
    obj_loss = torch.sum((pred[:, :, :, 4] - target[:, :, :, 4]) ** 2 * target[:, :, :, 4])
    noobj_loss = torch.sum((pred[:, :, :, 4] - target[:, :, :, 4]) ** 2 * (1 - target[:, :, :, 4]))
    # 3. Classification loss
    class_loss = torch.sum((pred[:, :, :, 5:] - target[:, :, :, 5:]) ** 2 * target[:, :, :, 4].unsqueeze(-1))
    loss = lambd_coord * (loss_xy + loss_wh) + obj_loss + lambd_noobj * noobj_loss + class_loss

    return loss

def get_train_batches(train_images, train_labels, batch_size = BATCH_SIZE):
    indices = np.arange(len(train_images))
    np.random.shuffle(indices)
    train_images_numpy = np.array(train_images)
    train_labels_numpy = np.array(train_labels)
    for i in range(0, len(indices), batch_size):
        batch_indices = indices[i: i + batch_size]
        batch_images = train_images_numpy[batch_indices]
        batch_labels = train_labels_numpy[batch_indices]
        batch_images = np.transpose(batch_images, (0, 3, 1, 2))
        yield batch_images, batch_labels


def iou(box1, box2):
    # box: dictionary with keys (xmin, ymin, xmax, ymax)
    
    x1min, y1min, x1max, y1max = box1["xmin"], box1["ymin"], box1["xmax"], box1["ymax"]
    x2min, y2min, x2max, y2max = box2["xmin"], box2["ymin"], box2["xmax"], box2["ymax"]
    x1 = max(x1min, x2min)
    y1 = max(y1min, y2min)
    x2 = min(x1max, x2max)
    y2 = min(y1max, y2max)
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    union_area = (x1max - x1min) * (y1max - y1min) + (x2max - x2min) * (y2max - y2min) - inter_area
    return inter_area / union_area

def bbox_to_corners(box):
    # box: dictionary with keys (cx, cy, w, h)
    #   but keys (xmin, ymin, xmax, ymax) are 0 currently
    
    bbox = box.copy()
    cx, cy, w, h = bbox["cx"], bbox["cy"], bbox["w"], bbox["h"]
    width = bbox["width"]
    height = bbox["height"]
    xmin = int((cx - w / 2) * width)
    ymin = int((cy - h / 2) * height)
    xmax = int((cx + w / 2) * width)
    ymax = int((cy + h / 2) * height)
    bbox["xmin"] = xmin
    bbox["ymin"] = ymin
    bbox["xmax"] = xmax
    bbox["ymax"] = ymax
    return bbox

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
                box = {"cx": cx, "cy": cy, "w": w, "h": h,\
                        "class": cls, "confidence": confidence,\
                        "width": INPUT_SIDE, "height": INPUT_SIDE}
                box = bbox_to_corners(box)
                boxes.append(box)

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
    
if __name__ == "__main__":
    main()
