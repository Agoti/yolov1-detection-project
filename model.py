import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import tqdm
from constants import *
from dataloader import *
from yolo_utils import *
from loss import *
from dataloader_utils import *

# YOLO Net
# To save time, use pre-trained ResNet34
# remove the last two layers, and connect 
# YOLONet layers(last 4 conv and 2 fc)
class Net(nn.Module):

    def __init__(self, device="cuda"):
        super().__init__()
        if device == "cuda":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")

        resnet = models.resnet34(weights="ResNet34_Weights.IMAGENET1K_V1")
        # resnet = models.resnet34(pretrained=True)
        # remove the last two layers
        resnet_out_channels = resnet.fc.in_features
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        # YOLO layers
        self.yolo_neck = nn.Sequential(
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
        self.yolo_head = nn.Sequential(
            nn.Linear(N_GRID_SIDE * N_GRID_SIDE * 1024, 4096),
            nn.BatchNorm1d(4096),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(4096, N_GRID_SIDE * N_GRID_SIDE * (5 * N_BBOX + N_CLASSES)),
            nn.BatchNorm1d(N_GRID_SIDE * N_GRID_SIDE * (5 * N_BBOX + N_CLASSES)),
            # add sigmoid to the last layer
            nn.Sigmoid()
        )

        # initialize the weights
        self._initialize_weights()
        self.to(self.device)

    def _initialize_weights(self):
        for m in self.yolo_neck.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
        for m in self.yolo_head.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        x = self.resnet(inputs)
        x = self.yolo_neck(x)
        x = x.reshape(x.size()[0], -1)
        x = self.yolo_head(x)
        x = x.reshape(-1, N_GRID_SIDE, N_GRID_SIDE, 5 * N_BBOX + N_CLASSES)
        return x

    def fit(self, dataloader, n_epoch=10, learning_rate=0.001):

        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.937, weight_decay=0.0005)
        # milestone: 70
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70], gamma=0.1)
        train_images = dataloader.train_images
        train_labels = [label["label"] for label in dataloader.train_labels]
        for epoch in range(n_epoch):
            print("Epoch: {}".format(epoch))
            losses = []
            for batch_images, batch_labels in get_train_batches(train_images, train_labels):
                # convert batch_images and batch_labels to FloatTensor
                # move to device
                batch_images = torch.Tensor(batch_images).to(self.device).float()
                batch_labels = torch.Tensor(batch_labels).to(self.device).float()
                batch_images = batch_images.permute(0, 3, 1, 2)
                batch_labels = batch_labels.reshape(-1, N_GRID_SIDE, N_GRID_SIDE, 5 * N_BBOX + N_CLASSES)
                pred = self.forward(batch_images)
                loss = yolo_loss_be(pred, batch_labels)
                # print("Loss: {}".format(loss.item()))
                # check when loss is nan
                if torch.isnan(loss):
                    print("Loss is nan.")
                    print("Epoch: {}".format(epoch))
                    print("Pred: {}".format(pred))
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            # print loss with 2 decimal places
            print("Loss: {:.2f}".format(np.mean(losses)))
    
    def predict(self, test_images):
        # test_images: list of ndarrays
        # convert to a simple array
        test_images = np.array(test_images)
        test_images = torch.Tensor(test_images).to(self.device).float()
        test_images = test_images.permute(0, 3, 1, 2)
        # pred = self.forward(test_images)
        # send test_images in batches
        pred_boxes_list = []
        n_batches = len(test_images) // BATCH_SIZE
        print("Predicting")
        with torch.no_grad():
            with tqdm.tqdm(total=n_batches) as pbar:
                for i in range(n_batches):
                    # print("Predicting batch {}/{}".format(i, n_batches))
                    pbar.update(1)
                    batch_images = test_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
                    batch_pred = self.forward(batch_images)
                    for j in range(batch_pred.size()[0]):
                        pred = batch_pred[j].detach().cpu().numpy()
                        pred_boxes = label2box(pred)
                        pred_boxes = non_max_suppression(pred_boxes)
                        pred_boxes_list.append(pred_boxes)
                    del batch_images, batch_pred
            # last batch
            if len(test_images) % BATCH_SIZE != 0:
                batch_images = test_images[n_batches * BATCH_SIZE:]
                batch_pred = self.forward(batch_images)
                for j in range(batch_pred.size()[0]):
                    pred = batch_pred[j].detach().cpu().numpy()
                    pred_boxes = label2box(pred)
                    pred_boxes = non_max_suppression(pred_boxes)
                    pred_boxes_list.append(pred_boxes)
                del batch_images, batch_pred

        return pred_boxes_list
    
    def save_weights(self, path):
        torch.save(self.state_dict(), path)
        print("Weights saved.")

    def load_weights(self, path):
        if self.device == torch.device("cpu"):
            self.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load(path))
        self.eval()
        print("Weights loaded.")

    

def main():
    dataset = DataLoader()
    dataset.load_data()
    net = Net()
    net.fit(dataset, n_epoch=50, learning_rate=0.0001)

    
if __name__ == "__main__":
    main()
