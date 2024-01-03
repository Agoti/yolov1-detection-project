import numpy as np
import torch
import tqdm
import gc
from dataloader import *
from model import *
from constants import *
from yolo_utils import *
from loss import *
from evaluate import *
from dataloader_utils import *

config = {
    "learning_rate": 0.001,
    "set_lr": 0.0005, # "set_lr" is used to set the lr to 0.001 after start_from epoch
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "step_size": 10,
    "gamma": 0.9,
    "n_epoch": 150,
    "device": "cuda",
    "checkpoint_path": "checkpoints/",
    "checkpoint_interval": 10,
    "eval_interval": 2,
    "start_from": 50, 
    "more_data": True, # whether to use VOC2007 data
    "augment": True
}

class Train(object):
    def __init__(self, config):
        self.dataloader = DataLoader()
        self.dataloader.load_data(more_data=config["more_data"])
        self.model = Net(device=config["device"])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], betas=(0.9, 0.999))
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config["step_size"], gamma=config["gamma"])
        self.n_epoch = config["n_epoch"]
        self.device = config["device"]
        self.checkpoint_path = config["checkpoint_path"]
        self.checkpoint_interval = config["checkpoint_interval"]
        self.eval_interval = config["eval_interval"]
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.fix_random_seed()
        if config["start_from"] != 0:
            self.checkpoint_load(config["start_from"])
    
    def fix_random_seed(self):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    
    def checkpoint_save(self, epoch):
        self.model.save_weights(self.checkpoint_path + "weights{}.pth".format(epoch))
        torch.save(self.optimizer.state_dict(), self.checkpoint_path + "optimizer{}.pth".format(epoch))
        torch.save(self.scheduler.state_dict(), self.checkpoint_path + "scheduler{}.pth".format(epoch))
        print("Weights saved.")
    
    def checkpoint_load(self, epoch):
        self.model.load_weights(self.checkpoint_path + "weights{}.pth".format(epoch))
        try:
            self.optimizer.load_state_dict(torch.load(self.checkpoint_path + "optimizer{}.pth".format(epoch)))
            self.scheduler.load_state_dict(torch.load(self.checkpoint_path + "scheduler{}.pth".format(epoch)))
        except:
            pass
    
    def train_one_epoch(self):
        losses = []
        n_batches = len(self.dataloader.train_images) // BATCH_SIZE
        pbar = tqdm.tqdm(total=n_batches)
        for batch_images, batch_labels in self.dataloader.get_train_batch(is_aug=config["augment"]):
            # 代码是这么写的(肯定)
            pbar.update(1)
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            batch_images_tensor = torch.from_numpy(batch_images).to(self.model.device).float()
            batch_labels_tensor = torch.from_numpy(batch_labels).to(self.model.device).float()
            batch_images_tensor = batch_images_tensor.permute(0, 3, 1, 2)
            batch_labels_tensor = batch_labels_tensor.reshape(-1, N_GRID_SIDE, N_GRID_SIDE, 5 * N_BBOX + N_CLASSES)
            pred = self.model.forward(batch_images_tensor)
            # loss = yolo_loss_sq(pred, batch_labels_tensor)
            loss = yolo_loss_be(pred, batch_labels_tensor)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            pbar.set_description("Loss: {:.2f}".format(np.mean(losses)))
            torch.cuda.empty_cache()

        pbar.close()
        # remove the progress bar
        print("\r", end="")
        return np.mean(np.array(losses))
    
    def train(self):
        losses = []
        # set lr to 0.001
        new_lr = config["set_lr"]
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr
        for epoch in range(config["start_from"] + 1, self.n_epoch + 1):
            loss = self.train_one_epoch()
            losses.append(loss)
            print("Epoch {}/{}: ".format(epoch, self.n_epoch), end="")
            print("Loss: {:.4f}".format(loss), end="")
            print(", lr: {:.4f}".format(self.scheduler.get_last_lr()[0]))
            self.scheduler.step()
            if epoch % self.checkpoint_interval == 0:
                self.checkpoint_save(epoch)
                print("Checkpoint {} saved.".format(epoch))
            if epoch % self.eval_interval == 0:
                self.evaluate_on_test()
                self.evaluate_on_train()
        self.checkpoint_save(self.n_epoch)
        print("Training done.")
        self.evaluate_on_test()
    
    def evaluate_on_test(self):
        test_images = self.dataloader.test_images
        test_boxes = [label["boxes"] for label in self.dataloader.test_labels]
        pred_boxes = self.model.predict(test_images)
        mAP, APs = compute_mAP(pred_boxes, test_boxes, print_pr=True)
        print("mAP on test: {:.4f}".format(mAP))
        print("APs: {}".format(APs))
    
    def evaluate_on_train(self, max_n = 10000):
        max_n = min(max_n, len(self.dataloader.train_images))
        train_images = [label["resized_image"] for label in self.dataloader.train_labels[:max_n]]
        train_boxes = [label["resized_boxes"] for label in self.dataloader.train_labels[:max_n]]
        pred_boxes = self.model.predict(train_images)
        mAP, APs = compute_mAP(pred_boxes, train_boxes)
        print("mAP on train: {:.4f}".format(mAP))
        # print("APs: {}".format(APs))

if __name__ == "__main__":
    train = Train(config)
    train.train()
