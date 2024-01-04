import numpy as np
import torch
import tqdm
from dataloader import *
from model import *
from constants import *
from yolo_utils import *
from loss import *
from evaluate import *
from dataloader_utils import *

config = {
    "learning_rate": 0.001, # The initial learning rate
    "is_set_lr": False, # whether to set lr to 0.001 after start_from epoch
    "set_lr": 0.001, # "set_lr" is used to set the lr to 0.001 after start_from epoch
    "momentum": 0.937, # The momentum
    "weight_decay": 0.0005, # The weight decay
    "step_size": 10, # step size for StepLR scheduler
    "gamma": 0.8, # gamma for StepLR scheduler
    "n_epoch": 100, # The total number of epochs
    "device": "cuda", # device: cuda or cpu
    "checkpoint_path": "checkpoints/", # path to save checkpoints
    "checkpoint_interval": 50, # interval to save checkpoints
    "eval_interval": 10, # interval to evaluate on test set
    "start_from": 0,  # start from which epoch
    "more_data": False, # whether to use whole VOC2007 data
    "augment": True # whether to use data augmentation
}

class Train(object):
    """
    Train the model
    """

    def __init__(self, config):

        # initialize the dataloader, model, optimizer, scheduler
        self.dataloader = DataLoader()
        self.dataloader.load_data(more_data=config["more_data"])
        self.model = Net(device=config["device"])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config["step_size"], gamma=config["gamma"])

        # set device, checkpoint_path, checkpoint_interval, eval_interval
        self.device = config["device"]
        self.checkpoint_path = config["checkpoint_path"]
        self.checkpoint_interval = config["checkpoint_interval"]
        self.eval_interval = config["eval_interval"]
        # Other parameters
        self.is_set_lr = config["is_set_lr"]
        self.set_lr = config["set_lr"]
        self.start_from = config["start_from"]
        self.more_data = config["more_data"]
        self.augment = config["augment"]
        self.n_epoch = config["n_epoch"]

        # create checkpoint_path if not exists
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)
        self.fix_random_seed()

        # load checkpoint if start_from != 0
        if config["start_from"] != 0:
            self.checkpoint_load(config["start_from"])
    
    def fix_random_seed(self):
        """
        Fix random seed for reproducibility
        """
        torch.manual_seed(SEED)
        np.random.seed(SEED)
    
    def checkpoint_save(self, epoch):
        """
        Save the model weights, optimizer, scheduler
        """
        self.model.save_weights(self.checkpoint_path + "weights{}.pth".format(epoch))
        torch.save(self.optimizer.state_dict(), self.checkpoint_path + "optimizer{}.pth".format(epoch))
        torch.save(self.scheduler.state_dict(), self.checkpoint_path + "scheduler{}.pth".format(epoch))
        print("Weights saved.")
    
    def checkpoint_load(self, epoch):
        """
        Load the model weights, optimizer, scheduler
        """
        self.model.load_weights(self.checkpoint_path + "weights{}.pth".format(epoch))
        try:
            self.optimizer.load_state_dict(torch.load(self.checkpoint_path + "optimizer{}.pth".format(epoch)))
            self.scheduler.load_state_dict(torch.load(self.checkpoint_path + "scheduler{}.pth".format(epoch)))
        except:
            pass
    
    def train_one_epoch(self):
        """
        Train one epoch
        """

        losses = []
        # initialize the progress bar
        n_batches = len(self.dataloader.train_images) // BATCH_SIZE
        pbar = tqdm.tqdm(total=n_batches)

        # loop over the batches
        for batch_images, batch_labels in self.dataloader.get_train_batch(is_aug=self.augment):

            # update the progress bar
            pbar.update(1)

            # convert to tensor
            batch_images = np.array(batch_images)
            batch_labels = np.array(batch_labels)
            batch_images_tensor = torch.from_numpy(batch_images).to(self.model.device).float()
            batch_labels_tensor = torch.from_numpy(batch_labels).to(self.model.device).float()

            # permute the tensor, put the channel dimension first
            batch_images_tensor = batch_images_tensor.permute(0, 3, 1, 2)
            batch_labels_tensor = batch_labels_tensor.reshape(-1, N_GRID_SIDE, N_GRID_SIDE, 5 * N_BBOX + N_CLASSES)

            # forward, backward, update
            pred = self.model.forward(batch_images_tensor)
            # loss = yolo_loss_sq(pred, batch_labels_tensor)
            loss = yolo_loss_be(pred, batch_labels_tensor)
            losses.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # update the progress bar
            pbar.set_description("Loss: {:.2f}".format(np.mean(losses)))
            # clear the cache
            torch.cuda.empty_cache()

        # close the progress bar
        pbar.close()
        # remove the progress bar
        print("\r", end="")
        return np.mean(np.array(losses))
    
    def train(self):
        """
        Train the model
        """

        losses = []
        # set lr if is_set_lr is True
        if self.is_set_lr:
            new_lr = self.set_lr
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = new_lr

        # loop over the epochs
        for epoch in range(self.start_from + 1, self.n_epoch + 1):
            # train one epoch
            loss = self.train_one_epoch()
            # track the loss
            losses.append(loss)
            # print the loss
            print("Epoch {}/{}: ".format(epoch, self.n_epoch), end="")
            print("Loss: {:.4f}".format(loss), end="")
            print(", lr: {:.4f}".format(self.scheduler.get_last_lr()[0]))
            # shedule the lr
            self.scheduler.step()

            # save checkpoint
            if epoch % self.checkpoint_interval == 0:
                self.checkpoint_save(epoch)
                print("Checkpoint {} saved.".format(epoch))

            # evaluate
            if epoch % self.eval_interval == 0:
                self.evaluate_on_train()
        self.checkpoint_save(self.n_epoch)
        print("Training done.")
        self.evaluate_on_test()
    
    def evaluate_on_test(self):
        """
        Evaluate on test set
        Calculate mAP and AP for each class
        """

        # load test set
        test_images = self.dataloader.test_images
        test_boxes = [label["boxes"] for label in self.dataloader.test_labels]
        # predict bounding boxes on test set
        pred_boxes = self.model.predict(test_images)
        # calculate mAP and AP for each class
        mAP, APs = compute_mAP(pred_boxes, test_boxes, print_pr=True)
        print("mAP on test: {:.4f}".format(mAP))
        print("APs: {}".format(APs))
    
    def evaluate_on_train(self, max_n = 10000):
        """
        Evaluate on train set
        Calculate mAP and AP for each class
        max_n: maximum number of images to evaluate
        """

        # load train set
        max_n = min(max_n, len(self.dataloader.train_images))
        train_images = [label["resized_image"] for label in self.dataloader.train_labels[:max_n]]
        train_boxes = [label["resized_boxes"] for label in self.dataloader.train_labels[:max_n]]
        # predict bounding boxes on train set
        pred_boxes = self.model.predict(train_images)
        # calculate mAP and AP for each class
        mAP, APs = compute_mAP(pred_boxes, train_boxes)
        print("mAP on train: {:.4f}".format(mAP))
        # print("APs: {}".format(APs))

if __name__ == "__main__":
    train = Train(config)
    train.train()
