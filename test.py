from pytorch_lightning import  LightningModule
import torch
from data.data_process import TrainSet
from torch.utils.data import DataLoader
from train import TrainNet
import matplotlib.pyplot as plt

total_imgs = 11554
train_imgs = int(11554 * 0.9)
test_imgs  = total_imgs - train_imgs

def load_weights(path):
    pretrained_model = TrainNet.load_from_checkpoint(path)
    pretrained_model.freeze()
    return pretrained_model

def forward(pretrained_model, test_loader):
    test_num = 0
    true_pred = 0
    for x, y in test_loader:
        out = pretrained_model(x)
        preds = torch.argmax(out, dim=1)
        test_num += 1
        if y == preds:
            true_pred += 1
    return true_pred / test_num

if __name__ == '__main__':
    path = "D:/face_classification/checkpoint/face--epoch=34-val_loss=0.13-val_acc=0.98.ckpt"
    pretrained_model = load_weights(path)
    faceimg_test = TrainSet(test_imgs, train=False)
    test_loader = DataLoader(faceimg_test, batch_size=1)
    test_acc = forward(pretrained_model, test_loader)
    print(test_acc)


