from pytorch_lightning import  LightningModule
import torch
from data.data_process import TrainSet
from torch.utils.data import DataLoader
from train import TrainNet
import matplotlib.pyplot as plt
from torch import nn
total_imgs = 11554
train_imgs = int(11554 * 0.9)
test_imgs  = total_imgs - train_imgs
import numpy as np
import matplotlib.pyplot as plt

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


visualisation = {}
i = 0
results = []
keys = []
def hook(m, i, o):
    keys.append(m)
    results.append(o.detach().cpu())

def get_all_layers(model):
    for name, layer in model._modules.items():
        if isinstance(layer, nn.Sequential):
            pass
        else:
            layer.register_forward_hook(hook)

if __name__ == '__main__':
    path = "D:/face_classification/checkpoint/face--epoch=34-val_loss=0.13-val_acc=0.98.ckpt"
    pretrained_model = load_weights(path)
    faceimg_test = TrainSet(test_imgs, train=False)
    test_loader = DataLoader(faceimg_test, batch_size=1)
    test_acc = forward(pretrained_model, test_loader)
    # print(test_acc)
    model = pretrained_model.model
    # get_all_layers(model)
    # for x, y in test_loader:
    #     out = model(x)
    #     for key, layer in zip(keys, results):
    #         print(key, layer.shape)
    #         for i in range(layer.shape[1]):
    #             image = layer[0, i]
    #             #print(image)
    #             # image = np.transpose(image, (1, 2, 0))
    #             plt.axis('off')
    #             plt.title(key)
    #             plt.imshow(image * 255, cmap='gray')
    #             plt.show()
    #     break
    for key, val in model.state_dict().items():
        print(key)
        print(val.shape)