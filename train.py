import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import pytorch_lightning as pl
from model.SimpleNet import SimpleNet
from model.Linear import Linear
from data.data_process import TrainSet
from pytorch_lightning.metrics.functional import accuracy
from loss.CELoss import ce_loss
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

total_imgs = 11554
train_imgs = int(11554 * 0.9)
test_imgs  = total_imgs - train_imgs
class TrainNet(pl.LightningModule):

    def __init__(self, in_channels=1, num_classes=68):
        super().__init__()
        self.model = SimpleNet(in_channels, num_classes)
        #self.model = Linear(in_channels, num_classes)
        self.loss = ce_loss
        self.train_loss = []
        self.val_loss = []
        self.val_acc = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop. It is independent of forward
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)

        return loss

    def training_epoch_end(self, outputs):
        #print('training epoch end', outputs)
        step_loss = []
        for val in outputs:
            step_loss.append(val['loss'].item())
        self.train_loss.append(sum(step_loss)/len(step_loss))


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss, acc

    def validation_epoch_end(self, outputs):
        #print('valid epoch end', outputs)
        val_loss = []
        acc_loss = []
        for val in outputs:
            val_loss.append(val[0].item())
            acc_loss.append(val[1].item())

        self.val_loss.append(sum(val_loss) / len(val_loss))
        self.val_acc.append(100 * sum(acc_loss) / len(acc_loss))

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == 'fit' or stage is None:
            trainset = TrainSet(train_imgs, train=True)
            self.faceimg_train, self.faceimg_val = random_split(trainset, [train_imgs - test_imgs, test_imgs])

        # Assign test dataset for use in dataloader(s)
        if stage == 'test' or stage is None:
            self.faceimg_test = TrainSet(test_imgs, train=False)

    def train_dataloader(self):
        return DataLoader(self.faceimg_train, batch_size=32, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self.faceimg_val, batch_size=32, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.faceimg_test, batch_size=1)

def train():
    max_epochs = 35
    checkpoint_callback = ModelCheckpoint(monitor='val_acc',
                                          save_top_k=1,
                                          mode='max',
                                          verbose=True,
                                          dirpath=os.path.join(os.getcwd(), 'checkpoint'),
                                          filename='face--{epoch:02d}-{val_loss:.2f}-{val_acc:.2f}')

    #early_stopping = EarlyStopping('val_loss', min_delta=0.001, patience=4, verbose=True, mode='min')

    model = TrainNet(1)
    trainer = pl.Trainer(gpus=-1,
                         max_epochs=max_epochs,
                         weights_summary='full',
                         progress_bar_refresh_rate=20,
                         callbacks=[checkpoint_callback])
    trainer.fit(model)

    x_axis = range(max_epochs)
    plt.title("Train and Val Loss")
    plt.plot(x_axis, model.train_loss, color='green', label='training loss')
    plt.plot(x_axis, model.val_loss[1:], color='red', label='val loss')
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel('Loss')
    plt.show()
    plt.title("Val Accuracy")
    plt.plot(x_axis, model.val_acc[1:], color='blue', label='val acc')
    plt.xlabel("Epoch")
    plt.ylabel('Acc')
    plt.show()
    trainer.test()

if __name__ == "__main__":
    train()
