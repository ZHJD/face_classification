import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils.read_data import read_mat


file_path = "D:/face_classification/dataset/PIE_32x32.mat"

transform = transforms.Compose([
    # 转化到0-1之间
    transforms.ToTensor()
])


class TrainSet(Dataset):

    def __init__(self, imgs, train=True):
        self.data, self.gt = read_mat(file_path)
        self.gt = torch.from_numpy(self.gt)
        self.gt = self.gt.type(torch.LongTensor)
        if train:
            self.data = self.data[:imgs]
            self.gt = self.gt[:imgs]
        else:
            self.data = self.data[-imgs:]
            self.gt = self.gt[-imgs:]

        self.transform = transform

    def __getitem__(self, item):
        x, y = self.data[item], self.gt[item]

        # one_hot 编码
        #y = torch.from_numpy(y)
        if self.transform:
            x = self.transform(x)
        else:
            x = torch.from_numpy(x)
        return x, y

    def __len__(self):
        return self.data.shape[0]


if __name__ == "__main__":
    trainset = TrainSet(1)
    print(trainset[0])