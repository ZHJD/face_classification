import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt

def one_hot(x, num_classes=None):
    if not num_classes:
        num_classes = np.max(x) + 1
    one_hot_targets = np.eye(num_classes)[x]
    return one_hot_targets

# 共68类，编号是1-68
def read_mat(file_path, shuffle = True):
    data = scio.loadmat(file_path)
    face_images = data['fea']
    # 转化为图片的形式
    face_images = face_images.reshape(-1, 32, 32, 1, order='f')
    label = np.array(data['gnd'])
    label = label.reshape(-1)
    label = label - 1
    #label = one_hot(label - 1)

    if shuffle:
        # 打乱整个数据集，因为数据集是有规律的，依次按照第一类、第二类的顺序在排列
        np.random.seed(42)
        permutation = np.random.permutation(face_images.shape[0])
        face_images = face_images[permutation,:,:]
        label = label[permutation]

    return face_images, label

if __name__ == "__main__":
    file_path = "../dataset/PIE_32x32.mat"
    data, label = read_mat(file_path, False)
    print(data.shape)
    print(label[:5])

    last = -1
    idx = []
    for i in range(data.shape[0]):
        if last != label[i]:
            idx.append(i)
            last = label[i]
    print(label[0], label[1], label[2])


    plt.subplot(1, 3, 1)
    plt.imshow(data[0], cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(data[5], cmap='gray')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(data[2], cmap='gray')
    plt.axis('off')
    plt.show()


