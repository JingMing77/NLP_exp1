# -*- coding:utf-8 -*-
import random

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np


def Myloader(path):
    return Image.open(path).convert('RGB')


def find_label(str):
    # label (dog: 1 / cat: 0)
    if 'dog' in str:
        return 1
    elif 'cat' in str:
        return 0
    else:
        raise NotImplementedError(f'filename must include "cat" or "dog"')


# get a list of paths and labels.
def get_file_list(path, lens):
    data = []
    label = find_label(path)
    for i in range(lens[0], lens[1]):
        data.append([path % i, label])

    return data


def sp_noise(image, prob):
    """
    添加椒盐噪声
    prob:噪声比例
    """
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


def gasuss_noise(image, mean=0, var=0.001):
    """
        添加高斯噪声
        mean : 均值
        var : 方差
    """
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)
    out = image + noise
    if out.min() < 0:
        low_clip = -1.
    else:
        low_clip = 0.
    out = np.clip(out, low_clip, 1.0)
    out = np.uint8(out * 255)
    return out


class MyDataset(Dataset):
    def __init__(self, data, transform, loader, enhance=False, rand_list=None):
        self.data = data
        self.transform = transform
        self.loader = loader
        self.enhance = enhance
        self.rand_list = rand_list

    def __getitem__(self, item):
        img, label = self.data[item]
        img = self.loader(img)
        img = self.transform(img)
        if self.enhance and self.rand_list is not None:
            if self.rand_list[item] > 0.8:
                img = sp_noise(img, 0.3)
                img = Image.fromarray(img.transpose((1, 2, 0)))
                img = transforms.RandomHorizontalFlip(p=0.5)(img)
                img = transforms.ToTensor()(img)
            elif self.rand_list[item] < 0.2:
                img = gasuss_noise(img, mean=0, var=0.01)
                img = Image.fromarray(img.transpose((1, 2, 0)))
                img = transforms.RandomVerticalFlip(p=0.5)(img)
                img = transforms.ToTensor()(img)

        return img, label

    def __len__(self):
        return len(self.data)


def load_data(H=256, W=256, batch_size=64, num_workers=0, enhance=False):
    print('data processing...')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))  # normalization
    ])
    data1 = get_file_list('./data/train/cat.%d.jpg', [0, 1000])
    data2 = get_file_list('data/train/dog.%d.jpg', [0, 1000])
    data3 = get_file_list('data/val/cat.%d.jpg', [1000, 1250])
    data4 = get_file_list('data/val/dog.%d.jpg', [1000, 1250])
    data = data1 + data2 + data3 + data4  # train (1000+1000) + val (250+250)

    # shuffle
    np.random.shuffle(data)
    # train: val: test = 6: 2: 2
    total_n = len(data)
    train_data, val_data, test_data = data[:int(0.6 * total_n)], \
                                      data[int(0.6 * total_n):int(0.8 * total_n)], \
                                      data[int(0.8 * total_n):]
    rand_l = None
    if enhance:
        # add noise
        train_data = train_data + train_data
        rand_l = np.random.rand(len(train_data))

    train_data = MyDataset(train_data, transform=transform, loader=Myloader, enhance=enhance, rand_list=rand_l)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_data = MyDataset(val_data, transform=transform, loader=Myloader)
    val_loader = DataLoader(dataset=val_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_data = MyDataset(test_data, transform=transform, loader=Myloader)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # train_loader, val_loader, test_loader = load_data()
    a = 1
