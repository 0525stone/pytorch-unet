"""
source from youtube : https://www.youtube.com/watch?v=1gMnChpUS9k&list=PLqtXapA2WDqbE6ghoiEJIrmEnndQ7ouys&index=6

data들을 보통 preprocess를 하고 prefix suffix로 구분을 주는게 통상적인 방법인듯

의문
    왜 numpy로 했을까???

"""
## 라이브러리 추가하기
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

## 데이터 로더 구현하기
# data들을 보통 preprocess를 하고 prefix suffix로 구분을 주는게 통상적인 방법인듯
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        lst_label.sort()
        lst_input.sort()

        self.lst_label = lst_label
        self.lst_input = lst_input

    def __len__(self):
        return len(self.lst_label)

    def __getitem__(self, index):  # index로 data를 load하는 방법
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_input[index]))

        # 데이터의 값들을 확인해서 알게 된 255로 나누어줌
        label = label/255.0
        input = input/255.0

        # x y c
        if label.ndim == 2:
            label = label[:,:,np.newaxis] # 새로운 axis 추가
        if input.ndim == 2:
            input = input[:,:,np.newaxis]

        data = {'input':input, 'label':label}

        if self.transform:
            data = self.transform(data)

        return data
# ##
## 트랜스폼 구현하기 기본적인 것들

# numpy  Y, X, CH
# tensor CH, Y, X
class ToTensor(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)

        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}

        return data

class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']

        input = (input - self.mean) / self.std

        data = {'label': label, 'input': input}

        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)

        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)

        data = {'label': label, 'input': input}

        return data

class Rotate90(object):
# class Rotate로 할 경우에는 __init__(self, angle) 필요하겠지만, 무조건 90도만 돌릴때는 굳이 필요 없음
    def __call__(self, data):
        label, input = data['label'], data['input']


   # 돌리고 나서 바로 return으로 값 빼줌
        data = {'label': label, 'input': input}

        return data

##


transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])

data_dir='./datasets/'
dataset_train = Dataset(data_dir=os.path.join(data_dir,'train'), transform = transform)

data = dataset_train.__getitem__(0)

input = data['input'] # 머지 설명이랑 좀 다른데... shape는 똑같은데 난 왜 나오지
label = data['label']

plt.subplot(121)
plt.imshow(label.squeeze())

plt.subplot(122)
plt.imshow(input.squeeze())

plt.show()

##




