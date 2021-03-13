import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

""" network model """

"""
def get_tmp_size(models, input_size):
    tmp = (32, input_size[0], input_size[1], input_size[2])
    x = torch.zeros(tmp)
    x = models.forward(x)
    return tuple([x.size()[i] for i in range(1, x.dim())])
"""

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0), bias=False),
            #nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=(3,1), padding=(1,0), bias=False),
            #nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        x = self.sequential(x) + x
        return F.relu(x)

class DiscardNet(nn.Module):
    def __init__(self, in_channels, channels_num, blocks_num):
        super(DiscardNet, self).__init__()
        self.preproc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels_num, kernel_size=(3,1), padding=(1,0), bias=False),
            #nn.BatchNorm2d(self.channels[0]),
            nn.ReLU()
        )

        blocks = []
        for _i in range(blocks_num):
            blocks.append(ResBlock(channels_num))
        self.res_blocks = nn.Sequential(*blocks)

        self.postproc = nn.Sequential(
            nn.Conv2d(in_channels=channels_num, out_channels=1, kernel_size=(1,1), padding=(0,0), bias=False),
            #nn.ReLU()
        )

    def forward(self, x):
        x = self.preproc(x)
        x = self.res_blocks(x)
        x = self.postproc(x)
        x = x.view(x.size(0), -1)  # [B, C, H, W] -> [B, C*H*W]
        #print(x)
        return x
        #return F.log_softmax(x)

class PonNet(nn.Module):
    def __init__(self, in_channels, channels_num, blocks_num):
        super(PonNet, self).__init__()
        self.preproc = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels_num, kernel_size=(3,1), padding=(1,0), bias=False),
            #nn.BatchNorm2d(self.channels[0]),
            nn.ReLU()
        )

        blocks = []
        for _i in range(blocks_num):
            blocks.append(ResBlock(channels_num))
        self.res_blocks = nn.Sequential(*blocks)

        self.postproc = nn.Sequential(
            nn.Conv2d(in_channels=channels_num, out_channels=32, kernel_size=(3,1), padding=(1,0), bias=False),
            #nn.ReLU()
        )

        self.dence = nn.Sequential(
            #nn.Linear(1024,256),
            nn.Linear(1088,256),
            nn.Linear(256,2)
        )

    def forward(self, x):
        x = self.preproc(x)
        x = self.res_blocks(x)
        x = self.postproc(x)
        x = x.view(x.size(0), -1)
        x = self.dence(x)
        return x
        #return F.log_softmax(x)

""" dataset """

class FileDatasets(Dataset):
    def __init__(self, file_path):
        npz = np.load(file_path)
        self.data = npz['arr_0']
        self.label = npz['arr_1']

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], np.argmax(self.label[idx])

class FileDatasets2(Dataset):
    def __init__(self, file_list):
        self.data = []
        self.label = []
        for file_path in file_list:
            npz = np.load(file_path)
            self.data.extend(npz['arr_0'])
            self.label.extend(npz['arr_1'])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.data[idx], np.argmax(self.label[idx])

class Hai34Transform():
    def __init__(self):
        pass

    def swap_color(self, x, c1, c2, c3):
        for i in range(9):
            x[[i, 9+i, 18+i]] = x[[c1*9+i, c2*9+i, c3*9+i]]

    def swap_sangen(self, x, c1, c2, c3):
        pos = 31
        x[[pos, pos+1, pos+2]] = x[[pos+c1, pos+c2, pos+c3]]

    def reverse_num(self, x):
        for i in range(4):
            x[[i, 8-i, 9+i, 17-i, 18+i, 26-i]] = x[[8-i, i, 17-i, 9+i, 26-i, 18+i]]

    def random_trans(self, x):
        i_color = np.random.randint(6)
        i_sangen = np.random.randint(6)
        i_rev = np.random.randint(2)

        if i_color == 1:
            self.swap_color(x, 0, 2, 1)
        elif i_color == 2:
            self.swap_color(x, 1, 0, 2)
        elif i_color == 3:
            self.swap_color(x, 1, 2, 0)
        elif i_color == 4:
            self.swap_color(x, 2, 0, 1)
        elif i_color == 5:
            self.swap_color(x, 2, 1, 0)

        if i_sangen == 1:
            self.swap_sangen(x, 0, 2, 1)
        elif i_sangen == 2:
            self.swap_sangen(x, 1, 0, 2)
        elif i_sangen == 3:
            self.swap_sangen(x, 1, 2, 0)
        elif i_sangen == 4:
            self.swap_sangen(x, 2, 0, 1)
        elif i_sangen == 5:
            self.swap_sangen(x, 2, 1, 0)

        if i_rev == 1:
            self.reverse_num(x)

class FileDatasetsAug(Dataset):
    def __init__(self, file_path):
        npz = np.load(file_path)
        self.data = npz['arr_0']
        self.label = npz['arr_1']
        self.trans = Hai34Transform()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        tmp = np.vstack((self.data[idx], self.label[idx]))
        tmp = np.transpose(tmp)
        self.trans.random_trans(tmp)
        tmp = np.transpose(tmp)

        return tmp[:-1], np.argmax(tmp[-1])
