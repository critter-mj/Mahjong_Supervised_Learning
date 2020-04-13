import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse

from util import *

IN_CHANNELS = 240
#MID_CHANNELS = 256
MID_CHANNELS = 32
BLOCKS_NUM = 10
BATCH_SIZE = 32
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    def __init__(self, model, optimizer, criterion):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion

    def set_file_list(self, train_prefix, test_prefix):
        self.train_file_list = glob.glob(train_prefix)
        self.test_file_list = glob.glob(test_prefix)

    def epoch(self, is_train):
        if is_train:
            self.model.train()
            file_list = self.train_file_list
        else:
            self.model.eval()
            file_list = self.test_file_list
        epoch_loss = 0
        correct = 0
        total = 0

        for file_path in tqdm(file_list):
            file_data = FileDatasetsAug(file_path)
            data_loader = DataLoader(file_data, batch_size=BATCH_SIZE, shuffle=is_train, drop_last=is_train)

            for inputs, targets in data_loader:
                inputs = torch.unsqueeze(inputs.to(DEVICE).float(), -1)
                targets = targets.to(DEVICE).long()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                if is_train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                correct += predicted.eq(targets.data).cpu().sum().item()
                total += targets.size(0)

        epoch_loss /= total
        acc = 100 * correct / total
        return epoch_loss, acc

    def run_train(self, epoch_begin, epoch_end, test_interval, save_interval):
        writer = SummaryWriter(log_dir="./logs")
        for i in tqdm(range(epoch_begin, 1 + epoch_end)):
            train_loss, train_acc = self.epoch(True)
            print("train_loss:", train_loss, "train_acc:", train_acc)
            writer.add_scalar("train_acc", train_acc, i)

            if i % test_interval == 0 and 0 < len(self.test_file_list):
                test_loss, test_acc = self.epoch(False)
                print("test_loss:", test_loss, "test_acc:", test_acc)
                writer.add_scalar("test_acc", test_acc, i)

            if i % save_interval == 0:
                state = {'epoch': i, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                torch.save(state, "train_tmp.pth")
        writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    model = DiscardNet(IN_CHANNELS, MID_CHANNELS, BLOCKS_NUM)
    
    criterion = nn.CrossEntropyLoss()
    
    
    if args.load:
        tmp = torch.load("train_tmp.pth")
        model.load_state_dict(tmp['model'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(tmp['optimizer'])
        if DEVICE == 'cuda':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        epoch_begin = tmp['epoch'] + 1
    else:
        optimizer = torch.optim.Adam(model.parameters())
        epoch_begin = 0

    trainer = Trainer(model, optimizer, criterion)

    train_prefix = "../akochan_ui/tenhou_npz/discard/2018/20180101/discard_2018010100*.npz"
    #train_prefix = "tenhou_npz/discard/2018/20180101/discard_*.npz"
    test_prefix = "../akochan_ui/tenhou_npz/discard/2018/20180102/discard_2018010200*.npz"
    #test_prefix = "tenhou_npz/discard/2018/20180102/discard_*.npz"
    trainer.set_file_list(train_prefix, test_prefix)

    print(trainer.test_file_list)

    trainer.run_train(epoch_begin, 200, 1, 10)

