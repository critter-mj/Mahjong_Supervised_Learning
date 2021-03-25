import glob
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import argparse

from util import *

#MID_CHANNELS = 256
MID_CHANNELS = 128
#BLOCKS_NUM = 50
BLOCKS_NUM = 15
BATCH_SIZE = 64
BATCH_SIZE_TEST = 32
#LEARNING_RATE = 0.000005
LEARNING_RATE = 0.0001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class Trainer:
    def __init__(self, model, optimizer, criterion, file_batch):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion
        self.file_batch = file_batch

        self.epoch_cnt = 0

    def set_file_list(self, train_prefix, test_prefix):
        self.train_file_list = glob.glob(train_prefix)
        self.test_file_list = glob.glob(test_prefix)

    def test_epoch(self, test_data_loader):
        self.model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0

        for inputs, targets in tqdm(test_data_loader):
            inputs = torch.unsqueeze(inputs.to(DEVICE).float(), -1)
            targets = targets.to(DEVICE).long()

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_correct += predicted.eq(targets.data).cpu().sum().item()
            test_total += targets.size(0)

        print("test_total:", test_total, "test_loss:", test_loss * BATCH_SIZE_TEST / test_total, "test_acc:", 100 * test_correct / test_total)


    def train_epoch(self):
        epoch_train_total = 0

        test_file_data = FileDatasets2(self.test_file_list)
        test_data_loader = DataLoader(test_file_data, batch_size=BATCH_SIZE_TEST, shuffle=False, drop_last=True)

        for i in range(len(self.train_file_list) // self.file_batch):
            print("iteration:", i)
            train_file = self.train_file_list[i*self.file_batch:i*self.file_batch+self.file_batch]
            train_file_data = FileDatasets2(train_file)
            train_data_loader = DataLoader(train_file_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for inputs, targets in tqdm(train_data_loader):
                inputs = torch.unsqueeze(inputs.to(DEVICE).float(), -1)
                targets = targets.to(DEVICE).long()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += predicted.eq(targets.data).cpu().sum().item()
                train_total += targets.size(0)
                epoch_train_total += targets.size(0)

            if i % 40 == 0:
                print("train_total:", train_total, "train_loss:", train_loss * BATCH_SIZE / train_total, "train_acc:", 100 * train_correct / train_total)
                
                print("test:")
                self.test_epoch(test_data_loader)

                state = {'epoch': self.epoch_cnt, 'iteration': i, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                torch.save(state, "train_tmp.pth")

        return epoch_train_total

    def run_train(self, epoch_begin, epoch_end, test_interval):
        for _i in range(epoch_begin, 1 + epoch_end):
            epoch_train_total = self.train_epoch()
            self.epoch_cnt += 1
            print("epoch", self.epoch_cnt, "epoch_train_total:", epoch_train_total)

def main_func(args):
    if args.action_type == 'dahai':
        IN_CHANNELS = 560
        FILE_BATCH_SIZE = 10
        model = DiscardNet(IN_CHANNELS, MID_CHANNELS, BLOCKS_NUM)
        #test_prefix = "../akochan_ui/tenhou_npz/discard/2017/20171001/discard_201710010*.npz"
        test_prefix = "../akochan_ui/tenhou_npz/discard/2017/20171001/discard_2017100123*.npz"
        train_prefix = "../akochan_ui/tenhou_npz/discard/2017/20170*/discard_20170*.npz"
    elif args.action_type == 'kan':
        IN_CHANNELS = 567
        FILE_BATCH_SIZE = 100
        model = FuuroNet(IN_CHANNELS, MID_CHANNELS, BLOCKS_NUM)
        test_prefix = "../akochan_ui/tenhou_npz/kan/2017/20171001/kan_20171001*.npz"
        train_prefix = "../akochan_ui/tenhou_npz/kan/2017/20170*/kan_20170*.npz"
        #train_prefix = "../akochan_ui/tenhou_npz/kan/2017/20170101/kan_20170101*.npz"

    criterion = nn.CrossEntropyLoss()

    if args.purpose == 'dump_cpu_model':
        last_state = torch.load('train_tmp.pth')
        model.load_state_dict(last_state['model'])
        model = model.to('cpu')
        torch.save(model.state_dict(), args.action_type + '_cpu_state_dict.pth')
        return
    elif args.purpose == 'test':
        model.load_state_dict(torch.load(args.action_type + '_cpu_state_dict.pth'))
        trainer = Trainer(model, None, criterion, None)
        trainer.set_file_list(train_prefix, test_prefix)

        test_file_data = FileDatasets2(trainer.test_file_list)
        test_data_loader = DataLoader(test_file_data, batch_size=BATCH_SIZE_TEST, shuffle=False, drop_last=True)
        trainer.test_epoch(test_data_loader)
        return    
    
    if args.load:
        tmp = torch.load("train_tmp.pth")
        model.load_state_dict(tmp['model'])
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        optimizer.load_state_dict(tmp['optimizer'])
        if DEVICE == 'cuda':
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()
        epoch_begin = tmp['epoch'] + 1
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, eps=0.0000001)
        epoch_begin = 0

    trainer = Trainer(model, optimizer, criterion, FILE_BATCH_SIZE)
    trainer.set_file_list(train_prefix, test_prefix)

    print("train_files_num:", len(trainer.train_file_list))
    print("test_files_num:", len(trainer.test_file_list))
    print("train_batch_size:", BATCH_SIZE, "test_batch_size:", BATCH_SIZE_TEST)

    trainer.run_train(epoch_begin, 4, 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--purpose', choices=('train', 'test', 'dump_cpu_model'), default='train')
    parser.add_argument('--action_type', choices=('dahai', 'kan'))
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    main_func(args)
