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

IN_CHANNELS = 560
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
    def __init__(self, model, optimizer, criterion):
        self.model = model.to(DEVICE)
        self.optimizer = optimizer
        self.criterion = criterion

    def set_file_list(self, train_prefix, test_prefix):
        self.train_file_list = glob.glob(train_prefix)
        self.test_file_list = glob.glob(test_prefix)

    def epoch2(self):
        train_files = self.train_file_list
        train_batch = BATCH_SIZE
        test_files = self.test_file_list
        test_batch = BATCH_SIZE_TEST
        epoch_train_total = 0
        file_batch = 10

        for i in range(len(train_files) // file_batch):
            print("iteration:", i)
            train_file = train_files[i*file_batch:i*file_batch+file_batch]
            train_file_data = FileDatasets2(train_file)
            train_data_loader = DataLoader(train_file_data, batch_size=train_batch, shuffle=False, drop_last=True)
            test_file = test_files
            test_file_data = FileDatasets2(test_file)
            test_data_loader = DataLoader(test_file_data, batch_size=test_batch, shuffle=False, drop_last=True)

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
                print("test:")
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

                print("train_total:", train_total, "train_loss:", train_loss * train_batch / train_total, "train_acc:", 100 * train_correct / train_total)
                print("test_total:", test_total, "test_loss:", test_loss * test_batch / test_total, "test_acc:", 100 * test_correct / test_total)

        return epoch_train_total

    def epoch(self, is_train):
        if is_train:
            self.model.train()
            file_list = self.train_file_list
            batch = BATCH_SIZE
        else:
            self.model.eval()
            file_list = self.test_file_list
            batch = BATCH_SIZE_TEST
        epoch_loss = 0
        correct = 0
        total = 0
        
        """
        for file_path in tqdm(file_list):
            file_data = FileDatasets(file_path)
            data_loader = DataLoader(file_data, batch_size=batch, shuffle=is_train, drop_last=False)

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

        file_data = FileDatasets2(file_list)
        data_loader = DataLoader(file_data, batch_size=batch, shuffle=is_train, drop_last=True)

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
            del loss
            if total % (batch * 500) == 0:
                print("total:", total, "loss:", epoch_loss / total, "acc:", 100 * correct / total)
        """

        for i in range(len(file_list) // 100):
            file_list2 = file_list[i*100:i*100+100]
            file_data = FileDatasets2(file_list2)
            data_loader = DataLoader(file_data, batch_size=batch, shuffle=is_train, drop_last=True)

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
                if (not is_train) and total % (batch * 200) == 0:
                    print("total:", total, "loss:", epoch_loss / total, "acc:", 100 * correct / total)

            if is_train and i%50 == 9:
                print("train_total:", total, "loss:", epoch_loss / total, "acc:", 100 * correct / total)

        epoch_loss /= total
        acc = 100 * correct / total
        return epoch_loss, acc, total

    def run_train(self, epoch_begin, epoch_end, test_interval, save_interval):
        #writer = SummaryWriter(log_dir="./logs")
        for i in range(epoch_begin, 1 + epoch_end):
            """
            train_loss, train_acc, train_total = self.epoch(True)
            print("train_loss:", train_loss, "train_acc:", train_acc, "train_total:", train_total)
            #writer.add_scalar("train_acc", train_acc, i)

            if i % test_interval == 0 and 0 < len(self.test_file_list):
                test_loss, test_acc, test_total = self.epoch(False)
                print("test_loss:", test_loss, "test_acc:", test_acc, "test_total", test_total)
                #writer.add_scalar("test_acc", test_acc, i)
            """
            epoch_train_total = self.epoch2()
            print("epoch", i, "epoch_train_total:", epoch_train_total)

            if i % save_interval == 0:
                state = {'epoch': i, 'model': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}
                torch.save(state, "train_tmp.pth")
        #writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', action='store_true')
    args = parser.parse_args()

    model = DiscardNet(IN_CHANNELS, MID_CHANNELS, BLOCKS_NUM)
    
    criterion = nn.CrossEntropyLoss()
    
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

    trainer = Trainer(model, optimizer, criterion)

    #test_prefix = "../akochan_ui/tenhou_npz/discard/2017/20171001/discard_201710010*.npz"
    test_prefix = "../akochan_ui/tenhou_npz/discard/2017/20171001/discard_2017100123*.npz"
    train_prefix = "../akochan_ui/tenhou_npz/discard/2017/20170*/discard_20170*.npz"
    trainer.set_file_list(train_prefix, test_prefix)

    print("train_files_num:", len(trainer.train_file_list))
    print("test_files_num:", len(trainer.test_file_list))
    print("train_batch_size:", BATCH_SIZE, "test_batch_size:", BATCH_SIZE_TEST)

    trainer.run_train(epoch_begin, 4, 1, 10)

