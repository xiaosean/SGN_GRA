from torch.utils.data import Dataset, DataLoader
import os
import torch
import numpy as np
import h5py
import random
import os.path as osp
import sys
from six.moves import xrange
import math
import scipy.misc
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import matplotlib.pyplot as plt

from transform import spatial_transform, temporal_transform, jm_SSL, fm_SSL
from collections import Counter

class NTUDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = np.array(y, dtype='int')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        
        return [self.x[index], int(self.y[index])]

class NTUDataLoaders(object):
    def __init__(self, dataset ='NTU', case = "CS", seg = 20, data_volume = 100):
        """
        # Input:
        # dataset:      Name of the dataset. (NTU RGB+D: NTU, N-UCLA: NUCLA)
        # case:         CS for Cross-Subject, CV for Cross-View. NUCLA default CS. 
        # seg:          Number of frames for each training data.
        # data_volume:  The percentage of the trainig data for semi-supervised learning.
        """
        self.dataset = dataset
        self.case = case
        self.seg = seg
        self.data_volume = data_volume

        # 2021-03-05 Ken 增加切割 Subset 的 func
        self.create_datasets(data_volume)
        self.get_tsne_subset()
        
        self.train_set = NTUDataset(self.train_X, self.train_Y)
        self.test_set = NTUDataset(self.test_X, self.test_Y)
        self.train_subset = NTUDataset(self.train_subX, self.train_subY)
        self.tsne_set = NTUDataset(self.tsne_X, self.tsne_Y)

    def get_train_loader(self, batch_size, num_workers):
        return DataLoader(self.train_set, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            collate_fn=self.collate_fn_fix_train, pin_memory=True, drop_last=True)

    def get_test_loader(self, batch_size, num_workers, situation=1):
        if situation == 1 or situation == 0:
            return DataLoader(self.test_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=self.collate_fn_fix_test, pin_memory=True, drop_last=True)
        if situation == 2:
            return DataLoader(self.test_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=self.collate_fn_fix_test_jm, pin_memory=True, drop_last=True)
        if situation == 3:
            return DataLoader(self.test_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=self.collate_fn_fix_test_fm, pin_memory=True, drop_last=True)

    def get_train_subloader(self, batch_size, num_workers):
        return DataLoader(self.train_subset, batch_size=batch_size,
                            shuffle=True, num_workers=num_workers,
                            collate_fn=self.collate_fn_fix_train_semi, pin_memory=True, drop_last=True)

    def get_tsne_loader(self, batch_size, num_workers):
        return DataLoader(self.tsne_set, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=self.collate_fn_fix_test, pin_memory=True, drop_last=True)

    def get_train_size(self):
        return len(self.train_Y)

    def get_test_size(self):
        return len(self.test_Y)

    def get_train_subsize(self):
        return len(self.train_subY)

    def get_tsne_size(self):
        return len(self.tsne_Y)

    def get_tsne_subset(self):
        """
        # Create subset(tsne_X, tsne_Y) for visualization of T-SNE.
        """
        tsne_filepath = './data/ntu/NTU_tsne.h5'
        if os.path.exists(tsne_filepath):
            f = h5py.File(tsne_filepath , 'r')
            self.tsne_X =  f['tsne_X'][:]
            self.tsne_Y =  f['tsne_Y'][:]
            return None

        temp_x = []
        temp_y = []
        sm = 0
        for i in range(0, 60, 10):
            temp_x.append(self.train_X[np.where(self.train_Y == i)])
            temp_y.append(self.train_Y[np.where(self.train_Y == i)])
            sm+=np.where(self.train_Y == i,1,0).sum()

        self.tsne_X = [item for sublist in temp_x for item in sublist]
        self.tsne_Y = [item for sublist in temp_y for item in sublist]
        
        with h5py.File(tsne_filepath, 'w') as hf:
            hf.create_dataset('tsne_X',  data=self.tsne_X)
            hf.create_dataset('tsne_Y',  data=self.tsne_Y)
        

    # 20210305 Ken 增加切割 Subset 的 func
    def get_subset(self, percent):
        """
        # Create traininig subset. Load the trianing subset if exist .h file. Otherwise, Create .h file. 
        # Input:
        #   percent:    The percentage of the training set for semi-supervised learning.
        """
        if self.dataset == "NTU":
            if percent == 5 or percent == 10 or percent == 20 or percent == 40:
                # 2021-03-07 Ken 判斷是否已有 Subset .h5 file
                subset_filepath = './data/ntu/NTU_' + self.metric + '_subset_' + str(percent) + '_percent.h5'
                if os.path.exists(subset_filepath):
                    f = h5py.File(subset_filepath , 'r')
                    return f['train_x'][:], f['train_y'][:]
                else:
                    if self.metric == "CS":
                        if percent == 5:
                            class_volume_limitation = 33
                        elif percent == 10:
                            class_volume_limitation = 66
                        elif percent == 20:
                            class_volume_limitation = 132
                        elif percent == 40:
                            class_volume_limitation = 264
                    
                    if self.metric == "CV":
                        if percent == 5:
                            class_volume_limitation = 31
                        elif percent == 10:
                            class_volume_limitation = 62
                        elif percent == 20:
                            class_volume_limitation = 124
                        elif percent == 40:
                            class_volume_limitation = 248

                    class_num = 60
                    subset_volume = class_num * class_volume_limitation
                    # print("subset_volume: ", subset_volume)

                    self.extract_x = []
                    self.extract_y = []
                    skip_list = []
                    index = 0
                    for label in self.train_Y[:]:
                        index += 1

                        # 計算已經抽取出的 data 數量，用於顯示進度、設定中止條件
                        extracted_num = len(self.extract_y)
                        print("Extracting a Subset ... {}%\r".format(
                            round((extracted_num/subset_volume)*100, 2)
                        ), end="")#, flush=True

                        # 每挑過一個數字便檢核數量是否足夠
                        c = Counter(self.extract_y)
                        for check_label in range(class_num):
                            if c[check_label] == class_volume_limitation:
                                skip_list.append(check_label)

                        if label in skip_list: # 數量已經足夠的跳過
                            continue
                        else:
                            self.extract_x.append(self.train_X[index-1])
                            self.extract_y.append(label)

                        if extracted_num == subset_volume:
                            break
                    print("\n")
                    self.extract_x = np.array(self.extract_x)
                    self.extract_y = np.array(self.extract_y)
                    # 2021-03-07 Ken 第一次切完 Subset 之後存檔，後續依照相同路徑直接讀取 .h5 file
                    with h5py.File(subset_filepath, 'w') as hf:
                        hf.create_dataset('train_x',  data=self.extract_x)
                        hf.create_dataset('train_y',  data=self.extract_y)
                    return self.extract_x, self.extract_y
            else:
                print("Please input 5, 10, 20, 40(%)")
                return None, None
        
        if self.dataset == "NUCLA":
            if percent == 5 or percent == 15 or percent == 30 or percent == 40:
                subset_filepath = './data/N-UCLA/NUCLA' + '_subset_' + str(percent) + '_percent.h5'
                if os.path.exists(subset_filepath):
                    f = h5py.File(subset_filepath , 'r')
                    return f['train_x'][:], f['train_y'][:]
                else:              
                    if percent == 5:
                        class_volume_limitation = 5
                    elif percent == 15:
                        class_volume_limitation = 15
                    elif percent == 30:
                        class_volume_limitation = 30
                    elif percent == 40:
                        class_volume_limitation = 40

                    class_num = 10
                    subset_volume = class_num * class_volume_limitation

                    self.extract_x = []
                    self.extract_y = []
                    skip_list = []
                    index = 0
                    print(self.train_Y)
                    for label in self.train_Y[:]:
                        index += 1

                        # 計算已經抽取出的 data 數量，用於顯示進度、設定中止條件
                        extracted_num = len(self.extract_y)
                        print("Extracting a Subset ... {}%\r".format(
                            round((extracted_num/subset_volume)*100, 2)
                        ), end="")#, flush=True

                        # 每挑過一個數字便檢核數量是否足夠
                        c = Counter(self.extract_y)
                        for check_label in range(class_num):
                            if c[check_label] == class_volume_limitation:
                                skip_list.append(check_label)

                        if label in skip_list: # 數量已經足夠的跳過
                            continue
                        else:
                            self.extract_x.append(self.train_X[index-1])
                            self.extract_y.append(label)

                        if extracted_num == subset_volume:
                            break
                    print("\n")
                    self.extract_x = np.array(self.extract_x)
                    self.extract_y = np.array(self.extract_y)
                    # 2021-03-07 Ken 第一次切完 Subset 之後存檔，後續依照相同路徑直接讀取 .h5 file
                    with h5py.File(subset_filepath, 'w') as hf:
                        hf.create_dataset('train_x',  data=self.extract_x)
                        hf.create_dataset('train_y',  data=self.extract_y)
                    return self.extract_x, self.extract_y
            else:
                print("Please input 5, 10, 20, 40(%)")
                return None, None
                    
    def create_datasets(self, percent):
        """
        # Initial training set(train_X, train_Y), testing set(test_X, test_Y), training subset(train_subX, train_subY).
        # Input: 
        #   percent:     The percentage of the training set for semi-supervised learning.
        """
        if self.dataset == 'NTU':
            if self.case =="CS":
                self.metric = 'CS'
            elif self.case == "CV":
                self.metric = 'CV'
            path = osp.join('./data/ntu', 'NTU_' + self.metric + '.h5')

            f = h5py.File(path , 'r')
            self.train_X = f['x'][:]
            self.train_Y = np.argmax(f['y'][:],-1)
            self.val_X = f['valid_x'][:]
            self.val_Y = np.argmax(f['valid_y'][:], -1)
            self.test_X = f['test_x'][:]
            self.test_Y = np.argmax(f['test_y'][:], -1)
            f.close()

            ## Combine the training data and validation data togehter as ST-GCN
            self.train_X = np.concatenate([self.train_X, self.val_X], axis=0)
            self.train_Y = np.concatenate([self.train_Y, self.val_Y], axis=0)
            # print(self.train_X)
            # print(self.train_Y)
            self.val_X = self.test_X
            self.val_Y = self.test_Y

            # self.get_subset(percent)
            path_p = osp.join('./data/ntu', 'NTU_' + self.metric + f'_subset_{percent}_percent.h5')
            
            # Get percent% data
            f = h5py.File(path_p , 'r')
            self.train_subX = f['train_x'][:]
            self.train_subY = f['train_y'][:]
            f.close()
        
        if self.dataset == "NUCLA":
            path = osp.join('./data/N-UCLA', 'NUCLA.h5')
            f = h5py.File(path , 'r')
            self.train_X = f['x'][:]
            self.train_Y = f['y'][:]
            self.test_X = f['test_x'][:]
            self.test_Y = f['test_y'][:]
            f.close()
            self.val_X = self.test_X
            self.val_Y = self.test_Y

            self.get_subset(percent)
            path_p = osp.join('./data/N-UCLA', f'NUCLA_subset_{percent}_percent.h5')
            
            # Get percent% data
            f = h5py.File(path_p , 'r')
            self.train_subX = f['train_x'][:]
            self.train_subY = f['train_y'][:]
            f.close()

    def collate_fn_fix_train(self, batch):
        x, y = zip(*batch)

        x, y = self.Tolist_fix(x, y, train=1)
        lens = np.array([x_.shape[0] for x_ in x], dtype=np.int)
        idx = lens.argsort()[::-1]  # sort sequence by valid length in descending order
        y = np.array(y)[idx]
        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        transform_x = x.clone()

        x1, gt1 = jm_SSL(x)
        x2, gt2 = fm_SSL(x)

        y = torch.LongTensor(y)         

        return [x, y, x1, gt1, x2, gt2]
    
    def collate_fn_fix_train_semi(self, batch):
        x, y = zip(*batch)

        x, y = self.Tolist_fix(x, y, train=1)
        lens = np.array([x_.shape[0] for x_ in x], dtype=np.int)
        idx = lens.argsort()[::-1]  # sort sequence by valid length in descending order
        y = np.array(y)[idx]
        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        transform_x = x.clone()

        x1, gt1 = jm_SSL(x)
        x2, gt2 = fm_SSL(x)
        x = spatial_transform(x, ["rot"])

        y = torch.LongTensor(y)         

        return [x, y, x1, gt1, x2, gt2]

    def collate_fn_fix_test_jm(self, batch):
        x, y = zip(*batch)
        x, labels = self.Tolist_fix(x, y ,train=1)
        idx = range(len(x))
        y = np.array(y)

        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        y = torch.LongTensor(y)

        x, gt1 = jm_SSL(x)
        #x, gt2 = fm_SSL(x)

        return [x, y]

    def collate_fn_fix_test_fm(self, batch):
        x, y = zip(*batch)
        x, labels = self.Tolist_fix(x, y ,train=1)
        idx = range(len(x))
        y = np.array(y)

        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        y = torch.LongTensor(y)

        #x, gt1 = jm_SSL(x)
        x, gt2 = fm_SSL(x)

        return [x, y]
    
    def collate_fn_fix_test(self, batch):
        x, y = zip(*batch)
        x, labels = self.Tolist_fix(x, y ,train=1)
        idx = range(len(x))
        y = np.array(y)

        x = torch.stack([torch.from_numpy(x[i]) for i in idx], 0)
        y = torch.LongTensor(y)

        #x, gt1 = jm_SSL(x)
        #x, gt2 = fm_SSL(x)

        return [x, y]


    def Tolist_fix(self, joints, y, train = 1):
        seqs = []
        flip = np.flip(joints, 1)
        for idx, seq in enumerate(joints):
            zero_row = []
            for i in range(len(seq)):
                if self.dataset == "NTU":
                    if (seq[i, :] == np.zeros((1, 150))).all():
                            zero_row.append(i)
                elif self.dataset == "NUCLA":
                    if (seq[i, :] == np.zeros((1, 60))).all():
                            zero_row.append(i)

            seq = np.delete(seq, zero_row, axis = 0)
            
            if self.dataset == "NTU":
                seq = turn_two_to_one(seq)
            seqs = self.sub_seq(seqs, seq, train=train)

        return seqs, y

    def sub_seq(self, seqs, seq , train = 1):
        group = self.seg

        if self.dataset == 'SYSU' or self.dataset == 'SYSU_same':
            seq = seq[::2, :]

        if seq.shape[0] < self.seg:
            pad = np.zeros((self.seg - seq.shape[0], seq.shape[1])).astype(np.float32)
            seq = np.concatenate([seq, pad], axis=0)

        ave_duration = seq.shape[0] // group

        if train == 1:
            offsets = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            seq = seq[offsets]
            seqs.append(seq)

        elif train == 2:
            offsets1 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets2 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets3 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets4 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)
            offsets5 = np.multiply(list(range(group)), ave_duration) + np.random.randint(ave_duration, size=group)

            seqs.append(seq[offsets1])
            seqs.append(seq[offsets2])
            seqs.append(seq[offsets3])
            seqs.append(seq[offsets4])
            seqs.append(seq[offsets5])

        return seqs

"""
class AverageMeter(object):
    #Computes and stores the average and current value
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
"""


def turn_two_to_one(seq):
    new_seq = list()

    for idx, ske in enumerate(seq):
        if (ske[0:75] == np.zeros((1, 75))).all():
            new_seq.append(ske[75:])
        elif (ske[75:] == np.zeros((1, 75))).all():
            new_seq.append(ske[0:75])
        else:
            new_seq.append(ske[0:75])
            new_seq.append(ske[75:])
    return np.array(new_seq)

class PseudoDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        
        return [self.x[index], self.y[index]]
    
def collate_fn_ssl(batch):
    x, y = zip(*batch)
    x = torch.stack(list(x), dim=0)

    x1, gt1 = jm_SSL(x)
    x2, gt2 = fm_SSL(x)
    x = spatial_transform(x, ["rot"])

    y = torch.LongTensor(y)         

    return [x, y, x1, gt1, x2, gt2]