import random
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
import torch_geometric
import os
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
# from .utils import remove_self_loops
from sklearn.model_selection import KFold
np.random.seed(12345)

class SocialBotDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None,dataset="vendor-purchased-2019"):
        self.datasets = {"twibot-20":0,"cresci-15":1,"mgtab":2,"mgtab-large":3}
        self.cur_dataset = dataset
        super(SocialBotDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices, self.split_index = torch.load(self.processed_paths[self.datasets[self.cur_dataset]])
        # label_list = list(range(len(self.data.y)))
        # train_index= label_list[:self.split_index[0]]
        # val_index = label_list[self.split_index[0]:self.split_index[0]+self.split_index[1]]
        # test_index = label_list[-self.split_index[-1]:]
        # assert len(train_index+val_index+test_index) == len(label_list)
        # self.train_index = index_to_mask(train_index, size=self.data.x.size(0))
        # self.val_index = index_to_mask(val_index, size=self.data.x.size(0))
        # self.test_index = index_to_mask(test_index, size=self.data.x.size(0))
        self.train_index = self.split_index[0]
        self.val_index = self.split_index[1]
        self.test_index = self.split_index[2]
        print(f'Train on {self.cur_dataset}')
        print(f'Number of edges: {self[0].edge_index.size(1)}')
        print(f'Number of nodes: {self[0].x.size(0)}')
        print(f'Number of labeled nodes: {len(self.data.y)}')
        print(f'Number of social bots: {sum(self.data.y)}')
        print(f'Number of training nodes: {sum(self.train_index)}')
        print(f'Number of test nodes: {sum(self.test_index)}')



    @property
    def raw_file_names(self):
        return ["twibot20.pickle","cresci15.pickle","mgtab.pickle","mgtab-large.pickle"]

    @property
    def processed_file_names(self):
        return ['twibot20.pt','cresci-15.pt','mgtab.pt',"mgtab-large.pt"]

    def download(self):
        pass

    def process(self):
        if os.path.exists(self.processed_paths[self.datasets[self.cur_dataset]]):
            return
        filepath = os.path.join("data","raw",self.raw_file_names[self.datasets[self.cur_dataset]])
        data_list = []
        if self.cur_dataset == "cresci-15":
            with open(filepath, 'rb') as f:
                features, edge_index, labels, [train_index,val_index,test_index] = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                features, edge_index, labels, split_index = pickle.load(f)
        # node_features = np.array(node_features)
        # if self.pre_transform is not None:
        #     features = node_features[:,0:300]
        # else:
        #     features = node_features
        # edge_index = np.array(G.edges).T
        # edge_index = remove_self_loops(edge_index)
        if self.cur_dataset != "cresci-15":
            label_list = list(range(len(labels)))
            train_index = label_list[:split_index[0]]
            val_index = label_list[split_index[0]:split_index[0] + split_index[1]]
            test_index = label_list[-split_index[-1]:]
        train_mask = index_to_mask(train_index, size=features.size(0))
        val_mask = index_to_mask(val_index, size=features.size(0))
        test_mask = index_to_mask(test_index, size=features.size(0))
        data_list.append(Data(x=features.float(), edge_index=edge_index, y=labels,train_mask=train_mask,val_mask=val_mask,test_mask=test_mask))
        data, slices = self.collate(data_list)
        torch.save((data, slices, [train_mask,val_mask,test_mask]), self.processed_paths[self.datasets[self.cur_dataset]])


def index_to_mask(index, size):
    mask = torch.zeros((size, ), dtype=torch.bool)
    mask[index] = True
    return mask

class DataDiffusion(Dataset):
    def __init__(self, data):
        self.data = data
    def __getitem__(self, index):
        item = self.data[index]
        return item
    def __len__(self):
        return len(self.data)