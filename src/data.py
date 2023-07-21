import pandas as pd
import numpy as np
import ipdb
import tqdm
import socket
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sklearn
import os


class StockPrice(Dataset):
    """
    stock price dataset, only for 20 days interval data.
    For training, each time we get a stock of a given day
    For validation or test, each time we get all stocks of a given day
    """
    def __init__(self, index, mode="train", train_path="/data2/tianyu/factor_group_590_v1_t20", label_path="/data203/tianyu/d2r"):
        self.mode = mode
        self.train_path = train_path
        self.label_path = label_path

        # process data and corresponding label
        name_list = sorted(os.listdir(train_path))
        if isinstance(index, list):
            name_list = [name_list[i] for i in index]
        elif isinstance(index, slice):
            name_list = name_list[index]
        else:
            raise ValueError(f"Not Support Index Type:{type(index)}")

        label_list = []
        for day in name_list:
            label_list.append(pd.read_csv(os.path.join(label_path, day), header=None))
        
        self.data = name_list
        self.label = label_list

        pre_path = os.path.dirname(label_path)
        
        if self.mode == "train":
            if os.path.exists(os.path.join(pre_path, f"{socket.gethostname()}_stock_data.npy")):
                stock_data = np.load(os.path.join(pre_path, f"{socket.gethostname()}_stock_data.npy")).tolist()
                stock_data = list(map(lambda x: (x[0], float(x[1])), stock_data))
            else:
                stock_data = []
                for i, day in tqdm.tqdm(enumerate(name_list), total=len(name_list), desc="Preprocssing..."):
                    label = label_list[i]
                    for j in range(label.shape[0]):
                        stock_name = str(label.iloc[j, 0]).rjust(6, "0")+".npy"
                        stock_path = os.path.join(train_path, day, stock_name)
                        if os.path.exists(stock_path):
                            stock_label = label.iloc[j, 1]
                            stock_data.append((stock_path, stock_label))
                        else:
                            continue
                np.save(os.path.join(pre_path, f"{socket.gethostname()}_stock_data.npy"), stock_data)
            random.shuffle(stock_data)
            self.stock_data = stock_data
    
    def __getitem__(self, idx):
        if self.mode != "train":
            # get all stock data of a specific day
            day = self.data[idx]
            label = self.label[idx]
            # get all stocks id from file "codelist"
            all_stocks = pd.read_csv(os.path.join(self.train_path, day, "codelist"), header=None).iloc[:, 0]
            # only select stocks which appear in both training and inference
            all_stocks = pd.merge(all_stocks, label)

            all_stocks_name = all_stocks[0].apply(lambda x: str(x).rjust(6, "0")+".npy")
            all_stocks_data = np.stack([np.load(os.path.join(self.train_path, day, stock)) for stock in all_stocks_name])
            
            return torch.from_numpy(all_stocks_data), torch.tensor(all_stocks[1]) # , all_stocks[0].tolist()
        
        else:
            # get single stock data of a specific day
            stock_path, stock_label = self.stock_data[idx]
            return torch.from_numpy(np.load(stock_path)), torch.tensor(stock_label)

    def __len__(self):
        if self.mode != "train":
            return len(self.data)
        else:
            return len(self.stock_data)

    def collate_fn(self, batch):
        if self.mode != "train":
            batch_stock_data = torch.cat([x[0] for x in batch])
            batch_stock_label = torch.cat([x[1] for x in batch])
        else:
            batch_stock_data = torch.stack([x[0] for x in batch])
            batch_stock_label = torch.stack([x[1] for x in batch])
        return batch_stock_data, batch_stock_label
            
