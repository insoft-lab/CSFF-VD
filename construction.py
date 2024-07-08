import pandas as pd
import torch
import csv
import os
from tqdm import tqdm
from torch_geometric.data import Data, DataLoader
import pickle

from torch_geometric.nn import TopKPooling, GCNConv, ResGatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from scipy.special import softmax

import csv
import numpy as np
import random
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import f1_score
import csv
from sklearn.preprocessing import normalize
import numpy as np
from models import GGNN_embedding


# define which dataset to use and the epoch of the parameter
# dataset : ffmpeg, qemu and reveal
dataset = "reveal"
epoch_of_parameter = 9
return_intermediate = True

if (dataset == "ffmpeg"):
    epoch_of_parameter = 2
elif (dataset == "qemu"):
    epoch_of_parameter = 11
elif (dataset == "reveal"):
    epoch_of_parameter = 95

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_list_path = f'./data-process/{dataset}_train_data.pkl'
val_list_path = f'./data-process/{dataset}_val_data.pkl'
test_list_path = f'./data-process/{dataset}_test_data.pkl'
output_csv_path = f'embeding/256-embeddings_{dataset}.csv'

# 加载数据
paths = [train_list_path, val_list_path, test_list_path]
split_names = ['train', 'val', 'test']  # 用于记录属于哪个数据集

# 准备模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GGNN_embedding().to(device)
model.load_state_dict(torch.load(
    f'models/best_model/best_f1_{dataset}_modelGCN-{epoch_of_parameter}.pth', map_location=device))
model.eval()

# 准备CSV文件
with open(output_csv_path, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    headers = ['name', 'label', 'split'] + \
        [f'vector{i + 1}' for i in range(256)]
    csvwriter.writerow(headers)

    # 遍历每个数据集
    for path, split_name in zip(paths, split_names):
        with open(path, 'rb') as f:
            data_list = pickle.load(f)

        # 处理每个数据项
        for data in tqdm(data_list, desc=f"Processing {split_name}"):
            name = data.file_name
            label = data.y.item() if torch.is_tensor(data.y) else data.y

            # 确保数据在正确的设备上
            data.to(device)

            # 获取嵌入向量
            with torch.no_grad():
                embedding = model(data, return_intermediate=True)

            # 将嵌入向量从GPU转移到CPU，并转换为列表
            embedding_list = embedding.cpu().view(-1).tolist()

            # 写入CSV文件
            csvwriter.writerow([name, label, split_name] + embedding_list)


input_csv_path = f'embeding/256-embeddings_{dataset}.csv'  # 假设这是之前保存的CSV文件的路径
embeddings = []
rows = []
#
#
#
with open(input_csv_path, 'r', newline='') as csvfile:
    csvreader = csv.reader(csvfile)
    next(csvreader)  # 跳过表头
    for row in csvreader:
        # 假设嵌入向量从第4列开始，到倒数第二列结束
        embedding = [float(value) for value in row[3:-1]]
        embeddings.append(embedding)
        # 保存整行数据以便后续使用
        rows.append(row)
