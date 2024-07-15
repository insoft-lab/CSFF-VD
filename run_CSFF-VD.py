import itertools
from sklearn.model_selection import RepeatedKFold
from torch_geometric.data import Data
from models import SuperGAT
from util.common_utils import *
import copy
import os
# from dhg.models import CSFF
import torch.optim as optim
from config import get_config
from collections import defaultdict
import csv
import torch
from sklearn.preprocessing import normalize
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score

# load configuration
cfg = get_config('config/config_CSFF-VD.yaml')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set random seed , The random seed can be adjusted according to the experimental requirements

seed = 321
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def HG_supervised_embedding(X, y, train_index, test_index, G):
    seed = 321
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # transform data to device
    X = torch.Tensor(X).to(device)
    label_encoder = LabelEncoder()
    y_int = label_encoder.fit_transform(y)

    # 然后将整数标签数组转换为 torch.Tensor
    y = torch.tensor(y_int).long().to(device)
    # y = torch.Tensor(y).squeeze().long().to(device)
    train_index = torch.Tensor(train_index).long().to(device)
    test_index = torch.Tensor(test_index).long().to(device)
    G = G.to(device)

    # model initialization
    CSFF_model = SuperGAT(
        in_ch=X.shape[1], n_hid=cfg['n_hid'], dropout=cfg['drop_out'])
    # CSFF_model = MLP(in_ch=X.shape[1],n_hid=cfg['n_hid'],dropout=cfg['drop_out'])
    CSFF_model = CSFF_model.to(device)

    optimizer = optim.Adam(CSFF_model.parameters(),
                           lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg['milestones'], gamma=cfg['gamma'])
    criterion = torch.nn.CrossEntropyLoss()

    # model training
    best_f1 = 0.0  # 初始化最佳F1分数
    best_model_wts = copy.deepcopy(CSFF_model.state_dict())
    for epoch in range(cfg['max_epoch']):
        CSFF_model.train()
        optimizer.zero_grad()
        outputs = CSFF_model(X, G)
        loss = criterion(outputs[train_index], y[train_index])
        loss.backward()
        optimizer.step()

        # 每个epoch更新学习率
        scheduler.step()

        # print(f'Epoch {epoch + 1}/{cfg["max_epoch"]}, Loss: {loss.item()}')
        # 模型评估

    CSFF_model.eval()
    with torch.no_grad():
        outputs = CSFF_model(X, G)
        probs = torch.softmax(outputs[test_index], dim=1)[
            :, 1].cpu().numpy()  # 获取正类的概率值
        preds = torch.max(outputs[test_index], 1)[1].cpu().numpy()
        y_test = y[test_index].cpu().numpy()
    auc = roc_auc_score(y_test, probs)
    # 计算Precision, Recall和F1分数
    precision = precision_score(y_test, preds)
    recall = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)

    print(f"AUC: {auc}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"accuracy: {accuracy}")
    print('#######################')

    # 返回计算得到的指标
    return auc, precision, recall, f1, accuracy


def load_embeddings_labels_and_G(csv_file_path):
    embeddings = []
    labels = []
    cluster_indices_dict = defaultdict(list)
    train_index = []
    val_index = []
    test_index = []

    with open(csv_file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        headers = next(csvreader)  # Read and store the headers
        # Find the index of the "split" column
        split_col_idx = headers.index("split")

        for idx, row in enumerate(csvreader):
            embedding = [float(value) for value in row[3:-2]]
            embeddings.append(embedding)
            labels.append(row[1])
            cluster_labels = row[-1].split(', ')
            for cluster_label in cluster_labels:
                if cluster_label != 'None':
                    cluster_indices_dict[cluster_label.strip()].append(idx)
            # Collect indices based on the values in the "split" column
            if row[split_col_idx] == 'train':
                train_index.append(idx)
            elif row[split_col_idx] == 'test':
                test_index.append(idx)
            elif row[split_col_idx] == 'val':
                val_index.append(idx)
    embeddings = normalize(embeddings, norm='l2')
    X = torch.tensor(embeddings, dtype=torch.float)
    y = np.array(labels)
    ftext_index = test_index
    # graph_file_path = 'embeding/' + cfg['project'] + '_graphfile.pt'
    graph_file_path = 'embeding/' + cfg['project'] + '_graphfile.pt'
    if os.path.exists(graph_file_path):
        G = torch.load(graph_file_path)
    else:
        edge_list, weight = edge_index_from_features(X, k=cfg['K'])
        G = Data(x=X, edge_index=edge_list)
        # Save the generated graph
        torch.save(G, graph_file_path)

    return X, y, G, train_index, ftext_index


# train and test
def train_and_test():
    auc_list = []
    F1_list = []
    precision_list = []
    recall_list = []
    accuracy_list = []

    print(cfg['project'])
    # csv_file_path = 'embeding/' + cfg['project'] + '.csv'
    csv_file_path = 'embeding/256-embeddings_' + cfg['project'] + '.csv'

    X, y, G, train_index, test_index = load_embeddings_labels_and_G(
        csv_file_path)

    auc, precision, recall, fmeasure, accuracy = HG_supervised_embedding(
        X, y, train_index, test_index, G)

    label_encoder = LabelEncoder()
    # defect prediction

    auc_list.append(auc)
    precision_list.append(precision)
    recall_list.append(recall)
    F1_list.append(fmeasure)
    accuracy_list.append(accuracy)

    avg = []
    avg.append(average_value(auc_list))
    avg.append(average_value(precision_list))
    avg.append(average_value(recall_list))
    avg.append(average_value(F1_list))
    avg.append(average_value(accuracy_list))

    name = ['auc', 'precision', 'recall', 'F1', 'accuracy']
    results = []
    results.append(auc_list)
    results.append(precision_list)
    results.append(recall_list)
    results.append(F1_list)
    results.append(accuracy_list)

    df = pd.DataFrame(data=results)
    df.index = name
    df.insert(0, 'avg', avg)

    # If the folder does not exist, create the folder
    save_path = './result/CSFF-VD/'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    # Record model parameters
    param_suffix = str(cfg['project']) + '_' + str(cfg['n_hid']) + '_' + str(cfg['lr']) + '_' + str(
        cfg['drop_out']) + '_' + str(cfg['max_epoch'])

    df.to_csv(save_path + '/' + param_suffix + '.csv')


# Execute  projects

opt_project = 'ffmpeg'

cfg['k'] = 5
if (opt_project == "qemu"):
    cfg['project'] = "qemu"
    cfg['n_hid'] = 64
    cfg['lr'] = 0.01
    cfg['drop_out'] = 0.3
    cfg['max_epoch'] = 200
elif (opt_project == "ffmpeg"):
    cfg['project'] = "ffmpeg"
    cfg['n_hid'] = 32
    cfg['lr'] = 0.01
    cfg['drop_out'] = 0.1
    cfg['max_epoch'] = 50
elif (opt_project == "reveal"):
    cfg['project'] = "reveal"
    cfg['n_hid'] = 64
    cfg['lr'] = 0.001
    cfg['drop_out'] = 0.1
    cfg['max_epoch'] = 100

train_and_test()
