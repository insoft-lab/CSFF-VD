import pandas as pd
import torch
import os
from tqdm import tqdm
from torch_geometric.data import Data, DataLoader
import pickle
from models import GGNN


from scipy.special import softmax
import torch
import random
import numpy as np
from sklearn import metrics

import torch.optim as optim

seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# 设置Python随机种子
random.seed(seed)
# 设置NumPy随机种子
np.random.seed(seed)

dataset = "qemu"

best_test_f1 = 0
max_patience = 100
max_epoch = 200

if (dataset == "reveal"):
    max_epoch = 200
    max_patience = 30
elif (dataset == "ffmpeg"):
    max_epoch = 100
    max_patience = 15
elif (dataset == "qemu"):
    max_epoch = 100
    max_patience = 15


# 设置PyTorch随机种子

model = GGNN().to('cuda')

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[100], gamma=0.9)
crit = torch.nn.CrossEntropyLoss()


train_list_path = f'data-process/{dataset}_train_data.pkl'
val_list_path = f'data-process/{dataset}_val_data.pkl'
test_list_path = f'data-process/{dataset}_test_data.pkl'

with open(train_list_path, 'rb') as f:
    train_dataset = pickle.load(f)
with open(val_list_path, 'rb') as f:
    val_dataset = pickle.load(f)
with open(test_list_path, 'rb') as f:
    test_dataset = pickle.load(f)
train_loader = DataLoader(train_dataset, batch_size=64,
                          shuffle=False, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=64,
                        shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=64,
                         shuffle=False, num_workers=0)

best_validation_loss = float('inf')
log_file = open(f'result/GGNN/result-{dataset}.txt', 'w')
best_loss = float('inf')


def train():
    model.train()
    loss_all = 0
    for data in tqdm(train_loader, desc="Training Progress"):

        data = data.to('cuda')
        # print('data',data)
        optimizer.zero_grad()
        output = model(data)
        label = [int(x) for x in data.y]
        label = torch.tensor(label).to('cuda')

        loss = crit(output, label)
        loss.backward()
        loss_all += loss.item()
        optimizer.step()
    scheduler.step()
    return loss_all / len(train_loader)


def evaluate_model(model, dataloader, device):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for data in dataloader:
            data.to(device)

            labels = [int(x) for x in data.y]
            labels = torch.tensor(labels).float().to('cuda')

            # 使用Softmax将输出转化为概率分数
            outputs = model(data).cpu().numpy()
            y_true.extend(labels.cpu().numpy())
            prob_scores = softmax(outputs, axis=1)
            # y_pred.extend(prob_scores.cpu().numpy())
            y_pred.extend(prob_scores)

    y_true = np.array(y_true)

    y_pred = np.array(y_pred)
    y_pred1 = y_pred[:, 1]
    y_pred_classes = np.argmax(y_pred, axis=1)
    auc_score = metrics.roc_auc_score(y_true, y_pred1)
    # 计算Precision、Recall和F1
    # y_pred_classes = np.argmax(y_pred, axis=1)
    accuracy = metrics.accuracy_score(y_true, y_pred_classes)
    precision = metrics.precision_score(y_true, y_pred_classes)
    recall = metrics.recall_score(y_true, y_pred_classes)
    f1 = metrics.f1_score(y_true, y_pred_classes)
    evaluate_list = []
    evaluate_list.append(auc_score)
    evaluate_list.append(accuracy)
    evaluate_list.append(precision)
    evaluate_list.append(recall)
    evaluate_list.append(f1)

    return evaluate_list


patience_counter = 0

for epoch in range(max_epoch):
    print('epoch:', epoch)
    loss = train()
    print('Training loss:', loss)
    val_list = evaluate_model(model, val_loader, device='cuda')
    test_list = evaluate_model(model, test_loader, device='cuda')
    print(f'Validation AUC: {val_list[0]:.4f}, Test AUC: {test_list[0]:.4f}')
    print(
        f'Validation Accuracy: {val_list[1]:.4f}, Test Accuracy: {test_list[1]:.4f}')
    print(
        f'Validation precision: {val_list[2]:.4f}, Test precision: {test_list[2]:.4f}')
    print(
        f'Validation recall: {val_list[3]:.4f}, Test recall: {test_list[3]:.4f}')
    print(f'Validation F1: {val_list[4]:.4f}, Test F1: {test_list[4]:.4f}')

    # 将信息写入txt文件
    log_file.write(f'Epoch {epoch}\n')
    log_file.write(f'Training Loss: {loss:.4f}\n')
    log_file.write(f'Validation AUC: {val_list[0]:.4f}\n')
    log_file.write(f'Test AUC: {test_list[0]:.4f}\n')
    log_file.write(f'Validation accuracy: {val_list[1]:.4f}\n')
    log_file.write(f'Test accuracy: {test_list[1]:.4f}\n')
    log_file.write(f'Validation precision: {val_list[2]:.4f}\n')
    log_file.write(f'Test precision: {test_list[2]:.4f}\n')
    log_file.write(f'Validation recall: {val_list[3]:.4f}\n')
    log_file.write(f'Test recall: {test_list[3]:.4f}\n')
    log_file.write(f'Validation F1: {val_list[4]:.4f}\n')
    log_file.write(f'Test F1: {test_list[4]:.4f}\n')
    log_file.write('\n')  # 为了区分每个epoch的记录

    if best_loss > loss:
        # 指定保存模型的目录
        save_dir = 'models/best_model'
        # 检查目录是否存在，不存在则创建
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 使用字符串格式化来加入epoch的值，并指定保存路径
        filename = f'{save_dir}/best_loss_{dataset}_modelGCN-{epoch}.pth'
        torch.save(model.state_dict(), filename)
        best_loss = loss
        print('best_loss:', best_loss)

    if best_test_f1 < test_list[4]:
        filename = f'{save_dir}/best_f1_{dataset}_modelGCN-{epoch}.pth'
        torch.save(model.state_dict(), filename)
        best_test_f1 = test_list[4]
        print('best_test_f1:', best_test_f1)
        patience_counter = 0
        print("Patience counter cleared.")
    else:
        patience_counter += 1
        print("Patience counter increased, current patience:", patience_counter)
        print("history best f1 in test set:", best_test_f1)

    if patience_counter >= max_patience:
        print("Maximum patience reached. Exiting.")
        break

log_file.close()
