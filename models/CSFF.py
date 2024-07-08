import torch
from torch import nn
from dhg.nn import HGNNConv
from torch_geometric.nn import ResGatedGraphConv, GATConv, GPSConv, SuperGATConv

from torch_geometric.nn import TopKPooling, GCNConv, ResGatedGraphConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from scipy.special import softmax


class GGNN(torch.nn.Module):  # 针对图进行分类任务
    def __init__(self):
        super(GGNN, self).__init__()

        self.conv1 = ResGatedGraphConv(256, 512)
        self.pool1 = TopKPooling(512, ratio=0.7)
        self.conv2 = ResGatedGraphConv(512, 512)
        self.pool2 = TopKPooling(512, ratio=0.7)
        self.conv3 = ResGatedGraphConv(512, 512)
        self.pool3 = TopKPooling(512, ratio=0.7)
        self.lin1 = torch.nn.Linear(512, 256)
        self.lin2 = torch.nn.Linear(256, 64)
        self.lin3 = torch.nn.Linear(64, 2)

        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)

        # self.bn2 = torch.nn.BatchNorm1d(512)
        # self.bn3 = torch.nn.BatchNorm1d(512)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (ResGatedGraphConv, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:n*1,其中每个图里点的个数是不同的

        x = F.relu(self.conv1(x, edge_index))  # n*128
        x = self.dropout(x)

        x, edge_index, _, batch, _, _ = self.pool1(
            x, edge_index, None, batch)  # pool之后得到 n*0.8个点
        x1 = gap(x, batch)
        x = F.relu(self.conv2(x, edge_index))
        # x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x2 = gap(x, batch)
        # x = self.bn3(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = gap(x, batch)
        x = x1 + x2 + x3  # 获取不同尺度的全局特征

        x = self.lin1(x)
        x = self.act1(x)
        x = self.lin2(x)
        x = self.act2(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin3(x)

        return x


class SuperGAT(nn.Module):
    def __init__(self, in_ch, n_hid, dropout=0.5):
        super(SuperGAT, self).__init__()
        self.dropout = dropout
        self.hgcn1 = SuperGATConv(in_channels=in_ch, out_channels=n_hid)
        self.bn1 = nn.BatchNorm1d(n_hid)

        self.hgcn2 = SuperGATConv(in_channels=n_hid, out_channels=n_hid)
        self.bn2 = nn.BatchNorm1d(n_hid)

        self.fc1 = nn.Linear(n_hid, n_hid)
        self.bn3 = nn.BatchNorm1d(n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.bn4 = nn.BatchNorm1d(n_hid)
        self.fc3 = nn.Linear(n_hid, 2)

        self.act = nn.ReLU()

    def forward(self, x, G):

        x = self.act(self.bn1(self.hgcn1(x, G.edge_index)))
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.act(self.bn2(self.hgcn2(x, G.edge_index)))
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.act(self.bn3(self.fc1(x)))
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.act(self.bn4(self.fc2(x)))
        x = F.dropout(x, self.dropout, training=self.training)

        x = self.fc3(x)  # 直接返回logits，不使用激活函数
        return x


class GGNN_embedding(torch.nn.Module):  # 针对图进行分类任务
    def __init__(self):
        super(GGNN_embedding, self).__init__()

        self.conv1 = ResGatedGraphConv(256, 512)
        self.pool1 = TopKPooling(512, ratio=0.7)
        self.conv2 = ResGatedGraphConv(512, 512)
        self.pool2 = TopKPooling(512, ratio=0.7)
        self.conv3 = ResGatedGraphConv(512, 512)
        self.pool3 = TopKPooling(512, ratio=0.7)
        self.lin1 = torch.nn.Linear(512, 256)
        self.lin2 = torch.nn.Linear(256, 64)
        self.lin3 = torch.nn.Linear(64, 2)

        self.bn2 = torch.nn.BatchNorm1d(512)
        self.bn3 = torch.nn.BatchNorm1d(512)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)

        # self.bn2 = torch.nn.BatchNorm1d(512)
        # self.bn3 = torch.nn.BatchNorm1d(512)
        self.act1 = torch.nn.ReLU()
        self.act2 = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (ResGatedGraphConv, torch.nn.Linear)):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm1d):
                torch.nn.init.constant_(m.weight, 1)
                torch.nn.init.constant_(m.bias, 0)

    def forward(self, data, return_intermediate=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch  # x:n*1,其中每个图里点的个数是不同的

        x = F.relu(self.conv1(x, edge_index))  # n*128
        x = self.dropout(x)

        x, edge_index, _, batch, _, _ = self.pool1(
            x, edge_index, None, batch)  # pool之后得到 n*0.8个点
        x1 = gap(x, batch)
        x = F.relu(self.conv2(x, edge_index))
        # x = self.bn2(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x2 = gap(x, batch)
        # x = self.bn3(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.dropout(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = gap(x, batch)
        x = x1 + x2 + x3  # 获取不同尺度的全局特征

        x = self.lin1(x)
        if return_intermediate:
            return x  # 返回self.lin1(x)的结果
        x = self.act1(x)
        x = self.lin2(x)

        x = self.act2(x)
        # x = F.dropout(x, p=0.5, training=self.training)

        x = self.lin3(x)

        return x
