import os
import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data

import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn as nn



# 利用pyg搭建GCN
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        构建一个支持加权图的两层GCN模型

        参数:
        - input_dim: 输入特征的维度
        - hidden_dim: 隐藏层的特征维度
        - output_dim: 输出的节点特征维度
        """
        super(GCN, self).__init__()

        # 第一层 GCN
        self.gcn1 = GCNConv(input_dim, hidden_dim)

        # 第二层 GCN
        self.gcn2 = GCNConv(hidden_dim, output_dim)

        # MLP 用于重构加权邻接矩阵
        self.mlp = MLPDecoder(output_dim)

    def forward(self, data):
        """
        前向传播

        参数:
        - data: PyG 图对象，包含 edge_index, edge_attr 和 x (节点特征)

        返回:
        - x: 节点的最终特征表示
        - adj: 重构的加权邻接矩阵
        """
        # 获取图数据中的特征、边索引和边权重
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr

        # 第一层 GCN + 激活函数 ReLU
        x = F.relu(self.gcn1(x, edge_index, edge_weight))

        # 第二层 GCN
        x = self.gcn2(x, edge_index, edge_weight)

        # 使用 MLP 重构邻接矩阵
        adj = self.mlp(x)

        return x, adj

    def forward_mlp(self, x):
        return self.mlp(x)

class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128):
        """
        MLP 用于重构加权邻接矩阵

        参数:
        - input_dim: 节点嵌入的维度
        - hidden_dim: MLP 隐藏层维度
        """
        super(MLPDecoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),  # 输入是两个节点的嵌入拼接
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),  # 输出是边的权重
            nn.Sigmoid(),  # 压缩到 [0, 1]
        )

    def forward(self, z):
        """
        前向传播，重构邻接矩阵

        参数:
        - z: 节点嵌入，形状为 [num_nodes, embedding_dim]

        返回:
        - adj_matrix: 重构的加权邻接矩阵，形状为 [num_nodes, num_nodes]
        """
        num_nodes = z.size(0)

        # 获取所有节点对的索引（包含非对称对）
        i_indices, j_indices = torch.meshgrid(
            torch.arange(num_nodes), torch.arange(num_nodes), indexing="ij"
        )
        i_indices = i_indices.flatten()
        j_indices = j_indices.flatten()

        # 拼接每对节点的嵌入（注意非对称）
        edges = torch.cat([z[i_indices], z[j_indices]], dim=1)  # 拼接每对节点的嵌入

        # 使用 MLP 预测边的权重
        edge_weights = self.mlp(edges).squeeze(-1)

        # 构造加权邻接矩阵
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=z.device)
        adj_matrix[i_indices, j_indices] = edge_weights

        return adj_matrix

    """
    def forward(self, z):
        num_nodes = z.size(0)
        i_indices, j_indices = torch.triu_indices(num_nodes, num_nodes, offset=0)  # 获取上三角索引
        edges = torch.cat([z[i_indices], z[j_indices]], dim=1)  # 拼接每对节点的嵌入
        edge_probs = self.mlp(edges).squeeze(-1)  # 使用 MLP 预测边的概率
        return edge_probs, (i_indices, j_indices)
    """