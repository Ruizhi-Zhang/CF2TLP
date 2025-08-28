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
import math
from sklearn.metrics import roc_auc_score, precision_score
from torch.utils.data import Dataset, DataLoader

def prepare_data(enhanced_node_embeddings, context_length, train_ratio=0.8):
    nodes = len(next(iter(enhanced_node_embeddings.values())))  # 节点数量
    times = sorted(enhanced_node_embeddings.keys())
    train_size = int(len(times) * train_ratio)
    train_times = times[:train_size]  # 时间点列表

    # 用于存储每个节点的样本
    data = {node_idx: [] for node_idx in range(1, nodes + 1)}

    for node_idx in range(1, nodes + 1):
        # 提取单个节点的时间序列数据
        node_series = [enhanced_node_embeddings[t][node_idx - 1] for t in train_times]

        # 构造样本 x 和 y
        for i in range(len(node_series) - context_length):
            x = np.stack(node_series[i:i + context_length])  # 确保是二维数组
            y = np.stack(node_series[i + 1:i + context_length + 1])  # 确保是二维数组
            data[node_idx].append((x, y))

    return data, times

# 自定义数据集类
class NodeEmbeddingDataset(Dataset):
    def __init__(self, data):
        self.data = []
        for samples in data.values():
            self.data.extend(samples)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# 定义 Transformer 模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads, dim_feedforward=hidden_dim)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = x.permute(1, 0, 2)  # 转换为 (context_length, batch_size, input_dim)
        x = self.transformer(x)  # 输出 (context_length, batch_size, input_dim)
        x = x.permute(1, 0, 2)  # 转回 (batch_size, context_length, input_dim)
        return self.fc(x)

# 构建数据加载器
def create_dataloader(data, batch_size):
    dataset = NodeEmbeddingDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 训练模型
def train_model(model, dataloader, num_epochs, learning_rate, device):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
'''
class TransformerModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, hidden_dim, output_dim):
        super(TransformerModel, self).__init__()

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim
        )

        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.fc = nn.Linear(input_dim, output_dim)  # 输出层，将 Transformer 的输出映射为节点编码

    def forward(self, x):
        """
        前向传播，输入为节点嵌入和时间编码的拼接。

        参数:
            x (torch.Tensor): 输入的节点嵌入，形状为 [num_nodes, input_dim]。

        返回:
            out (torch.Tensor): Transformer 生成的节点编码，形状为 [num_nodes, output_dim]。
        """
        x = self.transformer_encoder(x)  # Transformer 编码
        out = self.fc(x)  # 映射到目标维度
        return out
'''