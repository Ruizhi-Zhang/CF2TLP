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

#这里的参数model是transformer
def test_model(model, enhanced_node_embeddings, context_length, device):
    model.eval()
    times = sorted(enhanced_node_embeddings.keys())
    nodes = len(next(iter(enhanced_node_embeddings.values())))

    predictions = {}
    with torch.no_grad():
        for t_idx in range(len(times) - context_length):#0-35, 0
            current_time = times[t_idx + context_length] #time[5]=170
            inputs = []

            # 构造输入序列
            for node_idx in range(1, nodes + 1): #1-31, node_idx:1
                input_series = [enhanced_node_embeddings[t][node_idx - 1] for t in times[t_idx:t_idx + context_length]]
                # for t in times[0:5], enhanced_node_embeddings[0][0];
                # enhanced_node_embeddings[1][0]
                inputs.append(np.stack(input_series))

            inputs = torch.tensor(np.stack(inputs), dtype=torch.float32).to(device)  # (nodes, context_length, input_dim)
            #节点1有35组序列，接着是节点2的35组

            # 模型预测
            #outputs = model(inputs).cpu().numpy()
            # 获取最后一天的预测值
            outputs = model(inputs).cpu().numpy()[:, -1, :]  # 取最后一天 (30, 80)

            # 存储预测值
            predictions[current_time] = outputs

            # 打印输入输出的形状
            print(f"Time {current_time}: Input Shape {inputs.shape}, Output Shape {outputs.shape}")

    return predictions