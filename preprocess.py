import os
import pandas as pd
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import torch.nn as nn
import math
from sklearn.metrics import roc_auc_score, precision_score

def load_and_threshold_matrices(folder_path, file_indices, threshold):
    """
    读取文件夹中的CSV文件，并根据阈值处理矩阵。

    Args:
        folder_path (str): 文件夹路径。
        file_indices (list of int): 需要读取的文件编号列表，例如 [1, 2, ..., 40]。
        threshold (float): 用于处理矩阵的阈值。

    Returns:
        dict: 包含处理后矩阵的字典，键为文件名数字，值为对应的矩阵。
    """

    processed_matrices = {}

    for idx in file_indices:
        file_path = os.path.join(folder_path, f"CT_Day_{idx}.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在！")

        # 读取CSV文件，跳过第一行和第一列
        matrix = pd.read_csv(file_path, index_col=0).values.astype(float)

        # 根据阈值处理矩阵：大于阈值保持原数据，小于或等于阈值设为 0
        thresholded_matrix = np.where(matrix > threshold, matrix, 0)

        # 保存结果到字典
        processed_matrices[idx] = thresholded_matrix

    return processed_matrices

def normalize_adjacency_matrix(adj_matrix):
    """
    对加权邻接矩阵进行对称归一化。

    参数:
    adj_matrix (torch.Tensor): 原始加权邻接矩阵 (shape: [num_nodes, num_nodes])

    返回:
    normalized_matrix (torch.Tensor): 对称归一化后的邻接矩阵
    """
    # 计算度矩阵
    degree = adj_matrix.sum(dim=1)  # shape: [num_nodes]
    # 避免除以 0，将度为 0 的地方设置为 1
    degree[degree == 0] = 1
    # 计算 D^(-1/2)
    d_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
    # 对称归一化公式
    normalized_matrix = d_inv_sqrt @ adj_matrix @ d_inv_sqrt
    return normalized_matrix

# 一个csv文件转为pyg的graph
def convert_matrix_to_graph(adj_matrix):
    """
    将加权邻接矩阵转换为 PyG 图格式，并进行对称归一化。

    参数:
    adj_matrix (torch.Tensor): 加权邻接矩阵 (shape: [num_nodes, num_nodes])

    返回:
    graph (torch_geometric.data.Data): PyG 图对象
    """
    # 如果输入是 NumPy 数组，先转换为 PyTorch 张量
    if isinstance(adj_matrix, np.ndarray):
        adj_matrix = torch.from_numpy(adj_matrix).float()

    # 对邻接矩阵进行对称归一化
    adj_matrix = normalize_adjacency_matrix(adj_matrix)

    # 提取非零元素的索引和权重
    edge_index = torch.nonzero(adj_matrix, as_tuple=False).T  # 转置后 shape: [2, num_edges]
    edge_attr = adj_matrix[edge_index[0], edge_index[1]]  # 获取边权重，shape: [num_edges]
    num_nodes = adj_matrix.size(0)  # 节点数量

    # 初始化节点特征为 one-hot 编码
    x = torch.eye(num_nodes)

    # 创建 PyG 图数据对象
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

#批量处理csv文件转graph
def convert_all_matrices_to_graphs(processed_matrices):
    """
    将字典中存储的多个加权邻接矩阵转换为 PyG 图格式。

    参数:
    processed_matrices (dict): 存储多个邻接矩阵的字典，键为矩阵名称，值为加权邻接矩阵 (torch.Tensor)

    返回:
    graph_dict (dict): 存储多个 PyG 图的字典，键与原字典相同，值为对应的 PyG 图对象
    """
    graph_dict = {}
    for key, matrix in processed_matrices.items():
        # 将矩阵转换为图格式
        graph_dict[key] = convert_matrix_to_graph(matrix)
    return graph_dict