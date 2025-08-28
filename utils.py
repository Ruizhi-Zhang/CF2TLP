import torch
from sklearn.metrics import roc_auc_score, precision_score
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import numpy as np
# 定义新的时间编码函数
def get_time_encoding(graph_id, embedding_dim):
    position = torch.tensor(graph_id, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embedding_dim))
    time_encoding = torch.zeros(embedding_dim)
    time_encoding[0::2] = torch.sin(position * div_term)  # 偶数维度
    time_encoding[1::2] = torch.cos(position * div_term)  # 奇数维度
    return time_encoding

def decode_predictions_to_adjacency(predictions, GCN, device):
    adjacency_dict = {}

    # 将解码器移至设备
    #decoder = decoder.to(device)
    GCN.eval()
    with torch.no_grad():
        for time, node_features in predictions.items():
            # node_features 的形状: (num_nodes, input_dim)
            node_features = torch.tensor(node_features, dtype=torch.float32).to(device)
            #adjacency_matrix = decoder(node_features).cpu().numpy()  # 解码并移回 CPU
            adjacency_matrix = GCN.forward_mlp(node_features).cpu().numpy()
            adjacency_dict[time] = adjacency_matrix  # 存储结果

    return adjacency_dict

#计算AUC
def calculate_metrics(adjacency_dict, graph_dict):
    auc_list = []
    precision_list = []

    for time in adjacency_dict:
        # 获取预测的邻接矩阵
        predicted_adjacency = adjacency_dict[time]  # (num_nodes, num_nodes)
        num_nodes = predicted_adjacency.shape[0]

        # 获取真实边集合
        true_edges = graph_dict.get(time, torch.empty(0, 2))  # shape: (num_edges, 2)
        true_edges = true_edges.numpy() if isinstance(true_edges, torch.Tensor) else true_edges

        # 构造 ground truth (label) 矩阵
        true_adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        true_adjacency[true_edges[:, 0], true_edges[:, 1]] = 1  # 标记真实边

        # 展平矩阵为一维向量 (num_nodes * num_nodes,)
        true_labels = true_adjacency.flatten().numpy()
        predicted_scores = predicted_adjacency.flatten()

        # 计算 AUC
        if len(set(true_labels)) > 1:  # 避免只有一个类别导致无法计算 AUC
            auc = roc_auc_score(true_labels, predicted_scores)
        else:
            auc = float('nan')  # 类别单一时无法计算 AUC

        # 计算 precision (假设阈值为 0.5)
        predicted_binary = (predicted_scores > 0.5).astype(int)
        precision = precision_score(true_labels, predicted_binary, zero_division=0)

        # 存储当前时间的 AUC 和 precision
        auc_list.append((time, auc))
        precision_list.append((time, precision))

        print(f"Time {time}: AUC={auc:.4f}, Precision={precision:.4f}")

    return auc_list, precision_list

def calculate_auc_and_precision(adjacency_dict, graph_dict):
    results = []

    for time, predicted_adjacency in adjacency_dict.items():
        # 获取真实的边
        if time not in graph_dict:
            print(f"Warning: Time {time} not found in graph_dict.")
            continue

        true_edges_data = graph_dict[time]

        # 检查 true_edges 的类型并提取
        if isinstance(true_edges_data, Data):
            true_edges = true_edges_data.edge_index.t()  # 提取边并转置为 (num_edges, 2)
        else:
            print(f"Error: Unexpected data type at time {time}: {type(true_edges_data)}")
            continue

        # 确保 true_edges 是 2D 张量
        if true_edges.ndimension() != 2 or true_edges.shape[1] != 2:
            print(f"Error: true_edges at time {time} is not a 2D tensor or does not have shape (num_edges, 2). Found shape: {true_edges.shape}")
            continue

        # 获取节点数量
        num_nodes = predicted_adjacency.shape[0]

        # 构造真实邻接矩阵
        true_adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
        true_adjacency[true_edges[:, 0], true_edges[:, 1]] = 1  # 填充真实边
        true_labels = true_adjacency.flatten().numpy()

        # 确保预测邻接矩阵为 NumPy 数组
        if isinstance(predicted_adjacency, torch.Tensor):
            predicted_scores = predicted_adjacency.flatten().detach().cpu().numpy()
        elif isinstance(predicted_adjacency, np.ndarray):
            predicted_scores = predicted_adjacency.flatten()
        else:
            print(f"Error: predicted_adjacency at time {time} is not a valid type: {type(predicted_adjacency)}")
            continue

        # 计算 AUC
        if len(set(true_labels)) > 1:  # 避免只有一个类别导致无法计算 AUC
            auc = roc_auc_score(true_labels, predicted_scores)
        else:
            auc = float('nan')  # 类别单一时无法计算 AUC

        # 计算 Precision
        threshold = 0.5
        predicted_binary = (predicted_scores > threshold).astype(int)
        precision = precision_score(true_labels, predicted_binary, zero_division=0)

        # 保存结果
        results.append((time, auc, precision))
        print(f"Time {time}: AUC={auc:.4f}, Precision={precision:.4f}")

    return results