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
from preprocess import *
from utils import *
from GCN import *
from Transformer import *
from generate import *
import pickle

def main():
    # 传入40个图csv
    folder_path = 'D:/code_practice/The_3rd_paper/snapshots'
    file_indices = [1,  34,  68, 102, 136, 170, 204 ,238, 272,306,
                340, 374, 408, 442, 476, 510,544, 578,612, 646,
                680, 714, 748, 782, 816, 850, 884, 918, 952, 986,
                987,988, 989,990,991,992,993,994,995,996,
                997,998,999, 1000, 1001, 1002, 1003, 1004, 1005, 1006]
    threshold = 0.05

    matrixs = load_and_threshold_matrices(folder_path, file_indices, threshold)
    graph_dict = convert_all_matrices_to_graphs(matrixs)

    """
    数据传入GCN，生成并存储节点编码
    """
    input_dim_gcn = 30  # 输入维度（节点特征的维度）
    hidden_dim = 128  # 隐藏层维度
    output_dim = 64  # 输出维度（节点嵌入的维度）


    # 初始化 GCN 模型
    model = GCN(input_dim_gcn, hidden_dim, output_dim)

    # 设置优化器
    optimizer_gcn = optim.Adam(model.parameters(), lr=0.01)

    # 训练参数
    epochs = 100
    node_embeddings = {}  # 用于存储每个图的节点嵌入

    model.train()
    for epoch in range(epochs):
        # 对单个图进行训练
        for graph_id, graph in graph_dict.items():
            # print(f"Training on graph {graph_id}...")
            # 创建目标邻接矩阵
            num_nodes = graph.x.size(0)  # 从图中获取节点数
            adj_target = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
            adj_target[graph.edge_index[0], graph.edge_index[1]] = 1
            optimizer_gcn.zero_grad()

            # 获取节点嵌入
            _, adj_pred = model(graph)  # 节点编码,
            # adj_pred = torch.sigmoid(torch.matmul(z, z.t()))  # 内积预测邻接矩阵

            # 计算重构损失, 改为MSE
            loss = F.mse_loss(adj_pred, adj_target)
            loss.backward()
            optimizer_gcn.step()

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # 存储该图的节点嵌入
    model.eval()
    for graph_id, graph in graph_dict.items():
        z, _ = model(graph)
        node_embeddings[graph_id] = z.detach().clone()
    # print(f"Finished training on graph {graph_id}, final loss: {loss.item():.4f}")

    print("GCN Training on all graphs complete!")


    # 批量在40个图的节点编码后面添加时间编码
    # 时间编码维度
    time_dim = 16

    # 存储增强后的节点嵌入
    enhanced_node_embeddings = {}

    for graph_id, node_embedding in node_embeddings.items():
        # 获取时间编码
        num_nodes = node_embedding.shape[0]  # 节点数
        time_encoding = get_time_encoding(graph_id, time_dim)
        time_encoding_expanded = time_encoding.unsqueeze(0).repeat(num_nodes, 1)  # 扩展时间编码到每个节点

        # 拼接节点嵌入和时间编码
        enhanced_embedding = torch.cat([node_embedding, time_encoding_expanded], dim=1)

        # 存储增强后的嵌入
        enhanced_node_embeddings[graph_id] = enhanced_embedding # 有时间编码
        # enhanced_node_embeddings[graph_id] = node_embedding  # 无时间编码

    # 保存增强后的节点嵌入
    torch.save(enhanced_node_embeddings, "enhanced_node_embeddings.pth")
    print("Enhanced node embeddings with time encoding saved to 'enhanced_node_embeddings.pth'.")

    #训练transformer参数
    context_length =9
    batch_size = 4
    num_epochs = 10
    learning_rate = 1e-3
    hidden_dim = 128
    num_heads = 4
    num_layers = 2
    train_ratio = 0.8

    # 准备数据
    data, times = prepare_data(enhanced_node_embeddings, context_length, train_ratio)
    dataloader = create_dataloader(data, batch_size)

    # 打印示例输入输出
    for i, (x, y) in enumerate(dataloader):
        print(f"Batch {i + 1} - Input (x): {x.shape}, Target (y): {y.shape}")
        break  # 只展示一个批次

    # 定义transformer模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_t = TransformerModel(input_dim=80, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers)

    # 训练transformer模型
    train_model(model_t, dataloader, num_epochs, learning_rate, device)
    print("Transformer Training complete!")

    # 测试transformer模型并生成预测值
    predictions = test_model(model_t, enhanced_node_embeddings, context_length, device)
    print(f"Generated predictions for {len(predictions)} time points.")

    #  加上时间编码再截取前64维,
    predictions_time = {}
    for time, graph in predictions.items():
        predictions_time[time] = graph[:, :64]

    # 使用 with open 和 pickle.dump 保存数据
    with open("predictions_time.pkl", "wb") as f:  # "wb" 表示以二进制写入模式
        pickle.dump(predictions_time, f)


    # 解码 predictions 为邻接矩阵
    # adjacency_dict = decode_predictions_to_adjacency(predictions, model, device)  # 没有时间信息编码
    adjacency_dict = decode_predictions_to_adjacency(predictions_time, model, device)

    with open('adjacency_dict.pkl', 'wb') as f:
        pickle.dump(adjacency_dict, f)

    # 计算 AUC 和 Precision
    results = calculate_auc_and_precision(adjacency_dict, graph_dict)


    # 输出最终结果AUC和precision
    auc_list = []
    precision_list = []
    for time, auc, precision in results:
        print(f"Time {time}: AUC={auc:.4f}, Precision={precision:.4f}")
        auc_list.append(round(auc, 4))  # 存储 AUC 值
        precision_list.append(round(precision, 4))
    print('auc_list:', auc_list)
    print('precision_list:', precision_list)

    #  计算MSE
    """   
    MSE_list = []
    for graph_id, adj_pred in adjacency_dict.items():
        if graph_id in graph_dict:
            # 获取真实图数据
            graph_data = graph_dict[graph_id]
            num_nodes = graph_data.x.size(0)

            # 构造真实的加权邻接矩阵
            adj_real = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device='cpu')
            adj_real[graph_data.edge_index[0], graph_data.edge_index[1]] = graph_data.edge_attr

            # 确保预测矩阵是 torch.Tensor 类型并与真实矩阵在同一设备上
            if isinstance(adj_pred, np.ndarray):
                adj_pred = torch.tensor(adj_pred, dtype=torch.float32, device='cpu')

            # 计算预测矩阵和真实矩阵之间的 MSE
            try:
                mse = F.mse_loss(adj_pred, adj_real)
                MSE_list.append(round(mse.item(), 4))
            except Exception as e:
                print(f"Error calculating MSE for graph {graph_id}: {e}")
        else:
            print(f"Graph ID {graph_id} not found in graph_dict.")
    print('MSE_list:', MSE_list)
    """

if __name__ == '__main__':
    main()
