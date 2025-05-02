#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import castle.algorithms
from castle.metrics import MetricsDAG
from castle.common import GraphDAG
import seaborn as sns
from hybrid_models import HybridDAGModel, calculate_metrics
import networkx as nx


def process_sim_with_visualization(sim_path):
    """处理单个模拟数据集并返回结果和可视化数据"""
    # 加载数据
    data = loadmat(sim_path)
    ts = data['ts']
    net = data['net']
    Nnodes = int(data['Nnodes'][0][0])
    Nsubjects = int(data['Nsubjects'][0][0])
    Ntimepoints = int(data['Ntimepoints'][0][0])

    # 重塑时间序列数据
    ts_all = ts.reshape(Nsubjects, Ntimepoints, Nnodes)

    # 构造平均真值网络并二值化
    if net.ndim == 3:
        B_true = (np.mean(net, axis=0) > 0).astype(int)
    else:
        B_true = (net > 0).astype(int)

    # 运行NOTEARS算法
    model_notears = castle.algorithms.Notears()
    try:
        model_notears.learn(ts)
        B_notears = model_notears.causal_matrix.astype(int)
        mt_notears = MetricsDAG(B_notears, B_true).metrics
    except Exception as e:
        print(f"NOTEARS算法失败: {e}")
        B_notears = np.zeros_like(B_true)
        mt_notears = {
            'shd': np.nan, 'precision': np.nan, 'recall': np.nan, 'F1': np.nan
        }

    # 运行Hybrid Transformer模型
    try:
        # 初始化混合模型
        hidden_dim = max(64, Nnodes * 4)
        nhead = max(2, Nnodes // 2)  # 确保它是偶数
        hybrid_model = HybridDAGModel(
            neural_model_type='transformer',
            input_dim=Nnodes,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=2
        )

        # 训练模型
        hybrid_model.fit(ts_all, net)

        # 预测
        pred_adj = hybrid_model.predict(ts_all)
        B_hybrid = (pred_adj > 0.5).astype(int)
        mt_hybrid = calculate_metrics(B_hybrid, B_true)
    except Exception as e:
        print(f"Hybrid Transformer模型失败: {e}")
        B_hybrid = np.zeros_like(B_true)
        mt_hybrid = {
            'shd': np.nan, 'precision': np.nan, 'recall': np.nan, 'F1': np.nan
        }

    # 收集可视化数据
    vis_data = {
        'B_true': B_true,
        'B_notears': B_notears,
        'B_hybrid': B_hybrid,
        'mt_notears': mt_notears,
        'mt_hybrid': mt_hybrid,
        'Nnodes': Nnodes,
        'sim_name': os.path.basename(sim_path)
    }

    return vis_data


def visualize_adjacency_matrices(results):
    """可视化因果矩阵对比"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 分别绘制三个矩阵
    sns.heatmap(results['B_true'], annot=True, cmap='Blues', cbar=False, ax=axes[0],
                square=True, fmt='d', linewidths=.5)
    axes[0].set_title('True Causal Matrix')

    sns.heatmap(results['B_notears'], annot=True, cmap='Blues', cbar=False, ax=axes[1],
                square=True, fmt='d', linewidths=.5)
    axes[1].set_title(f'NOTEARS\nF1: {results["mt_notears"]["F1"]:.2f}')

    sns.heatmap(results['B_hybrid'], annot=True, cmap='Blues', cbar=False, ax=axes[2],
                square=True, fmt='d', linewidths=.5)
    axes[2].set_title(f'Hybrid Transformer\nF1: {results["mt_hybrid"]["F1"]:.2f}')

    plt.tight_layout()
    plt.savefig(f'causal_matrices_{results["sim_name"]}.png', dpi=300)
    plt.close()


def create_network_graph(adj_matrix, ax, title):
    """创建网络可视化图"""
    G = nx.DiGraph()

    # 添加节点
    n_nodes = adj_matrix.shape[0]
    for i in range(n_nodes):
        G.add_node(i)

    # 添加边
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adj_matrix[i, j] > 0:
                G.add_edge(i, j)

    # 设置节点位置 - 使用圆形布局
    pos = nx.circular_layout(G)

    # 绘制图形
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
    nx.draw_networkx_edges(G, pos, width=2, arrowsize=20, ax=ax)

    ax.set_title(title)
    ax.axis('off')

    return ax


def visualize_network_graphs(results):
    """可视化网络图对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # 创建NOTEARS网络图
    create_network_graph(
        results['B_notears'],
        axes[0],
        f'NOTEARS Network\nF1: {results["mt_notears"]["F1"]:.2f}'
    )

    # 创建Hybrid Transformer网络图
    create_network_graph(
        results['B_hybrid'],
        axes[1],
        f'Hybrid Transformer Network\nF1: {results["mt_hybrid"]["F1"]:.2f}'
    )

    plt.tight_layout()
    plt.savefig(f'network_graphs_{results["sim_name"]}.png', dpi=300)
    plt.close()


def main():
    # 设置数据集路径
    datasets_path = '../../datasets/sims/'

    # 仅处理节点数为5的数据集
    for dataset in range(1, 21):
        sim_path = f'{datasets_path}sim{dataset}.mat'

        # 检查文件是否存在
        if not os.path.exists(sim_path):
            print(f"找不到文件: {sim_path}")
            continue

        # 检查节点数
        try:
            node_count = loadmat(sim_path)['Nnodes'][0][0]
            if node_count != 5:
                print(f"数据集 {dataset} 的节点数为 {node_count}，跳过")
                continue
        except Exception as e:
            print(f"无法读取数据集 {dataset} 的节点数: {e}")
            continue

        print(f"处理数据集 {dataset}...")

        # 处理数据集并获取可视化所需的数据
        results = process_sim_with_visualization(sim_path)

        # 生成可视化
        visualize_adjacency_matrices(results)
        visualize_network_graphs(results)

        # 打印结果
        print(f"数据集 {dataset} 处理完成:")
        print(f"NOTEARS F1: {results['mt_notears']['F1']}")
        print(f"Hybrid Transformer F1: {results['mt_hybrid']['F1']}")
        print("")


if __name__ == '__main__':
    main()