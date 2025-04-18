#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import castle.algorithms
import numpy as np
import torch
from scipy.io import loadmat
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
import csv

# 设置 Castle 后端为 PyTorch
os.environ['CASTLE_BACKEND'] = 'pytorch'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def process_sim(sim_path):
    data = loadmat(sim_path)
    # 确保数据类型为 float64 (double)
    ts = torch.tensor(data['ts'], dtype=torch.float64, device=device)
    net = torch.tensor(data['net'], dtype=torch.float64, device=device)

    # 从 mat 文件读取维度
    Nnodes = int(data['Nnodes'][0][0])
    Nsubjects = int(data['Nsubjects'][0][0])
    Ntimepoints = int(data['Ntimepoints'][0][0])

    # 特征矩阵 X
    X = ts.cpu().numpy()  # castle 库需要 numpy 数组
    X = X.astype(np.float64)  # 确保 numpy 数组也是 float64

    # 构造平均真值网络并二值化
    if net.dim() == 3:
        # 多个受试者：先在 dim=0 上求均值
        B_true = (torch.mean(net, dim=0) != 0).int()
    else:
        # 单一网络
        B_true = (net != 0).int()

    B_true = B_true.cpu().numpy()  # 转回 numpy 以供评估

    # 用 RL 算法学习结构
    model = castle.algorithms.RL()

    try:
        model.learn(X)
    except ValueError as e:
        print(f"⚠️ RL 在 {os.path.basename(sim_path)} 上失败：{e}")
        return os.path.basename(sim_path), {
            'shd': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'F1': np.nan
        }

    B_est = model.causal_matrix.astype(int)

    # 评价指标
    mt = MetricsDAG(B_est, B_true)
    return os.path.basename(sim_path), mt.metrics


def main():
    # 设置 PyTorch 默认数据类型为 float64
    torch.set_default_dtype(torch.float64)

    # 仿真数据目录（相对于此脚本）
    sim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../datasets/sims'))
    sim_files = sorted(glob.glob(os.path.join(sim_dir, 'sim*.mat')))

    metrics_lists = {
        'SHD': [],
        'F1': [],
        'Precision': [],
        'Recall': []
    }

    print("开始遍历 NetSim 仿真数据并进行结构学习：\n")
    for sim_path in sim_files:
        Node = loadmat(sim_path)['Nnodes'][0][0]
        if Node != 5:
            continue

        print("CURRENT DATA:", sim_path)
        name, metrics = process_sim(sim_path)

        # 使用字典来存储和更新指标
        current_metrics = {
            'SHD': metrics['shd'],
            'F1': metrics['F1'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall']
        }

        # 更新指标列表
        for metric_name, value in current_metrics.items():
            if not np.isnan(value):
                metrics_lists[metric_name].append(value)

        print(f"SHD = {current_metrics['SHD']:.4f}, "
              f"F1 = {current_metrics['F1']:.4f}, "
              f"Precision = {current_metrics['Precision']:.4f}, "
              f"Recall = {current_metrics['Recall']:.4f}")
        print("")

    # 计算并输出统计结果
    for metric_name, values in metrics_lists.items():
        values_tensor = torch.tensor(values, device=device, dtype=torch.float64)
        mean = torch.mean(values_tensor).item()
        var = torch.var(values_tensor).item()
        print(f"{metric_name} mean: {mean:.4f}")
        print(f"{metric_name} variance: {var:.4f}")

    # 清理 GPU 内存
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()