#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob

import castle.algorithms
import numpy as np
from scipy.io import loadmat
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
import csv

def process_sim(sim_path):
    data = loadmat(sim_path)
    ts = data['ts']     # (Nsubjects*Ntimepoints, Nnodes)
    net = data['net']   # (Nsubjects, Nnodes, Nnodes) 或 (Nnodes, Nnodes)

    # 从 mat 文件读取维度
    Nnodes = int(data['Nnodes'][0][0])
    Nsubjects = int(data['Nsubjects'][0][0])
    Ntimepoints = int(data['Ntimepoints'][0][0])

    # 特征矩阵 X
    X = ts  # 已经展平，无需 reshape

    # 构造平均真值网络并二值化
    if net.ndim == 3:
        # 多个受试者：先在 axis=0 上求均值
        B_true = (np.mean(net, axis=0) != 0).astype(int)
    else:
        # 单一网络
        B_true = (net != 0).astype(int)

    # 用 PC 算法学习结构
    model = castle.algorithms.Notears()
    # print(f"模型运行在设备: {model.device}")

    try:
        model.learn(X)
    except ValueError as e:
        print(f"⚠️ GES 在 {os.path.basename(sim_path)} 上失败：{e}")
        # 返回一个全 NaN 的 metrics，方便后续过滤
        return os.path.basename(sim_path), {
            'shd': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'F1': np.nan
        }

    B_est = model.causal_matrix.astype(int)

    print(B_est)

    # 评价指标
    mt = MetricsDAG(B_est, B_true)
    return os.path.basename(sim_path), mt.metrics

def main():
    # 仿真数据目录（相对于此脚本）
    sim_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/sims'))
    sim_files = sorted(glob.glob(os.path.join(sim_dir, 'sim*.mat')))

    SHD = []
    F1 = []
    Precision = []
    Recall = []

    print("开始遍历 NetSim 仿真数据并进行结构学习：\n")
    for sim_path in sim_files:

        Node = loadmat(sim_path)['Nnodes'][0][0]
        if Node != 5:
            continue

        print("CURRENT DATA:", sim_path)
        name, metrics = process_sim(sim_path)
        current_SHD = metrics['shd']
        current_F1 = metrics['F1']
        current_precision = metrics['precision']
        current_recall = metrics['recall']

        if not np.isnan(current_SHD):
            SHD.append(current_SHD)

        if not np.isnan(current_F1):
            F1.append(current_F1)

        if not np.isnan(current_precision):
            Precision.append(current_precision)

        if not np.isnan(current_recall):
            Recall.append(current_recall)

        print("SHD = ", current_SHD, " F1 = ", current_F1, " Precision = ", current_precision, " Recall = ", current_recall)
        print("")


    print("SHD mean:", np.mean(SHD))
    print("SHD variance:", np.var(SHD))

    print("precision mean:", np.mean(Precision))
    print("precision variance:", np.var(Precision))

    print("recall mean:", np.mean(Recall))
    print("recall variance:", np.var(Recall))

    print("F1 mean:", np.mean(F1))
    print("F1 variance:", np.var(F1))

if __name__ == '__main__':
    main()
