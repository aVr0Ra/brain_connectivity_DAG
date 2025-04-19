#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import castle.algorithms
from castle.common import GraphDAG
from castle.metrics import MetricsDAG
import cdt
from cdt.data import load_dataset
import networkx as nx


def process_sachs_data():
    # 加载Sachs数据集
    data, graph = load_dataset('sachs')

    # 特征矩阵 X
    X = data.values

    # 将NetworkX图转换为邻接矩阵
    nodes = list(graph.nodes())
    n = len(nodes)
    B_true = np.zeros((n, n), dtype=int)

    # 填充邻接矩阵
    for i, node_i in enumerate(nodes):
        for j, node_j in enumerate(nodes):
            if graph.has_edge(node_i, node_j):
                B_true[i, j] = 1

    # 用PC算法学习结构
    model = castle.algorithms.PC(alpha=0.2)

    try:
        model.learn(X)
    except ValueError as e:
        print(f"⚠️ PC算法在Sachs数据集上失败：{e}")
        # 返回一个全NaN的metrics，方便后续过滤
        return {
            'shd': np.nan,
            'precision': np.nan,
            'recall': np.nan,
            'F1': np.nan
        }

    B_est = model.causal_matrix.astype(int)

    print(B_est)
    print(B_true)

    # 评价指标
    mt = MetricsDAG(B_est, B_true)
    return mt.metrics


def main():
    print("开始在Sachs数据集上进行结构学习：\n")

    metrics = process_sachs_data()

    current_SHD = metrics['shd']
    current_F1 = metrics['F1']
    current_precision = metrics['precision']
    current_recall = metrics['recall']

    print("Sachs数据集结果:")
    print("SHD = ", current_SHD, " F1 = ", current_F1, " Precision = ", current_precision, " Recall = ", current_recall)
    print("")

    if not np.isnan(current_SHD):
        print("SHD:", current_SHD)

    if not np.isnan(current_F1):
        print("F1:", current_F1)

    if not np.isnan(current_precision):
        print("precision:", current_precision)

    if not np.isnan(current_recall):
        print("recall:", current_recall)


if __name__ == '__main__':
    main()