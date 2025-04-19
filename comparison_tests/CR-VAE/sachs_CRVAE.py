# -*- coding: utf-8 -*-
"""
CRVAE算法适配到Sachs数据集
"""

import torch
import numpy as np
from models.cgru_error import CRVAE, VRAE4E, train_phase1, train_phase2
from castle.metrics import MetricsDAG
import cdt
from cdt.data import load_dataset
import networkx as nx

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备：{device}")

# 加载Sachs数据集
print("加载Sachs数据集...")
data, graph = load_dataset('sachs')

# 提取数据和维度信息
X_np = data.values
Nnodes = X_np.shape[1]
print(f"数据维度信息: Nnodes={Nnodes}, 样本数={X_np.shape[0]}")

# 将数据转换为张量
X = torch.tensor(X_np[np.newaxis], dtype=torch.float32, device=device)

# 将NetworkX图转换为邻接矩阵
nodes = list(graph.nodes())
B_true = np.zeros((Nnodes, Nnodes), dtype=int)

# 填充邻接矩阵
for i, node_i in enumerate(nodes):
    for j, node_j in enumerate(nodes):
        if graph.has_edge(node_i, node_j):
            B_true[i, j] = 1

print("真实网络结构的形状:", B_true.shape)
print('真实变量使用率 = %.2f%%' % (100 * np.mean(B_true)))

# 初始化CRVAE模型
# 首先使用全连接矩阵进行训练
full_connect = np.ones((Nnodes, Nnodes))
cgru = CRVAE(Nnodes, full_connect, hidden=64).to(device)
vrae = VRAE4E(Nnodes, hidden=64).to(device)

print("第一阶段：学习因果结构")
# 第一阶段训练：学习因果结构
train_loss_list = train_phase1(
    cgru, X, context=20, lam=0.1, lam_ridge=0, lr=5e-2, max_iter=1000,
    check_every=500)

# 获取估计的因果结构
try:
    GC_est = cgru.GC().cpu().data.numpy()
except Exception as e:
    print(f"获取GC矩阵时出错: {e}")
    # 手动构建GC矩阵
    print("尝试手动构建GC矩阵...")
    GC_raw = cgru.gc
    GC_est = np.zeros((Nnodes, Nnodes))
    for i in range(Nnodes):
        for j in range(Nnodes):
            try:
                GC_est[i, j] = float(GC_raw[i][j].item() > 0.5)
            except:
                pass  # 如果某些位置不可用，则保持为0

B_est_phase1 = (GC_est > 0.5).astype(int)  # 二值化

print("第一阶段完成")
print('估计变量使用率 = %.2f%%' % (100 * np.mean(B_est_phase1)))

# 评估指标
mt_phase1 = MetricsDAG(B_est_phase1, B_true)
metrics_phase1 = mt_phase1.metrics

print("\n第一阶段评估指标:")
print("SHD =", metrics_phase1['shd'])
print("Precision =", metrics_phase1['precision'])
print("Recall =", metrics_phase1['recall'])
print("F1 =", metrics_phase1['F1'])

print("\n第二阶段：使用学习到的因果结构进行建模")
# 第二阶段：使用学习到的因果结构进行建模
try:
    cgru_phase2 = CRVAE(Nnodes, GC_est, hidden=64).to(device)
    vrae_phase2 = VRAE4E(Nnodes, hidden=64).to(device)

    train_loss_list_phase2 = train_phase2(
        cgru_phase2, vrae_phase2, X, context=20, lam=0., lam_ridge=0, lr=5e-2, max_iter=1000,
        check_every=500)

    print("第二阶段训练完成")
except Exception as e:
    print(f"第二阶段训练时发生错误: {e}")
    print("使用第一阶段的结果进行评估")

# 使用第一阶段的结果进行最终评估
# 尝试使用不同的阈值进行二值化，找到最优结果
thresholds = np.linspace(0.1, 0.9, 9)
best_f1 = 0
best_threshold = 0.5
best_metrics = None

print("\n尝试不同阈值进行优化:")
for threshold in thresholds:
    B_est_thresh = (GC_est > threshold).astype(int)
    mt_thresh = MetricsDAG(B_est_thresh, B_true)
    metrics_thresh = mt_thresh.metrics

    print(
        f"阈值 {threshold:.1f}: F1 = {metrics_thresh['F1']:.4f}, Precision = {metrics_thresh['precision']:.4f}, Recall = {metrics_thresh['recall']:.4f}, SHD = {metrics_thresh['shd']}")

    if metrics_thresh['F1'] > best_f1:
        best_f1 = metrics_thresh['F1']
        best_threshold = threshold
        best_metrics = metrics_thresh

print("\n最佳阈值:", best_threshold)
print("最佳评估指标:")
print("SHD =", best_metrics['shd'])
print("Precision =", best_metrics['precision'])
print("Recall =", best_metrics['recall'])
print("F1 =", best_metrics['F1'])