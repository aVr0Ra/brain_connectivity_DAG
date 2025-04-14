# brain_connectivity.py (更新版)
from scipy.io import loadmat
import numpy as np
from topo_linear import TOPO_Enhanced, threshold_W
from castle import MetricsDAG
import utils


def process_dataset(dataset_path):
    data = loadmat(dataset_path)
    net = data['net']  # 真实连接矩阵 (Nsubjects, Nnodes, Nnodes)
    ts = data['ts']  # 时间序列 (Nsubjects*Ntimepoints, Nnodes)

    Nnodes = int(data['Nnodes'][0][0])
    Nsubjects = int(data['Nsubjects'][0][0])
    Ntimepoints = int(data['Ntimepoints'][0][0])

    if Nnodes != 5:
        return None, None

    # 重塑为受试者维度
    ts_all = ts.reshape(Nsubjects, Ntimepoints, Nnodes)
    X_combined = ts_all.reshape(-1, Nnodes)

    # 初始化增强模型
    model = TOPO_Enhanced(dims=[Nnodes, 64, 1], alpha=0.01, reg_lambda=0.01)
    W_est, _ = model.fit(X_combined, n_restarts=3)

    # 生成预测矩阵
    B_est = (threshold_W(W_est) != 0).astype(int)

    # 真实值处理
    B_true = (np.mean(net, axis=0) != 0).astype(int)

    # 计算指标
    met = MetricsDAG(B_est, B_true)
    print("Metrics:", met.metrics)

    return B_est, B_true


if __name__ == '__main__':
    for dataset in range(1, 29):
        print(f"\nProcessing dataset: sim{dataset}")
        B_est, B_true = process_dataset(f'../datasets/sims/sim{dataset}.mat')

        if B_est is None and B_true is None:
            print("Nnode != 5, continue to save time")

        # 可视化对比
        print("Predicted Network:")
        print(B_est)
        print("True Network:")
        print(B_true)