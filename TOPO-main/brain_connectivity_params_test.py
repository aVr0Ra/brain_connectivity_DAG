import numpy as np
from scipy.io import loadmat
import os
from timeit import default_timer as timer
import itertools
import random

# 导入相关模块
import utils
from topo_linear import TOPO_linear  # 请确保 TOPO_linear 为模拟退火版本且已支持 fit_multiple
from castle import MetricsDAG
# 设置随机种子
rd_int = int(np.random.randint(10000, size=1)[0])
print(f"Random seed: {rd_int}")
utils.set_random_seed(rd_int)

# 只使用 sim1.mat 数据
data_path = os.path.join("..", "datasets", "sims", "sim14.mat")
data = loadmat(data_path)
print("Loaded data from:", data_path)

# 根据原来的处理流程
net = data['net']       # shape: (Nsubjects, Nnodes, Nnodes)
ts = data['ts']         # shape: (Ntimepoints, Nnodes)
Nnodes = int(data['Nnodes'][0][0])
Nsubjects = int(data['Nsubjects'][0][0])
Ntimepoints = int(data['Ntimepoints'][0][0])

print("Original ts shape:", ts.shape)
ts_all = ts.reshape(Nsubjects, Ntimepoints, Nnodes)
print("Reshaped ts_all shape (subjects, timepoints, nodes):", ts_all.shape)
X = ts_all.reshape(-1, Nnodes)
print("Combined X shape:", X.shape)
B_true_linear = (np.mean(net, axis=0) != 0).astype(int)
print("B_true_linear shape:", B_true_linear.shape)

# 定义备用回归和评分函数
from sklearn.linear_model import LinearRegression, Ridge

def regress(X, y):
    reg = LinearRegression(fit_intercept=False)
    reg.fit(X=X, y=y)
    return reg.coef_

def score(X, W):
    M = X @ W
    R = X - M
    loss = 0.5 / X.shape[0] * np.sum(R ** 2)
    G_loss = -1.0 / X.shape[0] * (X.T @ R)
    return loss, G_loss

# 定义调参的参数网格（可根据需要调整范围）
T_init_list = [0.5, 1.0, 2.0]
cooling_rate_list = [0.9, 0.95, 0.98]
max_iter_list = [50, 100]
global_no_improve_list = [10, 20]
perturb_swaps_list = [1, 2]

best_params = None
best_F1 = -1
results = []

print("开始网格搜索调参...")

for T_init, cooling_rate, max_iter, global_no_improve, perturb_swaps in itertools.product(
        T_init_list, cooling_rate_list, max_iter_list, global_no_improve_list, perturb_swaps_list):
    print(f"\n测试参数: T_init={T_init}, cooling_rate={cooling_rate}, max_iter={max_iter}, "
          f"global_no_improve={global_no_improve}, perturb_swaps={perturb_swaps}")
    model = TOPO_linear(regress=regress, score=score, alpha=0.01, reg_lambda=0.01)
    # 使用 fit_multiple 进行多次随机初始化（n_restarts=3）
    start = timer()
    W_est, topo_est, Z_est, final_loss = model.fit_multiple(
        X=X, n_restarts=3,
        T_init=T_init, cooling_rate=cooling_rate,
        max_iter=max_iter, global_no_improve=global_no_improve,
        perturb_swaps=perturb_swaps, verbose=False
    )
    end = timer()
    # 对于评价，这里采用 threshold_W 对 W_est 进行二值化（阈值可调整），计算指标
    from topo_utils import threshold_W
    B_est = (threshold_W(W=W_est, threshold=0.3) != 0).astype(int)

    # B_est = (threshold_W(W=W_linear) != 0).astype(int)
    B_true_linear = np.array(B_true_linear).astype(int)
    # print("Estimated B (B_est):")
    # print(B_est)
    # print("True B (B_true_linear):")
    # print(B_true_linear)
    met = MetricsDAG(B_est, B_true_linear)
    print("Metrics:", met.metrics)

    # metrics = utils.count_accuracy(B_true_linear, B_est)
    F1 = met.metrics.get('F1')
    results.append({
        "T_init": T_init,
        "cooling_rate": cooling_rate,
        "max_iter": max_iter,
        "global_no_improve": global_no_improve,
        "perturb_swaps": perturb_swaps,
        "metrics": met.metrics,
        "time": end - start
    })
    print(f"参数组合结果: {met.metrics}, 耗时: {end - start:.4f}s")
    if F1 > best_F1:
        best_F1 = F1
        best_params = (T_init, cooling_rate, max_iter, global_no_improve, perturb_swaps)

print("\n调参结束。")
print("最佳参数组合:", best_params, "对应 F1 =", best_F1)
print("\n所有参数组合结果：")
for res in results:
    print(res)
