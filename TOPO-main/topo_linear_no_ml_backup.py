from topo_utils import (
    threshold_W, create_Z, create_new_topo, create_new_topo_greedy,
    find_idx_set_updated, gradient_l1, set_sizes_linear
)
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
import scipy.linalg as slin
from copy import copy
import random


class TOPO_linear:
    def __init__(self, score, regress, alpha=0.01, reg_lambda=0.01):
        """
        初始化 TOPO_linear 模型

        参数:
          score: 评分函数，用于计算重构误差及梯度（例如：obj_loss）
          regress: 基础回归函数（备用，默认采用内部 Ridge 回归）
          alpha: Ridge 回归正则化参数；若 alpha > 0 则采用 Ridge 回归，默认 0.01
          reg_lambda: 在评分函数中附加的 L2 正则系数，默认 0.01
        """
        super().__init__()
        self.score = score
        self.regress = regress  # 原始回归函数（备用）
        self.alpha = alpha
        self.reg_lambda = reg_lambda

    def _init_W_slice(self, idx_y, idx_x):
        """
        针对单个回归问题：用自变量 x（索引 idx_x）预测因变量 y（索引 idx_y）。
        当 alpha > 0 时，采用 Ridge 回归来稳定系数估计。
        """
        y = self.X[:, idx_y]
        x = self.X[:, idx_x]
        if self.alpha > 0:
            reg = Ridge(alpha=self.alpha, fit_intercept=False)
            reg.fit(x, y)
            w = reg.coef_
        else:
            w = self.regress(X=x, y=y)
        return w

    def _init_W(self, Z):
        """
        根据当前拓扑结构 Z 初始化权重矩阵 W。
        对于每个节点 j，使用允许的父节点（即 ~Z[:, j] 为 True）进行回归预测。
        """
        W = np.zeros((self.d, self.d))
        for j in range(self.d):
            idx = np.where(~Z[:, j])[0]
            if idx.size > 0:
                W[idx, j] = self._init_W_slice(idx_y=j, idx_x=idx)
            else:
                W[:, j] = 0
        return W

    def _h(self, W):
        """
        计算无环性约束函数 h(W) 及其梯度 G_h：
          h(W) = -log det(sI - |W|) + d*log(s)
          G_h = (sI - |W|)^{-T}
        """
        I = np.eye(self.d)
        s = 1
        M = s * I - np.abs(W)
        h = - np.linalg.slogdet(M)[1] + self.d * np.log(s)
        G_h = slin.inv(M).T
        return h, G_h

    def _update_topo_linear(self, W, topo, idx, opt=1):
        """
        根据候选拓扑更新索引 idx，生成新的权重矩阵和拓扑序列。
        """
        topo0 = copy(topo)
        W0 = np.zeros_like(W)
        i, j = idx
        i_pos, j_pos = topo.index(i), topo.index(j)

        W0[:, topo[:j_pos]] = W[:, topo[:j_pos]]
        W0[:, topo[(i_pos + 1):]] = W[:, topo[(i_pos + 1):]]
        topo0 = create_new_topo(topo=topo0, idx=idx, opt=opt)
        for k in range(j_pos, i_pos + 1):
            if len(topo0[:k]) != 0:
                W0[topo0[:k], topo0[k]] = self._init_W_slice(idx_y=topo0[k], idx_x=topo0[:k])
            else:
                W0[:, topo0[k]] = 0
        return W0, topo0

    def _score(self, X, W):
        """
        包装外部提供的评分函数，并增加 L2 正则化项：
          loss_total = loss_data + 0.5 * reg_lambda * ||W||^2
        """
        base_loss, base_grad = self.score(X, W)
        if self.reg_lambda > 0:
            reg_loss = 0.5 * self.reg_lambda * np.sum(W ** 2)
            reg_grad = self.reg_lambda * W
            return base_loss + reg_loss, base_grad + reg_grad
        else:
            return base_loss, base_grad

    def _perturb_topo(self, topo, num_swaps=1):
        """
        对当前拓扑序进行局部随机扰动（交换），num_swaps 控制交换次数。
        """
        new_topo = topo.copy()
        for _ in range(num_swaps):
            i, j = random.sample(range(len(new_topo)), 2)
            new_topo[i], new_topo[j] = new_topo[j], new_topo[i]
        return new_topo

    '''
    # 没有加入优化的代码
    def fit(self, X, topo: list, no_large_search=-1, size_small=-1, size_large=-1,
            verbose=False, max_no_improve=5, perturb_swaps=2, max_iter=100, global_no_improve=20):
        """
        在数据 X 上进行拓扑结构搜索，返回最终的权重矩阵、拓扑序列、Z 矩阵及最终损失。

        参数:
          max_no_improve: 连续局部迭代无改进时调用局部扰动的阈值
          perturb_swaps: 局部扰动时交换次数
          max_iter: 最大迭代次数
          global_no_improve: 连续无改进达到此值时进行全局重启（随机初始化拓扑）
        """
        vprint = print if verbose else lambda *a, **k: None
        self.n, self.d = X.shape
        size_small, size_large, no_large_search = set_sizes_linear(self.d, size_small, size_large, no_large_search)
        print(
            f"Parameter is automatically set up.\n size_small: {size_small}, size_large: {size_large}, no_large_search: {no_large_search}")

        self.X = X
        if not isinstance(topo, list):
            raise TypeError("topo must be a list")
        else:
            self.topo = topo

        Z = create_Z(self.topo)
        self.Z = Z
        self.W = self._init_W(self.Z)
        loss, G_loss = self._score(self.X, self.W)
        vprint(f"Initial loss: {loss}")
        h, G_h = self._h(W=self.W)
        idx_set_small, idx_set_large = find_idx_set_updated(G_h=G_h, G_loss=G_loss, Z=self.Z,
                                                            size_small=size_small, size_large=size_large)
        idx_set = list(idx_set_small)
        no_improve_counter = 0
        global_no_improve_counter = 0
        iter_count = 0
        large_space_used = 0

        while bool(idx_set) and iter_count < max_iter:
            iter_count += 1
            idx_len = len(idx_set)
            loss_collections = np.zeros(idx_len)
            for i in range(idx_len):
                W_c, topo_c = self._update_topo_linear(W=self.W, topo=self.topo, idx=idx_set[i])
                loss_c, _ = self._score(self.X, W_c)
                loss_collections[i] = loss_c

            best_candidate_loss = np.min(loss_collections)
            if loss > best_candidate_loss:
                vprint(
                    f"Iteration {iter_count}: loss {loss} > best candidate {best_candidate_loss}; updating topology.")
                self.topo = create_new_topo_greedy(self.topo, loss_collections, idx_set, loss)
                loss, _ = self._score(self.X, self._init_W(create_Z(self.topo)))
                no_improve_counter = 0
                global_no_improve_counter = 0
            else:
                no_improve_counter += 1
                global_no_improve_counter += 1
                vprint(
                    f"Iteration {iter_count}: No local improvement (count={no_improve_counter}, global={global_no_improve_counter}).")
                if no_improve_counter >= max_no_improve:
                    vprint(
                        f"Local no-improve threshold reached; applying local perturbation with {perturb_swaps} swaps.")
                    self.topo = self._perturb_topo(self.topo, num_swaps=perturb_swaps)
                    no_improve_counter = 0
                if global_no_improve_counter >= global_no_improve:
                    vprint(f"Global no-improve threshold reached; performing full random restart.")
                    self.topo = list(np.random.permutation(range(self.d)))
                    global_no_improve_counter = 0
            # 更新权重和约束矩阵
            self.Z = create_Z(self.topo)
            self.W = self._init_W(self.Z)
            loss, G_loss = self._score(self.X, self.W)
            h, G_h = self._h(W=self.W)
            idx_set_small, idx_set_large = find_idx_set_updated(G_h=G_h, G_loss=G_loss, Z=self.Z,
                                                                size_small=size_small, size_large=size_large)
            idx_set = list(idx_set_small)
        return self.W, self.topo, self.Z, loss
    
    '''

    def fit(self, X, topo: list, no_large_search=-1, size_small=-1, size_large=-1,
            verbose=False, max_iter=50, T_init=2.0, cooling_rate=0.9, global_no_improve=10, perturb_swaps=2):

        '''
        {'T_init': 1.0, 'cooling_rate': 0.98, 'max_iter': 100, 'global_no_improve': 20, 'perturb_swaps': 2,
        变差1组，变好1组，剩下不变

        {'T_init': 0.5, 'cooling_rate': 0.95, 'max_iter': 50, 'global_no_improve': 10, 'perturb_swaps': 1,
        变差2组，剩下不变

        {'T_init': 2.0, 'cooling_rate': 0.9, 'max_iter': 50, 'global_no_improve': 10, 'perturb_swaps': 2
        变好5组，变差9组，不变7组
        '''

        """
        在数据 X 上进行拓扑结构搜索，使用模拟退火策略搜索最优拓扑。

        参数:
          max_iter: 最大迭代次数
          T_init: 初始温度
          cooling_rate: 温度冷却速率（每次迭代后 T = T * cooling_rate）
          global_no_improve: 如果连续迭代无改进达到此值，则进行全局随机重启
          perturb_swaps: 当局部扰动时进行的交换次数

        返回: 最终的权重矩阵、拓扑序、Z 矩阵和最终损失
        """
        vprint = print if verbose else lambda *a, **k: None
        self.n, self.d = X.shape
        size_small, size_large, no_large_search = set_sizes_linear(self.d, size_small, size_large, no_large_search)
        print(
            f"Parameter is automatically set up.\n size_small: {size_small}, size_large: {size_large}, no_large_search: {no_large_search}")

        self.X = X
        if not isinstance(topo, list):
            raise TypeError("topo must be a list")
        else:
            self.topo = topo

        # 初始化
        self.Z = create_Z(self.topo)
        self.W = self._init_W(self.Z)
        current_loss, _ = self._score(self.X, self.W)
        vprint(f"Initial loss: {current_loss}")
        h, G_h = self._h(W=self.W)
        idx_set_small, idx_set_large = find_idx_set_updated(G_h=G_h, G_loss=None, Z=self.Z,
                                                            size_small=size_small, size_large=size_large)
        # 这里仅依据 G_h 来确定候选拓扑更新位置
        idx_set = list(idx_set_small)

        T = T_init
        global_no_improve_counter = 0
        iter_count = 0

        while bool(idx_set) and iter_count < max_iter:
            iter_count += 1
            candidate_losses = []
            candidate_topos = []
            for idx in idx_set:
                W_candidate, topo_candidate = self._update_topo_linear(W=self.W, topo=self.topo, idx=idx)
                loss_candidate, _ = self._score(self.X, W_candidate)
                candidate_losses.append(loss_candidate)
                candidate_topos.append(topo_candidate)
            candidate_losses = np.array(candidate_losses)
            best_idx = np.argmin(candidate_losses)
            best_candidate_loss = candidate_losses[best_idx]
            delta = best_candidate_loss - current_loss

            # 模拟退火接受准则
            if delta < 0:
                accept = True
                vprint(f"Iteration {iter_count}: Found improvement (delta={delta:.4f}); accepting update.")
            else:
                prob = np.exp(-delta / T) if T > 1e-8 else 0
                r = random.random()
                accept = r < prob
                vprint(f"Iteration {iter_count}: No improvement (delta={delta:.4f}, T={T:.4f}), " +
                       f"acceptance probability={prob:.4f}, random={r:.4f}; {'accept' if accept else 'reject'}.")

            if accept:
                self.topo = candidate_topos[best_idx]
                current_loss = best_candidate_loss
                global_no_improve_counter = 0
            else:
                global_no_improve_counter += 1
                # 如果未接受更新，先尝试局部扰动
                vprint(f"Iteration {iter_count}: Applying local perturbation with {perturb_swaps} swaps.")
                self.topo = self._perturb_topo(self.topo, num_swaps=perturb_swaps)
                if global_no_improve_counter >= global_no_improve:
                    vprint(
                        f"Iteration {iter_count}: Global no-improve threshold reached; performing full random restart.")
                    self.topo = list(np.random.permutation(range(self.d)))
                    global_no_improve_counter = 0

            # 更新温度
            T = T * cooling_rate

            # 更新 W, Z 和候选更新集合
            self.Z = create_Z(self.topo)
            self.W = self._init_W(self.Z)
            current_loss, _ = self._score(self.X, self.W)
            h, G_h = self._h(W=self.W)
            idx_set_small, idx_set_large = find_idx_set_updated(G_h=G_h, G_loss=None, Z=self.Z,
                                                                size_small=size_small, size_large=size_large)
            idx_set = list(idx_set_small)
            vprint(f"Iteration {iter_count}: Current loss: {current_loss:.4f}")

        return self.W, self.topo, self.Z, current_loss

    def fit_multiple(self, X, n_restarts=5, **kwargs):
        """
        对同一数据 X 进行多次随机初始化，每次随机产生一个初始拓扑，并调用 fit，
        最后返回损失最低的结果。

        参数:
          n_restarts: 重启次数
          kwargs: 传递给 fit 的其它参数
        """
        best_loss = np.inf
        best_result = None
        for i in range(n_restarts):
            init_topo = list(np.random.permutation(range(X.shape[1])))
            W, topo, Z, loss = self.fit(X, init_topo, **kwargs)
            if loss < best_loss:
                best_loss = loss
                best_result = (W, topo, Z, loss)
        return best_result

if __name__ == '__main__':
    import utils
    from timeit import default_timer as timer
    import itertools
    from sklearn.linear_model import LinearRegression, Ridge
    import numpy as np

    # 设置随机种子
    rd_int = int(np.random.randint(10000, size=1)[0])
    print(f"Random seed: {rd_int}")
    utils.set_random_seed(rd_int)

    # 模拟数据（可根据实际情况修改）
    n, d, s0 = 1000, 10, 20
    graph_type, sem_type = 'ER', 'gauss'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)

    # 定义备用的回归和评分函数
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

    # 定义要调参的参数网格
    T_init_list = [0.5, 1.0, 2.0]
    cooling_rate_list = [0.90, 0.95, 0.98]
    max_iter_list = [50, 100, 150]
    global_no_improve_list = [10, 20, 30]
    perturb_swaps_list = [1, 2, 3]

    best_params = None
    best_F1 = -1
    results = []

    print("开始网格搜索调参...")
    # 对每个参数组合进行搜索（这里使用 n_restarts=3 来降低单次结果的随机性）
    for T_init, cooling_rate, max_iter, global_no_improve, perturb_swaps in itertools.product(
            T_init_list, cooling_rate_list, max_iter_list, global_no_improve_list, perturb_swaps_list):
        print(f"\n测试参数: T_init={T_init}, cooling_rate={cooling_rate}, max_iter={max_iter}, "
              f"global_no_improve={global_no_improve}, perturb_swaps={perturb_swaps}")
        # 构造模型实例（模拟退火策略已在 fit/fit_multiple 方法中实现）
        model = TOPO_linear(regress=regress, score=score, alpha=0.01, reg_lambda=0.01)
        start = timer()
        # 采用多次重启方式
        W_est, topo_est, Z_est, final_loss = model.fit_multiple(
            X=X, n_restarts=3,
            T_init=T_init, cooling_rate=cooling_rate,
            max_iter=max_iter, global_no_improve=global_no_improve,
            perturb_swaps=perturb_swaps, verbose=False)
        end = timer()
        # 计算恢复网络的指标，这里仅关注 precision、F1、SHD 和 recall
        metrics = utils.count_accuracy(B_true, (np.abs(Z_est - 1) != 0))  # 此处 threshold_W(W) 可替换为适当阈值处理
        F1 = metrics.get("F1", 0)
        results.append({
            "T_init": T_init,
            "cooling_rate": cooling_rate,
            "max_iter": max_iter,
            "global_no_improve": global_no_improve,
            "perturb_swaps": perturb_swaps,
            "metrics": metrics,
            "time": end - start
        })
        print(f"结果指标: {metrics}, 耗时: {end - start:.4f}s")
        if F1 > best_F1:
            best_F1 = F1
            best_params = (T_init, cooling_rate, max_iter, global_no_improve, perturb_swaps)

    print("\n调参结束。")
    print("最佳参数组合:", best_params, "对应 F1 =", best_F1)
    print("\n所有参数组合结果：")
    for res in results:
        print(res)
