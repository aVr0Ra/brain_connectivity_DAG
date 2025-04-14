from topo_utils import threshold_W, create_Z, create_new_topo, create_new_topo_greedy, find_idx_set_updated, \
    gradient_l1, set_sizes_linear
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.special import expit as sigmoid
import scipy.linalg as slin
from copy import copy
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split


class NeuralNetRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, dropout_rate=0.1):
        """简单的神经网络回归器"""
        super(NeuralNetRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.model(x).squeeze()


class TOPO_linear:
    def __init__(self, score, regress, use_nn=False, hidden_dim=16, lr=0.005, epochs=100,
                 l2_reg=0.001, alpha=0.3):
        super().__init__()
        self.score = score
        self.regress = regress
        self.use_nn = use_nn
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.alpha = alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _nn_regress(self, X, y):
        """使用简化的神经网络方法代替复杂的梯度计算"""
        if X.shape[0] == 0 or X.shape[1] == 0:
            return np.zeros(X.shape[1])

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)

        # 创建模型
        adjusted_hidden_dim = min(self.hidden_dim, max(4, X.shape[1] * 2))
        model = NeuralNetRegressor(X.shape[1], adjusted_hidden_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.l2_reg)
        criterion = nn.MSELoss()

        # 训练模型
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

        # 使用线性回归近似获取系数
        linear_model = LinearRegression(fit_intercept=False)
        with torch.no_grad():
            model.eval()
            y_pred = model(X_tensor).cpu().numpy()

        # 使用神经网络的预测作为目标，训练线性模型
        linear_model.fit(X, y_pred)
        coeffs = linear_model.coef_

        # 清理CUDA内存
        torch.cuda.empty_cache()

        return coeffs

    def _init_W_slice(self, idx_y, idx_x):
        """初始化W矩阵的一部分，可以用神经网络或线性回归"""
        y = self.X[:, idx_y]
        x = self.X[:, idx_x]

        # 对于小数据集或无特征的情况，回退到线性回归
        if self.use_nn and x.shape[1] > 0 and x.shape[0] > 10:
            w = self._nn_regress(X=x, y=y)
        else:
            w = self.regress(X=x, y=y)

        return w

    def _init_W(self, Z):
        """初始化整个W矩阵"""
        W = np.zeros((self.d, self.d))
        for j in range(self.d):
            if (~Z[:, j]).any():
                if self.use_nn and np.sum(~Z[:, j]) > 0 and self.X.shape[0] > 10:
                    W[~Z[:, j], j] = self._nn_regress(X=self.X[:, ~Z[:, j]], y=self.X[:, j])
                else:
                    W[~Z[:, j], j] = self.regress(X=self.X[:, ~Z[:, j]], y=self.X[:, j])
            else:
                W[:, j] = 0
        return W

    def _h(self, W):
        """计算非循环约束的值和梯度"""
        I = np.eye(self.d)
        s = 1
        M = s * I - np.abs(W)
        h = - np.linalg.slogdet(M)[1] + self.d * np.log(s)
        G_h = slin.inv(M).T

        return h, G_h

    def _update_topo_linear(self, W, topo, idx, opt=1):
        """基于拓扑交换更新W矩阵"""
        topo0 = copy(topo)
        W0 = np.zeros_like(W)
        i, j = idx
        i_pos, j_pos = topo.index(i), topo.index(j)

        # 复制不变的部分
        W0[:, topo[:j_pos]] = W[:, topo[:j_pos]]
        W0[:, topo[(i_pos + 1):]] = W[:, topo[(i_pos + 1):]]

        # 创建新的拓扑
        topo0 = create_new_topo(topo=topo0, idx=idx, opt=opt)

        # 更新变化的部分
        for k in range(j_pos, i_pos + 1):
            if len(topo0[:k]) != 0:
                W0[topo0[:k], topo0[k]] = self._init_W_slice(idx_y=topo0[k], idx_x=topo0[:k])
            else:
                W0[:, topo0[k]] = 0
        return W0, topo0

    def fit(self, X, topo: list, no_large_search=-1, size_small=-1, size_large=-1, verbose=False, max_iter=20):
        """拟合模型"""
        vprint = print if verbose else lambda *a, **k: None
        self.n, self.d = X.shape
        size_small, size_large, no_large_search = set_sizes_linear(self.d, size_small, size_large, no_large_search)
        print(
            f"Parameter is automatically set up.\n size_small: {size_small}, size_large: {size_large}, no_large_search: {no_large_search}")
        print(f"Using neural networks: {self.use_nn}")

        self.X = X
        iter_count = 0
        large_space_used = 0
        if not isinstance(topo, list):
            raise TypeError
        else:
            self.topo = topo

        # 自动决定是否使用神经网络
        if self.use_nn and X.shape[0] < 1000:
            print("Limited data available, using linear model instead of neural network")
            self.use_nn = False

        # 初始化
        Z = create_Z(self.topo)
        self.Z = Z
        self.W = self._init_W(self.Z)
        loss, G_loss = self.score(X=self.X, W=self.W)
        vprint(f"Initial loss: {loss}")
        h, G_h = self._h(W=self.W)
        idx_set_small, idx_set_large = find_idx_set_updated(G_h=G_h, G_loss=G_loss, Z=self.Z, size_small=size_small,
                                                            size_large=size_large)
        idx_set = list(idx_set_small)

        # 主优化循环
        while bool(idx_set) and iter_count < max_iter:
            iter_count += 1
            idx_len = len(idx_set)
            loss_collections = np.zeros(idx_len)

            # 评估小集合中所有可能的交换
            for i in range(idx_len):
                W_c, topo_c = self._update_topo_linear(W=self.W, topo=self.topo, idx=idx_set[i])
                loss_c, _ = self.score(X=self.X, W=W_c)
                loss_collections[i] = loss_c

            # 如果在小集合中找到了更好的配置，更新
            if np.any(loss > np.min(loss_collections)):
                vprint(
                    f"Iteration {iter_count}: loss {loss} -> find better loss {np.min(loss_collections)} in small space")
                self.topo = create_new_topo_greedy(self.topo, loss_collections, idx_set, loss)
            else:
                # 如果没有，尝试大集合（如果我们还有尝试次数）
                if large_space_used < no_large_search:
                    vprint(f"Iteration {iter_count}: loss {loss} -> cannot find better loss in small space")
                    vprint(f"Using larger search space for {large_space_used + 1} times")
                    idx_set = list(set(idx_set_large) - set(idx_set_small))
                    idx_len = len(idx_set)
                    loss_collections = np.zeros(idx_len)

                    # 评估大集合中所有可能的交换
                    for i in range(idx_len):
                        W_c, topo_c = self._update_topo_linear(W=self.W, topo=self.topo, idx=idx_set[i])
                        loss_c, _ = self.score(X=self.X, W=W_c)
                        loss_collections[i] = loss_c

                    # 更新找到了更好的配置的情况
                    if np.any(loss > loss_collections):
                        large_space_used += 1
                        self.topo = create_new_topo_greedy(self.topo, loss_collections, idx_set, loss)
                        vprint(f"Iteration {iter_count}: loss {loss} -> find better loss in large space")
                    else:
                        vprint("Using larger search space, but we cannot find better loss")
                        break
                else:
                    vprint(f"We reach the number of chances to search large space: {no_large_search}")
                    break

            # 用新的拓扑更新模型
            self.Z = create_Z(self.topo)
            self.W = self._init_W(self.Z)
            loss, G_loss = self.score(X=self.X, W=self.W)
            h, G_h = self._h(W=self.W)
            idx_set_small, idx_set_large = find_idx_set_updated(G_h=G_h, G_loss=G_loss, Z=self.Z, size_small=size_small,
                                                                size_large=size_large)
            idx_set = list(idx_set_small)

        # 应用最终阈值，获得稀疏解
        W_thresholded = threshold_W(self.W.copy(), threshold=self.alpha)
        return W_thresholded, self.topo, self.Z, loss


if __name__ == '__main__':
    import utils
    from timeit import default_timer as timer

    rd_int = int(np.random.randint(10000, size=1)[0])
    print(f"random seed: {rd_int}")

    utils.set_random_seed(rd_int)
    n, d, s0 = 1000, 10, 20
    graph_type, sem_type = 'ER', 'gauss'
    verbose = False

    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)


    ## Linear Model
    def regress(X, y):
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X=X, y=y)
        return reg.coef_


    def score(X, W):
        M = X @ W
        R = X - M
        loss = 0.5 / X.shape[0] * (R ** 2).sum()
        G_loss = - 1.0 / X.shape[0] * X.T @ R

        return loss, G_loss


    # Test standard linear model
    model = TOPO_linear(regress=regress, score=score)
    topo_init = list(np.random.permutation(range(d)))
    start = timer()
    W_est, _, _, _ = model.fit(X=X, topo=topo_init, verbose=verbose)
    end = timer()
    acc = utils.count_accuracy(B_true, threshold_W(W=W_est) != 0)
    print("Linear model results:")
    print(acc)
    print(f'Linear time: {end - start:.4f}s')

    # Test neural model with simplified approach
    model_nn = TOPO_linear(regress=regress, score=score, use_nn=True, hidden_dim=16,
                           epochs=50)
    topo_init = list(np.random.permutation(range(d)))
    start = timer()
    W_est_nn, _, _, _ = model_nn.fit(X=X, topo=topo_init, verbose=verbose)
    end = timer()
    acc_nn = utils.count_accuracy(B_true, threshold_W(W=W_est_nn) != 0)
    print("\nNeural model results:")
    print(acc_nn)
    print(f'Neural time: {end - start:.4f}s')