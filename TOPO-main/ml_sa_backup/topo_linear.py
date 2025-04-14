# topo_linear_enhanced.py (新增文件)
import torch
import torch.nn as nn
import numpy as np
from topo_utils import create_Z, threshold_W
from topo_linear_backup import TOPO_linear


class NeuroLayer(nn.Module):
    """神经网络增强层"""

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class TOPO_Enhanced(TOPO_linear):
    """增强版拓扑优化模型，结合神经网络"""

    def __init__(self, dims, alpha=0.01, reg_lambda=0.01):
        super().__init__(score=self.neuro_score, regress=self.neuro_regress,
                         alpha=alpha, reg_lambda=reg_lambda)
        self.neuro_layer = NeuroLayer(dims[0], dims[1])

    def neuro_regress(self, X, y):
        """神经网络回归替代线性回归"""
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        optimizer = torch.optim.Adam(self.neuro_layer.parameters(), lr=0.01)
        for _ in range(100):
            pred = self.neuro_layer(X_tensor).squeeze()
            loss = torch.mean((pred - y_tensor) ** 2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return self.neuro_layer[-1].weight.detach().numpy().flatten()

    def neuro_score(self, X, W):
        """带神经网络增强的评分函数"""
        # 传统线性部分
        M = X @ W
        R = X - M
        base_loss = 0.5 / X.shape[0] * np.sum(R ** 2)

        # 神经网络增强
        X_tensor = torch.tensor(X, dtype=torch.float32)
        neuro_out = self.neuro_layer(X_tensor).detach().numpy()
        neuro_loss = 0.1 * np.mean((neuro_out - X) ** 2)  # 调节系数

        return base_loss + neuro_loss, -1.0 / X.shape[0] * X.T @ R

    def fit(self, X, n_restarts=3, **kwargs):
        best_loss = np.inf
        best_W = None
        for _ in range(n_restarts):
            init_topo = list(np.random.permutation(X.shape[1]))
            W, _, _, loss = super().fit(X, init_topo,
                                        T_init=2.0, cooling_rate=0.9,
                                        max_iter=50, **kwargs)
            if loss < best_loss:
                best_loss = loss
                best_W = W
        return best_W, best_loss