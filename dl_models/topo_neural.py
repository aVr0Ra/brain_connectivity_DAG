import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from torch.utils.data import DataLoader, TensorDataset
import math


class GNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, input_dim)
        self._initialize_weights()

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)  # 增加dropout
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.3, training=self.training)  # 增加dropout
        return torch.sigmoid(self.fc(x))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding, dilation=dilation)
        self.norm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)  # 增加dropout

    def forward(self, x):
        return self.dropout(F.relu(self.norm(self.conv(x))))


class TCNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.2):
        super(TCNModel, self).__init__()
        self.input_dim = input_dim

        # 减少模型复杂度
        hidden_dim = min(hidden_dim, input_dim * 2)
        num_layers = min(num_layers, 2)  # 限制层数

        # 初始投影层
        self.input_proj = nn.Conv1d(input_dim, hidden_dim, 1)

        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = hidden_dim
            layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels, hidden_dim, kernel_size=3,
                              padding=(3 - 1) * dilation, dilation=dilation),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
        self.tcn = nn.Sequential(*layers)

        # 简化输出层结构
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim * input_dim)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 减少噪声添加
        if self.training:
            x = x + torch.randn_like(x) * 0.005  # 降低噪声级别

        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = self.tcn(x)
        x = x.mean(dim=2)  # 全局平均池化
        x = F.dropout(F.relu(self.fc1(x)), p=0.2, training=self.training)
        x = self.fc2(x)
        x = torch.sigmoid(x.view(-1, self.input_dim, self.input_dim))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_layers):
        super(TransformerModel, self).__init__()
        self.input_dim = input_dim

        # 减少复杂度
        hidden_dim = min(hidden_dim, input_dim * 2)
        num_layers = min(num_layers, 2)  # 减少层数
        nhead = min(nhead, 4)  # 限制注意力头数

        # 确保hidden_dim能被nhead整除
        hidden_dim = (hidden_dim // nhead) * nhead

        self.pos_encoder = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 2,  # 减少前馈网络大小
            dropout=0.2,  # 增加dropout
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 简化输出层
        self.fc = nn.Linear(hidden_dim, input_dim * input_dim)

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # 全局平均池化
        x = self.fc(x)
        x = torch.sigmoid(x.view(-1, self.input_dim, self.input_dim))
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class DeepDAGModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.3):  # 增加默认dropout
        super(DeepDAGModel, self).__init__()
        self.input_dim = input_dim

        # 减少模型复杂度
        hidden_dim = min(hidden_dim, input_dim * 2)

        self.dropout = nn.Dropout(dropout)

        # 简化网络结构
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim * input_dim)

        # 添加L2正则化
        self.l2_reg = 0.001

        # 初始化权重
        self._initialize_weights()

    def forward(self, x):
        # 减少训练中的噪声
        if self.training:
            x = x + torch.randn_like(x) * 0.005  # 降低噪声级别

        x = torch.mean(x, dim=1)  # 在时间维度上平均
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        x = torch.sigmoid(x.view(-1, self.input_dim, self.input_dim))

        # 添加L2正则化
        l2_loss = 0
        for param in self.parameters():
            l2_loss += torch.norm(param, 2)

        if self.training:
            # 在训练中存储L2损失供优化器使用
            self.l2_loss = self.l2_reg * l2_loss

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class NeuralDAGTrainer:
    def __init__(self, model_type, input_dim, hidden_dim=64, num_layers=2, nhead=4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_type = model_type
        self.input_dim = input_dim

        # 调整隐藏层大小和层数
        hidden_dim = min(hidden_dim, input_dim * 4)  # 限制隐藏层大小
        num_layers = min(num_layers, 2)  # 限制层数

        if model_type == 'transformer':
            hidden_dim = (hidden_dim // nhead) * nhead

        if model_type == 'tcn':
            self.model = TCNModel(input_dim, hidden_dim, num_layers, dropout=0.2).to(self.device)
        elif model_type == 'transformer':
            self.model = TransformerModel(input_dim, hidden_dim, nhead, num_layers).to(self.device)
        elif model_type == 'deep_dag':
            self.model = DeepDAGModel(input_dim, hidden_dim, dropout=0.3).to(self.device)

        # 增加L2正则化强度
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.005)

        # 改进学习率调度器
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-5
        )

        self.criterion = nn.BCELoss()

        # 添加权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.model.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def train(self, train_data, train_net, val_data, val_net,
              epochs=50, batch_size=32, patience=5):
        X_train, y_train = self.prepare_data(train_data, train_net, is_training=True)
        X_val, y_val = self.prepare_data(val_data, val_net, is_training=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        min_delta = 0.001  # 最小改进阈值

        for epoch in range(epochs):
            self.model.train()
            total_train_loss = 0
            n_train_batches = 0

            # 使用DataLoader进行批次处理
            train_dataset = TensorDataset(X_train, y_train.expand(X_train.shape[0], self.input_dim, self.input_dim))
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()
                pred = self.model(batch_x)

                # 降低L1正则化强度
                l1_lambda = 0.0005
                l1_reg = torch.norm(pred, 1)

                loss = self.criterion(pred, batch_y)
                loss = loss + l1_lambda * l1_reg

                # 如果模型有L2损失属性则添加
                if hasattr(self.model, 'l2_loss'):
                    loss = loss + self.model.l2_loss

                loss.backward()
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                total_train_loss += loss.item()
                n_train_batches += 1

            # 验证
            self.model.eval()
            val_loss = 0
            n_val_batches = 0

            # 使用DataLoader进行批次处理
            val_dataset = TensorDataset(X_val, y_val.expand(X_val.shape[0], self.input_dim, self.input_dim))
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    pred = self.model(batch_x)
                    batch_loss = self.criterion(pred, batch_y)
                    val_loss += batch_loss.item()
                    n_val_batches += 1

            val_loss /= n_val_batches

            # 学习率调度
            self.scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                print(f'Epoch {epoch + 1}:')
                print(f'  Train Loss: {total_train_loss / n_train_batches:.4f}')
                print(f'  Val Loss: {val_loss:.4f}')
                print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # 早停机制 - 考虑最小改进阈值
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch + 1} (no improvement for {patience} epochs)')
                    break

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Restored best model with validation loss: {best_val_loss:.4f}")

    def prepare_data(self, ts_data, net_data, is_training=False):
        # 使用更稳健的数据标准化
        scaler = RobustScaler()  # 使用RobustScaler处理异常值
        ts_reshaped = ts_data.reshape(-1, ts_data.shape[-1])
        ts_scaled = scaler.fit_transform(ts_reshaped)
        ts_tensor = torch.FloatTensor(ts_scaled.reshape(ts_data.shape))

        # 数据增强 (仅在训练时)
        if is_training:
            # 添加弱噪声
            ts_tensor = ts_tensor + torch.randn_like(ts_tensor) * 0.01

            # 随机遮蔽一些值
            mask_prob = 0.05
            mask = (torch.rand_like(ts_tensor) > mask_prob).float()
            ts_tensor = ts_tensor * mask

            # 随机时间移位 (适用于时间序列)
            if ts_tensor.shape[1] > 5:  # 确保有足够的时间点
                shift = torch.randint(-2, 3, (ts_tensor.shape[0], 1, ts_tensor.shape[2]))
                for i in range(ts_tensor.shape[0]):
                    if shift[i, 0, 0] != 0:
                        ts_tensor[i] = torch.roll(ts_tensor[i], shifts=int(shift[i, 0, 0]), dims=0)

        # 准备目标邻接矩阵 - 使用动态阈值
        net_mean = np.mean(net_data, axis=0)

        # 动态阈值计算
        if np.sum(net_mean > 0) > 0:
            # 使用正值的中位数作为阈值
            positive_values = net_mean[net_mean > 0]
            if len(positive_values) > 0:
                threshold = np.median(positive_values) * 0.5  # 使用中位数的一半作为阈值
            else:
                threshold = 0.1
        else:
            threshold = 0.1

        net_binary = (net_mean > threshold).astype(float)
        net_tensor = torch.FloatTensor(net_binary)

        return ts_tensor.to(self.device), net_tensor.to(self.device)

    def predict(self, ts_data):
        self.model.eval()
        with torch.no_grad():
            # 使用相同的标准化方法
            scaler = RobustScaler()
            ts_reshaped = ts_data.reshape(-1, ts_data.shape[-1])
            ts_scaled = scaler.fit_transform(ts_reshaped)
            X = torch.FloatTensor(ts_scaled.reshape(ts_data.shape)).to(self.device)

            # 使用DataLoader进行批处理预测
            batch_size = 32
            dataset = TensorDataset(X)
            loader = DataLoader(dataset, batch_size=batch_size)

            predictions = []
            for (batch_x,) in loader:
                pred = self.model(batch_x)
                predictions.append(pred.cpu().numpy())

            # 将所有批次预测合并
            all_predictions = np.concatenate(predictions, axis=0)

            # 添加集成预测 - 使用不同阈值的平均值
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            ensemble_preds = []

            for thresh in thresholds:
                binary_pred = (all_predictions > thresh).astype(float)
                ensemble_preds.append(binary_pred)

            # 平均所有阈值的预测
            ensemble_result = np.mean(ensemble_preds, axis=0)

            # 取最终预测的平均值
            final_pred = np.mean(ensemble_result, axis=0)

            return final_pred

    def cross_validate(self, data, net_data, n_folds=5):
        """进行交叉验证以找到最佳超参数"""
        n_samples = data.shape[0]
        fold_size = n_samples // n_folds

        val_losses = []

        for fold in range(n_folds):
            val_indices = list(range(fold * fold_size, (fold + 1) * fold_size))
            train_indices = list(set(range(n_samples)) - set(val_indices))

            train_data_fold = data[train_indices]
            train_net_fold = net_data[train_indices]
            val_data_fold = data[val_indices]
            val_net_fold = net_data[val_indices]

            # 训练验证拆分
            X_train, y_train = self.prepare_data(train_data_fold, train_net_fold, is_training=True)
            X_val, y_val = self.prepare_data(val_data_fold, val_net_fold, is_training=False)

            # 重置模型参数
            self._initialize_weights()

            # 训练10个epoch用于快速验证
            self.train(train_data_fold, train_net_fold, val_data_fold, val_net_fold,
                       epochs=10, batch_size=32, patience=5)

            # 计算验证损失
            self.model.eval()
            with torch.no_grad():
                val_pred = self.model(X_val)
                val_loss = self.criterion(val_pred, y_val.expand(val_pred.shape))
                val_losses.append(val_loss.item())

        return np.mean(val_losses)