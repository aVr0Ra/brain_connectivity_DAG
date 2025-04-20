# hybird_models.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from castle import MetricsDAG
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LinearRegression
from timeit import default_timer as timer

from topo_linear import TOPO_linear, threshold_W
from topo_neural import TCNModel, TransformerModel, DeepDAGModel, GNNModel, NeuralDAGTrainer
from topo_utils import create_Z, set_sizes_linear


class HybridDAGModel:
    def __init__(self, neural_model_type, input_dim, hidden_dim=64, nhead=4, num_layers=2):
        self.neural_model_type = neural_model_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.nhead = nhead
        self.num_layers = num_layers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize neural trainer
        self.neural_trainer = NeuralDAGTrainer(
            model_type=neural_model_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=num_layers
        )

        # Topological model will be initialized during training
        self.topo_model = None

    def train_topo(self, X, initial_W=None):
        """Train topological model using TOPO_linear"""

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

        n, d = X.shape

        # 修复维度不匹配问题 - 如果特征维度不等于节点数
        if d != self.input_dim:
            # 假设特征是按节点分组的，每组有特定数量的特征
            if d % self.input_dim == 0:
                features_per_node = d // self.input_dim
                X_reshaped = np.zeros((n, self.input_dim))

                # 对每个节点的特征取平均，转换回节点级别的特征
                for i in range(self.input_dim):
                    start_idx = i * features_per_node
                    end_idx = (i + 1) * features_per_node
                    X_reshaped[:, i] = np.mean(X[:, start_idx:end_idx], axis=1)

                X = X_reshaped
                d = self.input_dim
                print(f"特征维度已调整：从{d * features_per_node}到{d}")

        # 如果提供了initial_W，使用它来创建初始拓扑排序
        if initial_W is not None:
            # 转换邻接矩阵为拓扑排序
            W_thresholded = threshold_W(initial_W, threshold=0.3)

            # 使用出度创建初始拓扑排序
            out_degrees = np.sum(W_thresholded, axis=1)
            topo_init = list(np.argsort(out_degrees))
        else:
            # 随机初始排序
            topo_init = list(np.random.permutation(range(d)))

        # 创建TOPO_linear模型
        model = TOPO_linear(regress=regress, score=score)

        # 拟合模型
        try:
            W_est, topo_order, Z_est, loss = model.fit(X=X, topo=topo_init, verbose=False)

            self.topo_model = {
                'W': W_est,
                'topo_order': topo_order,
                'Z': Z_est,
                'loss': loss
            }

            return W_est
        except Exception as e:
            print(f"Error in topological model training: {str(e)}")
            import traceback
            traceback.print_exc()
            return None

    def fit(self, ts_data, net_data):
        """Train both neural and topological models"""
        print(f"\nTraining neural model ({self.neural_model_type})...")

        # Split data for training
        n_subjects = ts_data.shape[0]
        train_size = int(0.8 * n_subjects)
        train_data = ts_data[:train_size]
        val_data = ts_data[train_size:]
        train_net = net_data[:train_size]
        val_net = net_data[train_size:]

        # 检查并打印数据形状，帮助调试
        print(f"Training data shape: {train_data.shape}")
        print(f"Training network shape: {train_net.shape}")

        # 训练神经模型
        self.neural_trainer.train(
            train_data=train_data,
            train_net=train_net,
            val_data=val_data,
            val_net=val_net,
            epochs=50,
            batch_size=32,
            patience=5
        )

        # 获取神经预测
        neural_pred = self.neural_trainer.predict(ts_data)

        print("\nTraining topological model...")

        # 为拓扑模型提取特征
        # 为每个节点提取4个统计特征: 均值, 标准差, 最大值, 最小值
        X_topo = np.zeros((n_subjects, self.input_dim))

        # 只使用均值特征以避免维度不匹配问题
        for i in range(n_subjects):
            X_topo[i] = np.mean(ts_data[i], axis=0)  # 只取时间维度平均值

        # 训练拓扑模型，使用神经预测作为初始化
        self.topo_W = self.train_topo(X_topo, initial_W=neural_pred)

        return self

    def predict(self, ts_data):
        """Make predictions using both models and combine results"""
        # Get neural prediction
        neural_pred = self.neural_trainer.predict(ts_data)

        # If topological model is available, refine the prediction
        if self.topo_model is not None:
            # Create binary prediction from neural model
            neural_binary = (neural_pred > 0.5).astype(int)

            # Get topological model prediction
            topo_W = self.topo_model['W']
            topo_binary = (np.abs(topo_W) > 0.3).astype(int)

            # Combine predictions (weighted average)
            # Neural model gets higher weight as it's trained directly on the time series
            combined_pred = (0.7 * neural_binary + 0.3 * topo_binary)

            # Enforce topological constraints
            topo_order = self.topo_model['topo_order']
            Z = self.topo_model['Z']

            # Ensure the result respects the topological ordering
            for i in range(len(topo_order)):
                for j in range(i + 1, len(topo_order)):
                    if Z[topo_order[i], topo_order[j]]:
                        combined_pred[topo_order[i], topo_order[j]] = 0

            return combined_pred
        else:
            # If topo model failed, just return neural predictions
            return neural_pred

def calculate_metrics(B_est, B_true):
    met = MetricsDAG(B_est, B_true)
    print("Metrics:", met.metrics)

    return met.metrics

'''
def calculate_metrics(B_est, B_true):
    """Calculate metrics with dynamic thresholding"""
    best_metrics = None
    best_f1 = -1

    # Use fine-grained threshold range
    for thresh in np.linspace(0.1, 0.9, 17):
        B_est_binary = (B_est > thresh).astype(int)

        TP = np.sum((B_est_binary == 1) & (B_true == 1))
        FP = np.sum((B_est_binary == 1) & (B_true == 0))
        FN = np.sum((B_est_binary == 0) & (B_true == 1))
        TN = np.sum((B_est_binary == 0) & (B_true == 0))

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_metrics = {
                'threshold': thresh,
                'shd': np.sum(B_est_binary != B_true),
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': (TP + TN) / (TP + TN + FP + FN),
                'specificity': TN / (TN + FP) if (TN + FP) > 0 else 0,
                'TP': int(TP),
                'FP': int(FP),
                'FN': int(FN),
                'TN': int(TN)
            }

    return best_metrics
'''

def run_hybrid_model(model_type, ts_data, net_data):
    """Run hybrid model training and evaluation"""
    n_nodes = ts_data.shape[-1]
    print(f"\nInput shape: {ts_data.shape}, Number of nodes: {n_nodes}")

    # Adjust model parameters based on dataset size
    hidden_dim = max(64, n_nodes * 4)
    nhead = max(2, n_nodes // 2)  # Ensure it's even
    print(f"Model parameters: hidden_dim={hidden_dim}, nhead={nhead}")

    try:
        # Initialize hybrid model
        trainer = HybridDAGModel(
            neural_model_type=model_type,
            input_dim=n_nodes,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=2
        )

        print(f"\nTraining {model_type.upper()}_TOPO hybrid model...")
        start = timer()

        # Fit the model
        trainer.fit(ts_data, net_data)

        # Predict
        pred_adj = trainer.predict(ts_data)
        end = timer()

        # Binary threshold the predictions
        B_est = (pred_adj > 0.5).astype(int)
        B_true = (np.mean(net_data, axis=0) > 0).astype(int)

        # Calculate detailed metrics
        metrics = calculate_metrics(B_est, B_true)
        print(f"\n{model_type.upper()}_TOPO Results:")
        print(f"Time: {end - start:.4f}s")
        print("Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Print comparison of predicted and true matrices
        print("\nPredicted adjacency matrix:")
        print(B_est)
        print("\nTrue adjacency matrix (mean):")
        print(B_true)

        return B_est, metrics
    except Exception as e:
        print(f"Error running {model_type}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == '__main__':
    from scipy.io import loadmat

    all_results = {}

    for dataset in range(21,29):
        print(f'\nProcessing dataset #{dataset}')

        # Load data
        try:
            data = loadmat(f'../../datasets/sims/sim{dataset}.mat')
            net = data['net']
            ts = data['ts']
            Nnodes = int(data['Nnodes'][0][0])
            Nsubjects = int(data['Nsubjects'][0][0])
            Ntimepoints = int(data['Ntimepoints'][0][0])
        except Exception as e:
            print(f"Error loading dataset {dataset}: {str(e)}")
            continue

        if Nnodes != 5:
            print('Node != 5, continue to save time')
            continue

        # Reshape time series data
        ts_all = ts.reshape(Nsubjects, Ntimepoints, Nnodes)

        # Run all hybrid models
        models = ['tcn', 'transformer', 'deep_dag']
        dataset_results = {}

        for model_type in models:
            try:
                B_est, metrics = run_hybrid_model(model_type, ts_all, net)
                if metrics is not None:  # Only store results if metrics are available
                    dataset_results[f"{model_type}_topo"] = metrics
                print(f"\nPredicted adjacency matrix shape: {B_est.shape if B_est is not None else 'None'}")
                print(f"True adjacency matrix shape: {net.shape}")
            except Exception as e:
                print(f"Error running {model_type}: {str(e)}")
                continue

        all_results[dataset] = dataset_results

    print(all_results)

    # Print average results across all datasets
    print("\nAverage Results Across All Datasets:")
    for model_type in [f"{m}_topo" for m in models]:
        try:
            # Get all datasets that have results for this model
            datasets_with_model = [d for d in all_results if model_type in all_results[d]]

            if datasets_with_model:
                # Calculate averages and variances for key metrics
                metrics_to_report = ['shd', 'precision', 'recall', 'F1']
                avg_metrics = {}
                var_metrics = {}

                for metric in metrics_to_report:
                    # Filter out NaN values before calculating statistics
                    values = [all_results[d][model_type][metric] for d in datasets_with_model
                              if not np.isnan(all_results[d][model_type].get(metric, np.nan))]

                    if values:  # Only calculate if there are non-NaN values
                        avg_metrics[metric] = np.mean(values)
                        var_metrics[metric] = np.var(values)
                    else:
                        avg_metrics[metric] = "N/A"
                        var_metrics[metric] = "N/A"

                print(f"\n{model_type.upper()}:")
                print(f"Number of successful runs: {len(datasets_with_model)}")
                for metric in metrics_to_report:
                    if isinstance(avg_metrics[metric], str):
                        print(f"{metric}: No valid data (all NaN)")
                    else:
                        print(f"{metric}: mean={avg_metrics[metric]:.4f}, var={var_metrics[metric]:.4f}")
            else:
                print(f"\n{model_type.upper()}: No successful runs")
        except Exception as e:
            print(f"Error calculating averages for {model_type}: {str(e)}")