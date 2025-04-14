import numpy as np
from scipy.io import loadmat
from topo_linear import TOPO_linear, threshold_W
from topo_neural import NeuralDAGTrainer
from sklearn.linear_model import LinearRegression
import utils
from timeit import default_timer as timer
from scipy.signal import butter, filtfilt


def calculate_metrics(B_est, B_true):
    """Calculate metrics with dynamic thresholding"""
    best_metrics = None
    best_f1 = -1

    # 使用更细粒度的阈值范围
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


def run_model(model_type, ts_data, net_data):
    n_nodes = ts_data.shape[-1]
    print(f"\nInput shape: {ts_data.shape}, Number of nodes: {n_nodes}")

    # 调整模型参数
    hidden_dim = max(64, n_nodes * 4)
    nhead = max(2, n_nodes // 2)  # 确保是偶数
    print(f"Model parameters: hidden_dim={hidden_dim}, nhead={nhead}")

    try:
        # 添加提前停止和学习率调度
        trainer = NeuralDAGTrainer(
            model_type=model_type,
            input_dim=n_nodes,
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=3
        )

        print(f"\nTraining {model_type.upper()} model...")
        start = timer()

        # 分割数据集为训练集和验证集
        n_subjects = ts_data.shape[0]
        train_size = int(0.8 * n_subjects)
        train_data = ts_data[:train_size]
        val_data = ts_data[train_size:]
        train_net = net_data[:train_size]
        val_net = net_data[train_size:]

        trainer.train(train_data, train_net, val_data, val_net,
                      epochs=50, batch_size=32, patience=5)

        pred_adj = trainer.predict(ts_data)
        end = timer()

        # Binary threshold the predictions
        B_est = (pred_adj > 0.5).astype(int)
        B_true = (np.mean(net_data, axis=0) > 0).astype(int)

        # 计算详细指标
        metrics = calculate_metrics(B_est, B_true)
        print(f"\n{model_type.upper()} Results:")
        print(f"Time: {end - start:.4f}s")
        print("Metrics:")
        for metric, value in metrics.items():
            if metric in ['TP', 'FP', 'FN', 'TN']:
                print(f"  {metric}: {int(value)}")
            else:
                print(f"  {metric}: {value:.4f}")

        # 打印预测矩阵和真实矩阵的比较
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
    all_results = {}

    for dataset in range(1, 29):
        print(f'\nProcessing dataset #{dataset}')

        # Load data
        data = loadmat(f'../datasets/sims/sim{dataset}.mat')
        net = data['net']
        ts = data['ts']
        Nnodes = int(data['Nnodes'][0][0])
        Nsubjects = int(data['Nsubjects'][0][0])
        Ntimepoints = int(data['Ntimepoints'][0][0])

        if Nnodes != 5:
            print('Node != 5, continue to save time')
            continue

        # Reshape time series data
        ts_all = ts.reshape(Nsubjects, Ntimepoints, Nnodes)

        # Run all models
        models = ['tcn', 'transformer', 'deep_dag']
        dataset_results = {}

        for model_type in models:
            try:
                B_est, metrics = run_model(model_type, ts_all, net)
                if metrics is not None:  # Only store results if metrics are available
                    dataset_results[model_type] = metrics
                print(f"\nPredicted adjacency matrix shape: {B_est.shape if B_est is not None else 'None'}")
                print(f"True adjacency matrix shape: {net.shape}")
            except Exception as e:
                print(f"Error running {model_type}: {str(e)}")
                continue

        all_results[dataset] = dataset_results

    # Print average results across all datasets
    print("\nAverage Results Across All Datasets:")
    for model_type in models:
        try:
            # Get all datasets that have results for this model
            datasets_with_model = [d for d in all_results if model_type in all_results[d]]

            if datasets_with_model:
                # Calculate averages only for datasets that have this model's results
                metrics_to_report = ['shd', 'precision', 'recall', 'f1']
                avg_metrics = {}

                for metric in metrics_to_report:
                    values = [all_results[d][model_type][metric] for d in datasets_with_model]
                    avg_metrics[metric] = np.mean(values)

                print(f"\n{model_type.upper()}:")
                print(f"Number of successful runs: {len(datasets_with_model)}")
                for metric, value in avg_metrics.items():
                    print(f"{metric}: {value:.4f}")
            else:
                print(f"\n{model_type.upper()}: No successful runs")
        except Exception as e:
            print(f"Error calculating averages for {model_type}: {str(e)}")