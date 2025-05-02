import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re


# 解析combined_results.txt文件中的所有数据集
def parse_combined_results(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # 使用正则表达式匹配所有数据集
    datasets = re.findall(r'dataset\s+(\d+)(.*?)(?=dataset|\Z)', content, re.DOTALL)

    results = []
    for dataset_id, dataset_content in datasets:
        # 提取三个矩阵
        matrices = re.findall(r'\[\[(.*?)\]\]', dataset_content, re.DOTALL)

        if len(matrices) >= 3:
            # 处理NOTEARS矩阵
            notears_matrix_str = matrices[0]
            notears_matrix = parse_matrix(notears_matrix_str)

            # 处理Transformer矩阵
            transformer_matrix_str = matrices[1]
            transformer_matrix = parse_matrix(transformer_matrix_str)

            # 处理真实矩阵
            true_matrix_str = matrices[2]
            true_matrix = parse_matrix(true_matrix_str)

            results.append({
                'dataset_id': int(dataset_id),
                'notears_matrix': notears_matrix,
                'transformer_matrix': transformer_matrix,
                'true_matrix': true_matrix
            })

    return results


# 辅助函数：解析矩阵字符串为numpy数组
def parse_matrix(matrix_str):
    # 清理字符串并分割成行
    lines = matrix_str.strip().replace('[', '').replace(']', '').split('\n')

    # 处理每一行
    matrix = []
    for line in lines:
        if line.strip():
            row = [int(x) for x in line.strip().split()]
            matrix.append(row)

    return np.array(matrix)


# 计算矩阵的性能指标
def calculate_metrics(predicted_matrix, true_matrix):
    # 将预测矩阵和真实矩阵展平为1D数组
    pred_flat = predicted_matrix.flatten()
    true_flat = true_matrix.flatten()

    # 计算TP, FP, TN, FN
    tp = np.sum((pred_flat == 1) & (true_flat == 1))
    fp = np.sum((pred_flat == 1) & (true_flat == 0))
    tn = np.sum((pred_flat == 0) & (true_flat == 0))
    fn = np.sum((pred_flat == 0) & (true_flat == 1))

    # 计算精度、召回率和F1分数
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # 计算准确率
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # 计算结构汉明距离（SHD）
    shd = np.sum(predicted_matrix != true_matrix)

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'shd': shd
    }


# 二值化矩阵的函数
def binarize_matrix(matrix, threshold=0.5):
    return (matrix > threshold).astype(int)


# 主函数
def main():
    # 解析combined_results.txt文件
    results = parse_combined_results('combined_results.txt')

    if not results:
        print("未能从文件中解析出任何数据集")
        return

    # 计算每个数据集的性能指标
    notears_metrics = []
    transformer_metrics = []

    # 收集所有矩阵以计算平均值
    all_notears_matrices = []
    all_transformer_matrices = []
    all_true_matrices = []

    for result in results:
        notears_matrix = result['notears_matrix']
        transformer_matrix = result['transformer_matrix']
        true_matrix = result['true_matrix']

        all_notears_matrices.append(notears_matrix)
        all_transformer_matrices.append(transformer_matrix)
        all_true_matrices.append(true_matrix)

        notears_result = calculate_metrics(notears_matrix, true_matrix)
        transformer_result = calculate_metrics(transformer_matrix, true_matrix)

        notears_metrics.append(notears_result)
        transformer_metrics.append(transformer_result)

    # 计算平均性能指标
    avg_notears = {metric: np.mean([res[metric] for res in notears_metrics]) for metric in notears_metrics[0].keys()}
    avg_transformer = {metric: np.mean([res[metric] for res in transformer_metrics]) for metric in
                       transformer_metrics[0].keys()}

    # 打印平均性能指标
    print("NOTEARS平均性能指标:")
    for metric, value in avg_notears.items():
        print(f"  {metric}: {value:.4f}")

    print("\nTransformer平均性能指标:")
    for metric, value in avg_transformer.items():
        print(f"  {metric}: {value:.4f}")

    # 计算平均邻接矩阵
    avg_notears_matrix = np.mean(all_notears_matrices, axis=0)
    avg_transformer_matrix = np.mean(all_transformer_matrices, axis=0)
    avg_true_matrix = np.mean(all_true_matrices, axis=0)

    # 二值化平均邻接矩阵
    binary_threshold = 0.3  # 阈值可以根据需要调整
    binary_notears_matrix = binarize_matrix(avg_notears_matrix, binary_threshold)
    binary_transformer_matrix = binarize_matrix(avg_transformer_matrix, binary_threshold)
    binary_true_matrix = binarize_matrix(avg_true_matrix, binary_threshold)

    # 第一张图 - 平均因果矩阵比较
    plt.figure(figsize=(20, 10))

    # 原始平均矩阵
    plt.subplot(2, 3, 1)
    sns.heatmap(avg_true_matrix, annot=True, cmap='Blues', cbar=True, fmt='.2f',
                xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.title('Avg True Adjacency Matrix')
    plt.xlabel('Node')
    plt.ylabel('Node')

    plt.subplot(2, 3, 2)
    sns.heatmap(avg_notears_matrix, annot=True, cmap='Blues', cbar=True, fmt='.2f',
                xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.title('Avg NOTEARS Adjacency Matrix')
    plt.xlabel('Node')
    plt.ylabel('Node')

    plt.subplot(2, 3, 3)
    sns.heatmap(avg_transformer_matrix, annot=True, cmap='Blues', cbar=True, fmt='.2f',
                xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.title('Avg Transformer Adjacency Matrix')
    plt.xlabel('Node')
    plt.ylabel('Node')

    # 二值化平均矩阵
    plt.subplot(2, 3, 4)
    sns.heatmap(binary_true_matrix, annot=True, cmap='Blues', cbar=True, fmt='.0f',
                xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.title(f'Binarized True Adjacency Matrix (threshold={binary_threshold})')
    plt.xlabel('Node')
    plt.ylabel('Node')

    plt.subplot(2, 3, 5)
    sns.heatmap(binary_notears_matrix, annot=True, cmap='Blues', cbar=True, fmt='.0f',
                xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.title(f'Binarized NOTEARS Adjacency Matrix (threshold={binary_threshold})')
    plt.xlabel('Node')
    plt.ylabel('Node')

    plt.subplot(2, 3, 6)
    sns.heatmap(binary_transformer_matrix, annot=True, cmap='Blues', cbar=True, fmt='.0f',
                xticklabels=range(1, 6), yticklabels=range(1, 6))
    plt.title(f'Binarized Transformer Adjacency Matrix (threshold={binary_threshold})')
    plt.xlabel('Node')
    plt.ylabel('Node')

    plt.tight_layout()
    plt.savefig('adjacency_matrices_comparison.png', dpi=300)
    plt.close()

    # 计算二值化矩阵的性能指标
    binary_notears_metrics = calculate_metrics(binary_notears_matrix, binary_true_matrix)
    binary_transformer_metrics = calculate_metrics(binary_transformer_matrix, binary_true_matrix)

    # 打印二值化矩阵的性能指标
    print("\n二值化NOTEARS矩阵的性能指标:")
    for metric, value in binary_notears_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\n二值化Transformer矩阵的性能指标:")
    for metric, value in binary_transformer_metrics.items():
        print(f"  {metric}: {value:.4f}")

    # 第二张图 - 模型性能对比柱状图
    metrics = ['precision', 'recall', 'f1', 'accuracy']

    plt.figure(figsize=(12, 6))

    x = np.arange(len(metrics))
    width = 0.35

    rects1 = plt.bar(x - width / 2, [avg_notears[m] for m in metrics], width, label='NOTEARS')
    rects2 = plt.bar(x + width / 2, [avg_transformer[m] for m in metrics], width, label='Transformer')

    plt.ylabel('Score')
    plt.title('Performance Comparison between NOTEARS and Transformer')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()

    # 在柱状图上添加文本标签
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            plt.annotate(f'{height:.2f}',
                         xy=(rect.get_x() + rect.get_width() / 2, height),
                         xytext=(0, 3),  # 3点文本偏移
                         textcoords="offset points",
                         ha='center', va='bottom')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=300)

    # 第三张图 - 二值化矩阵性能对比
    plt.figure(figsize=(12, 6))

    rects1 = plt.bar(x - width / 2, [binary_notears_metrics[m] for m in metrics], width, label='Binary NOTEARS')
    rects2 = plt.bar(x + width / 2, [binary_transformer_metrics[m] for m in metrics], width, label='Binary Transformer')

    plt.ylabel('Score')
    plt.title(f'Performance Comparison of Binarized Matrices (threshold={binary_threshold})')
    plt.xticks(x, metrics)
    plt.ylim(0, 1)
    plt.legend()

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.savefig('binary_performance_comparison.png', dpi=300)

    # 第四张图 - SHD比较
    plt.figure(figsize=(10, 6))

    models = ['NOTEARS', 'Transformer', 'Binary NOTEARS', 'Binary Transformer']
    shd_values = [
        np.mean([m['shd'] for m in notears_metrics]),
        np.mean([m['shd'] for m in transformer_metrics]),
        binary_notears_metrics['shd'],
        binary_transformer_metrics['shd']
    ]

    plt.bar(models, shd_values, color=['blue', 'green', 'lightblue', 'lightgreen'])
    plt.ylabel('Structural Hamming Distance (SHD)')
    plt.title('Average SHD Comparison')
    plt.grid(True, linestyle='--', alpha=0.3, axis='y')

    # 在柱状图上添加文本标签
    for i, v in enumerate(shd_values):
        plt.text(i, v + 0.1, f"{v:.2f}", ha='center')

    plt.tight_layout()
    plt.savefig('shd_comparison.png', dpi=300)

    # 第五张图 - 所有数据集的F1分数对比
    plt.figure(figsize=(12, 6))

    dataset_ids = [res['dataset_id'] for res in results]
    f1_notears = [res['f1'] for res in notears_metrics]
    f1_transformer = [res['f1'] for res in transformer_metrics]

    # 按数据集ID排序
    sorted_indices = np.argsort(dataset_ids)
    sorted_dataset_ids = [dataset_ids[i] for i in sorted_indices]
    sorted_f1_notears = [f1_notears[i] for i in sorted_indices]
    sorted_f1_transformer = [f1_transformer[i] for i in sorted_indices]

    plt.plot(sorted_dataset_ids, sorted_f1_notears, 'o-', label='NOTEARS')
    plt.plot(sorted_dataset_ids, sorted_f1_transformer, 's-', label='Transformer')
    plt.xlabel('Dataset ID')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison across Datasets')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('f1_comparison_across_datasets.png', dpi=300)

    # 为了保持原始代码中的散点图示例，我们还生成模拟数据的散点图比较
    np.random.seed(42)

    # 生成基本事实网络的嵌入表示
    def generate_ground_truth_embedding(n_samples=1000):
        # 创建一个有明确结构的分布
        x = np.random.normal(0, 3, n_samples)
        y = np.random.normal(0, 3, n_samples)

        # 添加一些结构
        mask = np.random.random(n_samples) < 0.5
        r = 5 + np.random.normal(0, 0.5, n_samples)
        theta = np.linspace(0, 2 * np.pi, n_samples)
        x[mask] = r[mask] * np.cos(theta[mask])
        y[mask] = r[mask] * np.sin(theta[mask])

        return np.column_stack([x, y])

    # 生成NOTEARS模型的分布数据
    def generate_notears_embedding(ground_truth, noise_level=0.8):
        # 从基本事实添加噪声和一些畸变
        n_samples = ground_truth.shape[0]
        noise = np.random.normal(0, noise_level, ground_truth.shape)

        # 添加偏移
        offset = np.random.uniform(-2, 2, ground_truth.shape)

        # 有些点保留原始分布，有些点添加噪音
        mask = np.random.random(n_samples) < 0.6
        result = ground_truth.copy()
        result[mask] += noise[mask] + offset[mask]

        # 添加一些额外的噪声点
        extra_noise_mask = np.random.random(n_samples) < 0.1
        result[extra_noise_mask, 0] = np.random.uniform(-10, 10, np.sum(extra_noise_mask))
        result[extra_noise_mask, 1] = np.random.uniform(-10, 10, np.sum(extra_noise_mask))

        return result

    # 生成Transformer模型的分布数据
    def generate_transformer_embedding(ground_truth, noise_level=0.5):
        # 从基本事实添加较少的噪声，因为Transformer往往更准确
        n_samples = ground_truth.shape[0]
        noise = np.random.normal(0, noise_level, ground_truth.shape)

        # 添加小偏移
        offset = np.random.uniform(-1, 1, ground_truth.shape)

        # 大部分点保留原始分布，少部分点添加噪音
        mask = np.random.random(n_samples) < 0.3
        result = ground_truth.copy()
        result[mask] += noise[mask] + offset[mask]

        # 添加少量噪声点
        extra_noise_mask = np.random.random(n_samples) < 0.05
        result[extra_noise_mask, 0] = np.random.uniform(-10, 10, np.sum(extra_noise_mask))
        result[extra_noise_mask, 1] = np.random.uniform(-10, 10, np.sum(extra_noise_mask))

        return result

    # 生成数据
    ground_truth_embedding = generate_ground_truth_embedding(1500)
    notears_embedding = generate_notears_embedding(ground_truth_embedding)
    transformer_embedding = generate_transformer_embedding(ground_truth_embedding)

    # 创建散点图
    plt.figure(figsize=(15, 6))

    # NOTEARS模型 vs 基本事实网络
    plt.subplot(1, 2, 1)
    plt.scatter(ground_truth_embedding[:, 0], ground_truth_embedding[:, 1], color='blue', alpha=0.6, s=10,
                label='Ground Truth')
    plt.scatter(notears_embedding[:, 0], notears_embedding[:, 1], color='red', alpha=0.6, s=10, label='NOTEARS')
    plt.title('NOTEARS vs Ground Truth Embedding')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    # Transformer模型 vs 基本事实网络
    plt.subplot(1, 2, 2)
    plt.scatter(ground_truth_embedding[:, 0], ground_truth_embedding[:, 1], color='blue', alpha=0.6, s=10,
                label='Ground Truth')
    plt.scatter(transformer_embedding[:, 0], transformer_embedding[:, 1], color='red', alpha=0.6, s=10,
                label='Transformer')
    plt.title('Transformer vs Ground Truth Embedding')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig('model_vs_groundtruth_comparison.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    main()