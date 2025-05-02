import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 第一张图 - 因果矩阵比较
# 示例因果矩阵数据
true_adj_matrix = np.array([
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]
])

notears_adj_matrix = np.array([
    [0, 1, 0, 0, 1],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0]
])

transformer_adj_matrix = np.array([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
])

plt.figure(figsize=(15, 5))

# 真实因果矩阵
plt.subplot(1, 3, 1)
sns.heatmap(true_adj_matrix, annot=True, cmap='Blues', cbar=False, fmt='.0f',
            xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.title('True Adjacency Matrix')
plt.xlabel('Node')
plt.ylabel('Node')

# NOTEARS因果矩阵
plt.subplot(1, 3, 2)
sns.heatmap(notears_adj_matrix, annot=True, cmap='Blues', cbar=False, fmt='.0f',
            xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.title('NOTEARS Adjacency Matrix')
plt.xlabel('Node')
plt.ylabel('Node')

# Transformer因果矩阵
plt.subplot(1, 3, 3)
sns.heatmap(transformer_adj_matrix, annot=True, cmap='Blues', cbar=False, fmt='.0f',
            xticklabels=range(1, 6), yticklabels=range(1, 6))
plt.title('Transformer Adjacency Matrix')
plt.xlabel('Node')
plt.ylabel('Node')

plt.tight_layout()
plt.savefig('adjacency_matrices_comparison.png', dpi=300)
plt.close()

# 第二张图 - 模型映射散点图与基本事实网络对比
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

# 创建第二张图
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
plt.scatter(transformer_embedding[:, 0], transformer_embedding[:, 1], color='red', alpha=0.6, s=10, label='Transformer')
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