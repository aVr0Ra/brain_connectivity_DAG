import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font",family='MicroSoft YaHei',weight="bold")

# Data
methods = ['线性模型', 'TCN', 'TRANSFORMER', 'DEEP_DAG']
metrics = ['SHD', 'Precision', 'Recall', 'F1']

means = {
    'SHD': [2.88, 3.63, 2.63, 3.75],
    'Precision': [0.56, 0.71, 0.96, 0.66],
    'Recall': [0.45, 0.43, 0.48, 0.43],
    'F1': [0.56, 0.58, 0.60, 0.49]
}

stds = {
    'SHD': [1.36, 1.65, 1.11, 2.00],
    'Precision': [0.32, 0.32, 0.11, 0.31],
    'Recall': [0.26, 0.25, 0.22, 0.23],
    'F1': [0.22, 0.20, 0.21, 0.23]
}

colors = ['blue', 'red', 'green', 'orange']

fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

for ax, metric in zip(axes, metrics):
    ax.bar(methods, means[metric], yerr=stds[metric], capsize=5, color=colors)
    ax.set_title(metric, fontsize=25)
    ax.set_xticklabels(methods, rotation=20, fontsize=15)
    ax.set_ylabel('Value' if metric == 'SHD' else None)
    if metric == 'SHD':
        ax.set_ylim(0, max(means[metric]) + max(stds[metric]) + 1)
    else:
        ax.set_ylim(0, max(means[metric]) + max(stds[metric]) + 0.1)

plt.show()
