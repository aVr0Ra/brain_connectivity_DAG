import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc("font",family='MicroSoft YaHei',weight="bold")

# Data
methods = ['TCN', 'TRANSFORMER', 'DEEP_DAG', '线性模型']
metrics = ['SHD', 'Precision', 'Recall', 'F1']

means = {
    'SHD': [3.76, 2.89, 6.14, 2.33],
    'Precision': [0.74, 0.92, 0.50, 0.66],
    'Recall': [0.49, 0.49, 0.45, 0.67],
    'F1': [0.55, 0.61, 0.45, 0.56]
}

stds = {
    'SHD': [2.09, 1.17, 3.62, 1.73],
    'Precision': [0.22, 0.14, 0.28, 0.26],
    'Recall': [0.24, 0.22, 0.22, 0.33],
    'F1': [0.22, 0.17, 0.22, 0.32]
}

colors = ['blue', 'red', 'green', 'orange']

fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)

for ax, metric in zip(axes, metrics):
    ax.bar(methods, means[metric], yerr=stds[metric], capsize=5, color=colors)
    ax.set_title(metric, fontsize=25)
    ax.set_xticklabels(methods, rotation=20, fontsize=15)
    ax.set_ylabel('Value' if metric == 'SHD' else None)
    ax.set_ylim(0, max(means[metric]) + max(stds[metric]) + 1)

plt.show()
