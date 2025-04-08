import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.io import loadmat
from utils import set_random_seed
from sklearn.preprocessing import StandardScaler

# 数据加载和预处理
data = loadmat('sim1.mat')  # 示例数据，替换为实际数据路径

net = data['net']  # shape: (Nsubjects, Nnodes, Nnodes)
ts = data['ts']  # shape: (Ntimepoints, Nnodes)
Nnodes = int(data['Nnodes'][0][0])
Nsubjects = int(data['Nsubjects'][0][0])
Ntimepoints = int(data['Ntimepoints'][0][0])

# 重塑数据使每个受试者的时间序列单独存储
ts_all = ts.reshape(Nsubjects, Ntimepoints, Nnodes)
X_combined = ts_all.reshape(-1, Nnodes)  # 将所有受试者的数据拼接成一个大矩阵

# 结合所有受试者的网络，取平均后二值化作为真值
B_true_linear = (np.mean(net, axis=0) != 0).astype(int)

# 设置随机种子
rd_int = int(np.random.randint(10000, size=1)[0])
set_random_seed(rd_int)

# 标准化时间序列数据（可以选择是否进行标准化）
scaler = StandardScaler()
X_combined_scaled = scaler.fit_transform(X_combined)

# 神经网络模型定义
model = Sequential()
model.add(Dense(128, input_dim=Nnodes, activation='relu'))  # 输入层，假设128个神经元
model.add(Dense(64, activation='relu'))  # 隐藏层，64个神经元
model.add(Dense(Nnodes * Nnodes, activation='linear'))  # 输出层，输出大小为Nnodes*Nnodes，回归任务

# 编译模型
model.compile(optimizer='adam', loss='mse')

# 训练神经网络模型
# X_combined_scaled 是训练数据，目标是扁平化的网络矩阵（Nnodes * Nnodes）
y_combined = net.reshape(Nsubjects, -1)  # 扁平化为Nnodes * Nnodes

model.fit(X_combined_scaled, y_combined, epochs=50, batch_size=32, verbose=1)

# 进行预测
y_pred_scaled = model.predict(X_combined_scaled)

# 将预测结果还原为矩阵形式
y_pred = y_pred_scaled.reshape(Nsubjects, Nnodes, Nnodes)

# 计算预测的网络的二值化结果
B_pred = (np.mean(y_pred, axis=0) != 0).astype(int)

# 评估模型：计算二值化后的网络与真实网络的比较
from castle import MetricsDAG
met = MetricsDAG(B_pred, B_true_linear)
metrics = met.metrics

# 输出评估结果
print("评估指标:", metrics)
