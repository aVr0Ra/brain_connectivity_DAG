from scipy.io import loadmat
import numpy as np
from topo_linear import TOPO_linear, threshold_W
from sklearn.linear_model import LinearRegression
import utils
from timeit import default_timer as timer
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from castle import MetricsDAG


def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y


if __name__ == '__main__':
    # 数据加载和预处理

    for dataset in range(1,29):
        data = loadmat(f'../datasets/sims/sim{dataset}.mat')
        print('\nCurrent dataset: #', dataset)

        net = data['net']  # shape: (Nsubjects, Nnodes, Nnodes)
        ts = data['ts']  # shape: (Ntimepoints, Nnodes)
        Nnodes = int(data['Nnodes'][0][0])
        Nsubjects = int(data['Nsubjects'][0][0])
        Ntimepoints = int(data['Ntimepoints'][0][0])

        print("Original ts shape:", ts.shape)
        # 如果需要滤波，可以取消下面两行的注释
        # fs = 1 / 3.0  # TR=3s
        # ts = bandpass_filter(ts, 0.01, 0.1, fs, order=3)

        # 重塑 ts，使得每个受试者的数据都包含在内
        ts_all = ts.reshape(Nsubjects, Ntimepoints, Nnodes)
        print("Reshaped ts_all shape (subjects, timepoints, nodes):", ts_all.shape)

        # 如果想要将所有受试者的数据拼接起来进行整体分析：
        X_combined = ts_all.reshape(-1, Nnodes)
        print("Combined X shape:", X_combined.shape)

        # 结合所有受试者的网络，取平均后二值化作为真值
        B_true_linear = (np.mean(net, axis=0) != 0)
        print("B_true_linear shape:", B_true_linear.shape)

        # 设置随机种子
        rd_int = int(np.random.randint(10000, size=1)[0])
        print(f"random seed: {rd_int}")
        utils.set_random_seed(rd_int)

        n, d = X_combined.shape
        verbose = False


        # 线性模型部分
        def regress(X, y):
            reg = LinearRegression(fit_intercept=False)
            reg.fit(X=X, y=y)
            return reg.coef_


        def score(X, W):
            M = X @ W
            R = X - M
            loss = 0.5 / X.shape[0] * np.sum(R ** 2)
            G_loss = -1.0 / X.shape[0] * (X.T @ R)
            return loss, G_loss


        model_linear = TOPO_linear(regress=regress, score=score)
        topo_init = list(np.random.permutation(range(d)))
        start = timer()
        W_linear, topo_linear_est, Z_linear, loss_linear = model_linear.fit(X=X_combined, topo=topo_init, verbose=verbose)
        end = timer()
        print(f"Linear model time: {end - start:.4f}s")

        # 绘图调试（可选）
        '''
        plt.figure()
        plt.imshow(W_linear, cmap='viridis', interpolation='none')
        plt.colorbar()
        plt.title("Estimated Connectivity Weights")
        plt.xlabel("Target Node")
        plt.ylabel("Source Node")
        plt.show()
        '''

        B_est = (threshold_W(W=W_linear) != 0).astype(int)
        B_true_linear = np.array(B_true_linear).astype(int)
        # print("Estimated B (B_est):")
        # print(B_est)
        # print("True B (B_true_linear):")
        # print(B_true_linear)
        met = MetricsDAG(B_est, B_true_linear)
        print("Metrics:", met.metrics)



'''
fdr (False Discovery Rate)：误报的比例

tpr (True Positive Rate)：有大约76.19%被模型正确识别出来，也就是召回率

fpr (False Positive Rate)：模型错误地预测为存在连接的比例约为29.17%

shd (Structural Hamming Distance)：预测网络与真实网络之间需要修改的边数

nnz (Non-zero count)：值为13，表示在阈值处理后，模型预测出总共有13条连接（即非零元素的个数）。



'''