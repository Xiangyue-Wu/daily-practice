import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_data():
    # 加载训练集，形状为 (n, n_rows, n_columns)
    file_dir = os.path.dirname(__file__)
    data_path = os.path.join(file_dir, 'X_training.npy')
    obj = np.load(data_path, allow_pickle=True)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.shape == ():
        obj = obj.item()
        X = obj.get('X_trianing', None)
        if X is None:
            X = obj.get('X_training', None)
        if X is None:
            raise ValueError('未在字典中找到键 X_trianing / X_training')
    else:
        X = obj
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 3:
        n, n_rows, n_columns = X.shape
        X_flat = X.reshape(n, -1)
    elif X.ndim == 2:
        D, n = X.shape if X.shape[0] > X.shape[1] else (X.shape[1], X.shape[0])
        X2 = X if X.shape[1] == n else X.T
        # 由任务背景：尺寸为 480×640，因此 D=307200=480*640
        n_rows, n_columns = 480,640
        X_flat = X2.T if X2.shape[0] == D else X2
        X_flat = X_flat.reshape(n, -1)
    else:
        raise ValueError('不支持的数据维度: {}'.format(X.ndim))
    return X_flat, n, n_rows, n_columns

def compute_eigenfaces(X_flat):
    # 中心化：每张图减去所有图的均值向量
    mean_face = X_flat.mean(axis=0)
    Xc = X_flat - mean_face
    # 小矩阵法：求解 Xc Xc^T 的特征分解，尺寸为 (n×n)
    # 令 Xc Xc^T u = λ u，则 v = Xc^T u 是 Xc^T Xc 的特征向量
    C_small = Xc @ Xc.T
    evals, U = np.linalg.eigh(C_small)
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    U = U[:, idx]
    # 将小矩阵特征向量映射到高维并单位化，得到特征脸矩阵 W (D×k)
    eps = 1e-6
    valid = evals > eps
    evals = evals[valid]
    U = U[:, valid]
    W_list = []
    for i in range(U.shape[1]):
        ui = U[:, i]
        lam = evals[i]
        v = Xc.T @ ui
        v = v / np.sqrt(lam)
        W_list.append(v)
    W = np.stack(W_list, axis=1)
    # 主成分系数：每张中心化图在特征脸上的投影 S (n×k)
    S = Xc @ W
    # 方差解释比例
    explained_ratio = evals / evals.sum()
    return mean_face, W, S, evals, explained_ratio

def reconstruct_with_q(mean_face, W, S, q):
    # 使用前 q 个主成分进行重构：X̂ = mean + S_q W_q^T
    Wq = W[:, :q]
    Sq = S[:, :q]
    Xc_hat = Sq @ Wq.T
    X_hat = Xc_hat + mean_face
    return X_hat

def compute_rmse_curve(X_flat, mean_face, W, S):
    # 针对 q=1..k 计算 RMSE = sqrt(mean((X - X̂_q)^2))
    k = W.shape[1]
    RMSE = []
    q_list = []
    for q in range(1, k + 1):
        X_hat = reconstruct_with_q(mean_face, W, S, q)
        rmse = np.sqrt(np.mean((X_flat - X_hat) ** 2))
        RMSE.append(float(rmse))
        q_list.append(int(q))
    return np.array(q_list), np.array(RMSE)

def _compute_grid(m, max_cols=6):
    # 计算合适的网格行列数，尽量多列以避免纵向过长
    if m <= 0:
        return 1, 1
    if m <= 3:
        cols = m
    else:
        cols = min(max_cols, max(3, int(np.ceil(np.sqrt(m) * 1.2))))
    rows = int(np.ceil(m / cols))
    return rows, cols

def plot_mean_and_eigenfaces(mean_face, W, n_rows, n_columns, save_path=None):
    colormap = plt.cm.gray
    k = W.shape[1]
    total = k + 1
    rows, cols = _compute_grid(total, max_cols=6)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    fig.set_facecolor('white')
    fig.suptitle('均值脸与特征脸', fontsize=16)
    axes = np.array(axes).reshape(-1)
    # 放置均值脸
    axes[0].imshow(mean_face.reshape(n_rows, n_columns), cmap=colormap)
    axes[0].axis('off')
    axes[0].set_title('均值脸', fontsize=10, pad=2)
    # 放置特征脸
    for i in range(k):
        ax = axes[i + 1]
        img = W[:, i].reshape(n_rows, n_columns)
        ax.imshow(img, cmap=colormap)
        ax.axis('off')
        ax.set_title('特征脸 {}'.format(i + 1), fontsize=9, pad=2)
    # 隐藏多余子图
    for j in range(total, rows * cols):
        axes[j].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return None

def plot_reconstructions(X_hat, n_rows, n_columns, title='重构图像', save_path=None):
    n = X_hat.shape[0]
    colormap = plt.cm.gray
    rows, cols = _compute_grid(n, max_cols=6)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    fig.set_facecolor('white')
    fig.suptitle(title, fontsize=16)
    axes = np.array(axes).reshape(-1)
    for i in range(n):
        ax = axes[i]
        img = X_hat[i, :].reshape(n_rows, n_columns)
        ax.imshow(img, cmap=colormap)
        ax.axis('off')
        ax.set_title('重构 {}'.format(i + 1), fontsize=9, pad=2)
    for j in range(n, rows * cols):
        axes[j].axis('off')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return None

def plot_rmse(q_list, RMSE, save_path=None):
    # 绘制 RMSE 随主成分数量的变化曲线
    fig = plt.figure('RMSE-主成分数量', figsize=(6, 5))
    fig.set_facecolor('white')
    plt.plot(q_list, RMSE, 'r+', markersize=8, linewidth=2)
    plt.xlabel('q', fontsize=14)
    plt.ylabel('RMSE', fontsize=14)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return None

def plot_explained_ratio(ratio, save_path=None):
    # 可选：解释方差比例与累积曲线
    cum = np.cumsum(ratio)
    fig = plt.figure('方差解释比例', figsize=(6, 5))
    fig.set_facecolor('white')
    plt.plot(np.arange(1, ratio.size + 1), ratio, 'b-o', label='单个比例')
    plt.plot(np.arange(1, ratio.size + 1), cum, 'g-s', label='累积比例')
    plt.xlabel('主成分序号', fontsize=14)
    plt.ylabel('比例', fontsize=14)
    plt.ylim(0, 1.05)
    plt.legend()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    return None

def main():
    # 1.加载数据并重塑
    X_flat, n, n_rows, n_columns = load_data()
    # 2.计算特征脸（小矩阵法）与主成分系数
    mean_face, W, S, evals, ratio = compute_eigenfaces(X_flat)
    # 3.可视化均值脸与特征脸
    plot_mean_and_eigenfaces(mean_face, W, n_rows, n_columns, save_path='mean_face_and_eigenfaces.png')
    # 4.使用不同 q 重构并展示（示例：q=1..min(8, k)）
    k = W.shape[1]
    q_to_show = min(13, k)
    X_hat_q = reconstruct_with_q(mean_face, W, S, q_to_show)
    plot_reconstructions(X_hat_q, n_rows, n_columns, save_path='Reconstructed_image.png')
    # 5.计算 RMSE 曲线
    q_list, rmse_list = compute_rmse_curve(X_flat, mean_face, W, S)
    plot_rmse(q_list, rmse_list, save_path='RMSE_vs_q.png')
    # 6.可选：方差解释比例
    plot_explained_ratio(ratio, save_path='Explained_variance_ratio.png')
    return None

if __name__ == '__main__':
    # 数据导入、初始变量定义等由函数内部完成
    main()
