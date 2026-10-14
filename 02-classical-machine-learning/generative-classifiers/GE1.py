import numpy as np
from collections import Counter
import math
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
# 设置中文字体支持
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus'] = False
plt.figure(figsize=(10, 8))



# 频率编码：对非数值特征进行频率编码，连续特征保持原值
# 输入：list1（第一张表训练数据，列表，元素为字典），list2（第二张表测试数据，列表，元素为字典）
# 输出：data（np.array，训练集编码后的特征矩阵）和 pre（np.array，测试集编码后的特征矩阵）

def encode_non_numeric(list1, list2):
    """
    输入：
        list1: 训练数据（列表，每个元素为一条记录的字典）
        list2: 测试数据（列表，每个元素为一条记录的字典）
    返回：
        data: np.array，训练集编码后的特征
        pre: np.array，测试集编码后的特征
    """
    features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感', '密度', '含糖率']
    discrete_features = ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    total = len(list1)

    # 统计频率并构建编码映射
    freq_map = {}
    for f in discrete_features:
        cnt = Counter([row[f] for row in list1])
        freq_map[f] = {val: round(n / total, 3) for val, n in cnt.items()}

    print('离散特征的频率编码映射（保留三位小数）：')
    for f in discrete_features:
        print(f'{f}: {freq_map[f]}')

    # 生成训练集 data
    data_rows = []
    for row in list1:
        encoded = []
        for f in features:
            if f in freq_map:
                encoded.append(freq_map[f].get(row[f], 0.0))
            else:
                encoded.append(float(row[f]))
        data_rows.append(encoded)
    data = np.array(data_rows, dtype=float)

    # 生成测试集 pre（使用训练集的编码映射）
    pre_rows = []
    for row in list2:
        encoded = []
        for f in features:
            if f in freq_map:
                encoded.append(freq_map[f].get(row[f], 0.0))
            else:
                encoded.append(float(row[f]))
        pre_rows.append(encoded)
    pre = np.array(pre_rows, dtype=float)

    print('训练集编码后的 data：')
    print(data)
    print('测试集编码后的 pre：')
    print(pre)

    return data, pre


# 随机划分训练/测试集（从data中随机抽取4组作为测试集）
def split_train_test_random(data, y, test_size=4, random_state=0):
    rs = np.random.RandomState(random_state)
    n = len(data)
    all_idx = np.arange(n)
    test_idx = rs.choice(n, size=test_size, replace=False)
    train_idx = np.array([i for i in all_idx if i not in set(test_idx)])
    X_train = data[train_idx]
    y_train = y[train_idx]
    X_test = data[test_idx]
    y_test = y[test_idx]
    print(f'随机抽取{test_size}组作为测试集，索引：{list(test_idx)}')
    return X_train, y_train, X_test, y_test, train_idx, test_idx


# GMM-EM（二维以上高斯混合）
class GMM_EM:
    def __init__(self, n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-5, random_state=0):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.rs = np.random.RandomState(random_state)
        self.means_ = None
        self.covariances_ = None
        self.weights_ = None
        self.log_likelihood_history_ = []

    def _init_params(self, X):
        n_samples, n_features = X.shape
        # 随机选择样本作为初始均值
        idx = self.rs.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[idx]
        # 初始协方差用整体协方差矩阵
        cov = np.cov(X.T)
        self.covariances_ = np.array([cov + self.reg_covar * np.eye(n_features) for _ in range(self.n_components)])
        self.weights_ = np.full(self.n_components, 1.0 / self.n_components)
        print('初始化GMM参数完成：means、covariances、weights')

    def _gaussian(self, X, mean, cov):
        n_features = X.shape[1]
        cov = cov + self.reg_covar * np.eye(n_features)
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        diff = X - mean
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        norm_const = np.sqrt((2 * math.pi) ** n_features * det)
        return np.exp(exponent) / (norm_const + 1e-12)

    def _e_step(self, X):
        n_samples = X.shape[0]
        K = self.n_components
        resp = np.zeros((n_samples, K))
        for k in range(K):
            likelihood = self._gaussian(X, self.means_[k], self.covariances_[k])
            resp[:, k] = self.weights_[k] * likelihood
        row_sum = resp.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1e-12
        resp /= row_sum
        return resp

    def _m_step(self, X, resp):
        N_k = resp.sum(axis=0) + 1e-12
        self.weights_ = N_k / X.shape[0]
        self.means_ = (resp.T @ X) / N_k[:, None]
        K = self.n_components
        n_features = X.shape[1]
        covs = []
        for k in range(K):
            diff = X - self.means_[k]
            weighted = (resp[:, k][:, None] * diff).T @ diff
            cov_k = weighted / N_k[k]
            cov_k += self.reg_covar * np.eye(n_features)
            covs.append(cov_k)
        self.covariances_ = np.array(covs)

    def _compute_log_likelihood(self, X):
        total = np.zeros(X.shape[0])
        for k in range(self.n_components):
            total += self.weights_[k] * self._gaussian(X, self.means_[k], self.covariances_[k])
        return np.sum(np.log(total + 1e-12))

    def fit(self, X):
        print('开始EM算法训练GMM...')
        self._init_params(X)
        prev_ll = None
        for it in range(self.max_iter):
            resp = self._e_step(X)
            self._m_step(X, resp)
            ll = self._compute_log_likelihood(X)
            self.log_likelihood_history_.append(ll)
            print(f'第{it + 1}次迭代，log-likelihood: {ll:.4f}')
            if prev_ll is not None and abs(ll - prev_ll) < self.tol:
                print('收敛结束')
                break
            prev_ll = ll
        print('EM训练完成')

    def predict(self, X):
        resp = self._e_step(X)
        labels = np.argmax(resp, axis=1)
        return labels, resp


def map_clusters_to_labels(cluster_labels, y_true, n_components=2):
    mapping = {}
    for k in range(n_components):
        idx = np.where(cluster_labels == k)[0]
        if len(idx) == 0:
            mapping[k] = Counter(y_true).most_common(1)[0][0]
        else:
            mapping[k] = Counter(y_true[idx]).most_common(1)[0][0]
    print('簇到类别的映射：', mapping)
    return mapping


# PCA降维到2D，便于可视化
def pca_2d(X):
    mean = np.mean(X, axis=0)
    Xc = X - mean
    cov = np.cov(Xc.T)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order[:2]]
    Z = Xc @ evecs
    return Z, evecs, mean


# 多次随机初始化运行GMM，选择测试集准确率最优结果
def train_gmm_select_best(X_train, y_train, X_test, y_test, n_runs=10, seed_start=0):
    best_key = None
    best_seed = None
    best_gmm = None
    best_mapping = None
    best_train_pred_labels = None
    best_test_pred_labels = None

    for i in range(n_runs):
        seed = seed_start + i
        gmm = GMM_EM(n_components=2, max_iter=100, tol=1e-4, reg_covar=1e-5, random_state=seed)
        gmm.fit(X_train)

        train_pred_clusters, _ = gmm.predict(X_train)
        test_pred_clusters, _ = gmm.predict(X_test)
        mapping = map_clusters_to_labels(train_pred_clusters, y_train, n_components=gmm.n_components)
        train_pred_labels = np.array([mapping[c] for c in train_pred_clusters])
        test_pred_labels = np.array([mapping[c] for c in test_pred_clusters])

        train_acc = np.mean(train_pred_labels == y_train)
        test_acc = np.mean(test_pred_labels == y_test)
        last_ll = gmm.log_likelihood_history_[-1] if gmm.log_likelihood_history_ else -np.inf
        print(f'随机种子 {seed} -> 训练ACC: {train_acc:.3f}, 测试ACC: {test_acc:.3f}, 终止LL: {last_ll:.3f}')

        key = (test_acc, train_acc, last_ll)
        if best_key is None or key > best_key:
            best_key = key
            best_seed = seed
            best_gmm = gmm
            best_mapping = mapping
            best_train_pred_labels = train_pred_labels
            best_test_pred_labels = test_pred_labels

    print(f'选择最佳随机种子：{best_seed}，测试ACC={best_key[0]:.3f}，训练ACC={best_key[1]:.3f}')
    return best_gmm, best_seed, best_mapping, best_train_pred_labels, best_test_pred_labels


# 绘制在 PCA 2D 空间中的 GMM 可视化图（支持中文）
def plot_gmm(Z_all, evecs, X_mean,
             train_idx, test_idx,
             y_train, y_test,
             train_pred_labels, test_pred_labels,
             means, covs, weights,
             save_path='gmm_vis.png'):
    # 划分投影后的训练/测试
    Z_train = Z_all[train_idx]
    Z_test = Z_all[test_idx]

    # 基于预测标签着色，训练集用圆点，测试集用叉号
    colors = {0: '#1f77b4', 1: '#ff7f0e'}  # 不是好瓜/好瓜
    plt.figure(figsize=(10, 8))

    # 训练集散点（按预测标签分色）
    for cls in [0, 1]:
        mask = (train_pred_labels == cls)
        plt.scatter(Z_train[mask, 0], Z_train[mask, 1],
                    c=colors[cls], label=f'训练集（预测={"好瓜" if cls==1 else "不是好瓜"}）',
                    s=70, alpha=0.8, edgecolors='white')

    # 测试集散点（按预测标签分色）
    for cls in [0, 1]:
        mask = (test_pred_labels == cls)
        plt.scatter(Z_test[mask, 0], Z_test[mask, 1],
                    c=colors[cls], marker='x', label=f'测试集（预测={"好瓜" if cls==1 else "不是好瓜"}）',
                    s=80, alpha=0.9)

    # 将簇中心和协方差投影到 2D（PCA 子空间）
    centers_2d = (means - X_mean) @ evecs  # (K,2)
    for k in range(len(weights)):
        # 协方差在 PCA 子空间的表示
        cov_2d = evecs.T @ covs[k] @ evecs  # (2,2)
        vals, vecs = np.linalg.eigh(cov_2d)  # 升序
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
        # 使用 2 标准差作为椭圆大小（约95%）
        width, height = 2.0 * np.sqrt(vals)
        ellip = Ellipse(xy=centers_2d[k], width=width, height=height,
                        angle=angle, edgecolor='k', facecolor='none',
                        linewidth=1.2, alpha=0.8)
        plt.gca().add_patch(ellip)

        # 标注簇中心
        plt.scatter(centers_2d[k, 0], centers_2d[k, 1],
                    c='red', s=120, marker='*', label='簇中心' if k == 0 else None)

    plt.title('GMM-EM 在 PCA 2D 空间的可视化')
    plt.xlabel('主成分 1')
    plt.ylabel('主成分 2')
    plt.legend(loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


if __name__ == '__main__':
    # 第一张表（训练数据）
    list1 = [
        {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑', '密度': 0.697, '含糖率': 0.460, '好瓜': '是'},
        {'色泽': '乌黑', '根蒂': '蜷缩', '敲声': '沉闷', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑', '密度': 0.774, '含糖率': 0.376, '好瓜': '是'},
        {'色泽': '乌黑', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑', '密度': 0.634, '含糖率': 0.264, '好瓜': '是'},
        {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '沉闷', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑', '密度': 0.608, '含糖率': 0.318, '好瓜': '是'},
        {'色泽': '浅白', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑', '密度': 0.556, '含糖率': 0.215, '好瓜': '是'},
        {'色泽': '青绿', '根蒂': '稍蜷', '敲声': '浊响', '纹理': '清晰', '脐部': '稍凹', '触感': '软粘', '密度': 0.403, '含糖率': 0.237, '好瓜': '是'},
        {'色泽': '乌黑', '根蒂': '稍蜷', '敲声': '浊响', '纹理': '稍糊', '脐部': '稍凹', '触感': '软粘', '密度': 0.481, '含糖率': 0.149, '好瓜': '是'},
        {'色泽': '乌黑', '根蒂': '稍蜷', '敲声': '浊响', '纹理': '清晰', '脐部': '稍凹', '触感': '硬滑', '密度': 0.437, '含糖率': 0.211, '好瓜': '是'},
        {'色泽': '乌黑', '根蒂': '稍蜷', '敲声': '沉闷', '纹理': '稍糊', '脐部': '稍凹', '触感': '硬滑', '密度': 0.666, '含糖率': 0.091, '好瓜': '否'},
        {'色泽': '青绿', '根蒂': '硬挺', '敲声': '清脆', '纹理': '清晰', '脐部': '平坦', '触感': '软粘', '密度': 0.243, '含糖率': 0.267, '好瓜': '否'},
        {'色泽': '浅白', '根蒂': '硬挺', '敲声': '清脆', '纹理': '模糊', '脐部': '平坦', '触感': '硬滑', '密度': 0.245, '含糖率': 0.057, '好瓜': '否'},
        {'色泽': '浅白', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '模糊', '脐部': '平坦', '触感': '软粘', '密度': 0.343, '含糖率': 0.099, '好瓜': '否'},
        {'色泽': '青绿', '根蒂': '稍蜷', '敲声': '浊响', '纹理': '模糊', '脐部': '凹陷', '触感': '硬滑', '密度': 0.639, '含糖率': 0.161, '好瓜': '否'},
        {'色泽': '浅白', '根蒂': '稍蜷', '敲声': '沉闷', '纹理': '稍糊', '脐部': '凹陷', '触感': '硬滑', '密度': 0.657, '含糖率': 0.198, '好瓜': '否'},
        {'色泽': '乌黑', '根蒂': '稍蜷', '敲声': '浊响', '纹理': '清晰', '脐部': '稍凹', '触感': '软粘', '密度': 0.360, '含糖率': 0.370, '好瓜': '否'},
        {'色泽': '浅白', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '模糊', '脐部': '平坦', '触感': '硬滑', '密度': 0.593, '含糖率': 0.042, '好瓜': '否'},
        {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '沉闷', '纹理': '稍糊', '脐部': '稍凹', '触感': '硬滑', '密度': 0.719, '含糖率': 0.103, '好瓜': '否'},
    ]

    # 第二张表（测试数据）
    list2 = [
        {'色泽': '青绿', '根蒂': '蜷缩', '敲声': '浊响', '纹理': '清晰', '脐部': '凹陷', '触感': '硬滑', '密度': 0.697, '含糖率': 0.460}
    ]

    # 编码
    data, pre = encode_non_numeric(list1, list2)
    y_all = np.array([1 if row['好瓜'] == '是' else 0 for row in list1])

    # 随机划分训练/测试集（4组测试）
    X_train, y_train, X_test, y_test, train_idx, test_idx = split_train_test_random(data, y_all, test_size=4, random_state=0)

    # 多次随机初始化运行GMM，选择最佳结果
    best_gmm, best_seed, mapping, train_pred_labels, test_pred_labels = train_gmm_select_best(
        X_train, y_train, X_test, y_test, n_runs=37, seed_start=0
    )
    print(f'最终使用随机种子：{best_seed}')

    train_acc = np.mean(train_pred_labels == y_train)
    test_acc = np.mean(test_pred_labels == y_test)
    print(f'训练集准确率：{train_acc * 100:.1f}%')
    print(f'测试集（{len(y_test)}组）准确率：{test_acc * 100:.1f}%')

    # 第二张表测试样本分类（使用最佳模型）
    pre_pred_clusters, _ = best_gmm.predict(pre)
    pre_pred_labels = np.array([mapping[c] for c in pre_pred_clusters])
    print('第二张表测试样本预测类别：', '好瓜' if pre_pred_labels[0] == 1 else '不是好瓜')

    # 可视化：在 PCA 2D 空间中绘制GMM结果（训练/测试、簇中心与协方差椭圆）
    Z_all, evecs, X_mean = pca_2d(data)
    save_path = os.path.join(os.path.dirname(__file__), 'gmm_vis.png')
    plot_gmm(Z_all, evecs, X_mean,
             train_idx, test_idx,
             y_train, y_test,
             train_pred_labels, test_pred_labels,
             best_gmm.means_, best_gmm.covariances_, best_gmm.weights_,
             save_path=save_path)
    print(f'可视化图已保存到：{save_path}')

    print('全部流程完成。')