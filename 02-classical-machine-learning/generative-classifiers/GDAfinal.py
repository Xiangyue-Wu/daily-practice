import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 数据集
X_data = np.array([
    [0, 0, 0, 0, 0, 0, 0.697, 0.460],
    [1, 0, 1, 0, 0, 0, 0.774, 0.376],
    [1, 0, 0, 0, 0, 0, 0.634, 0.264],
    [0, 0, 1, 0, 0, 0, 0.608, 0.318], 
    [2, 0, 0, 0, 0, 0, 0.556, 0.215], 
    [0, 1, 0, 0, 1, 1, 0.403, 0.237], 
    [1, 1, 0, 1, 1, 1, 0.481, 0.149],  
    [1, 1, 0, 0, 1, 0, 0.437, 0.211],  
    [1, 1, 1, 1, 1, 0, 0.666, 0.091],  
    [0, 2, 2, 0, 2, 1, 0.243, 0.267],   
    [2, 2, 2, 2, 2, 0, 0.245, 0.057], 
    [2, 0, 0, 2, 2, 1, 0.343, 0.099],  
    [0, 1, 0, 2, 0, 0, 0.639, 0.161], 
    [2, 1, 1, 1, 0, 0, 0.657, 0.198],  
    [1, 1, 0, 0, 1, 1, 0.360, 0.370],   
    [2, 0, 0, 2, 2, 0, 0.593, 0.042],  
    [0, 0, 1, 1, 1, 0, 0.719, 0.103]    
])

y_data = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 

X_pre = np.array([
    [0, 0, 0, 0, 0, 0, 0.697, 0.460]  
])

class GDA1D:
    """
    一维高斯判别分析分类器
    
    基于贝叶斯定理和高斯分布假设的生成式分类模型
    假设每个类别的特征服从一维高斯分布
    """
    
    def __init__(self, feature_idx=7):
        """
        初始化一维GDA分类器
        
        参数:
        feature_idx : int, 默认为7
            要使用的特征索引（从0开始计数）
        """
        self.feature_idx = feature_idx  # 特征索引
        self.classes_ = None            # 类别标签
        self.class_priors_ = None       # 先验概率 P(y)
        self.class_means_ = None        # 各类别的均值
        self.class_vars_ = None         # 各类别的方差
        
    def fit(self, X, y):
        """
        训练一维GDA模型
        
        参数:
        X : array-like, shape (n_samples, n_features)
            训练数据特征矩阵
        y : array-like, shape (n_samples,)
            训练数据标签向量
            
        返回值:
        self : 返回自身实例，支持链式调用
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # 提取指定特征
        X_feature = X[:, self.feature_idx]
        
        # 获取所有类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = len(y)
        
        # 初始化存储各类别统计量的数组
        self.class_means_ = np.zeros(n_classes)
        self.class_vars_ = np.zeros(n_classes)
        self.class_priors_ = np.zeros(n_classes)
        
        # 计算每个类别的均值、方差和先验概率
        for i, class_label in enumerate(self.classes_):
            # 获取当前类别的样本索引
            class_mask = (y == class_label)
            X_class = X_feature[class_mask]
            n_class = np.sum(class_mask)
            
            # 计算类别先验概率 P(y=c)
            self.class_priors_[i] = n_class / n_samples
            
            # 计算类别均值 μ_c
            self.class_means_[i] = np.mean(X_class)
            
            # 计算类别方差 σ²_c (使用无偏估计)
            if n_class > 1:
                self.class_vars_[i] = np.var(X_class, ddof=1)  # 使用无偏估计
            else:
                self.class_vars_[i] = 1e-6  # 单个样本时的处理
            
            # 防止方差为0
            if self.class_vars_[i] < 1e-10:
                self.class_vars_[i] = 1e-10
        
        return self
    
    def _gaussian_pdf(self, x, mean, var):
        """
        计算一维高斯分布的概率密度函数
        
        参数:
        x : array-like
            输入数据
        mean : float
            均值
        var : float
            方差
            
        返回值:
        pdf : array-like
            概率密度值
        """
        return (1.0 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)
    
    def predict_proba(self, X):
        """
        预测类别概率
        
        参数:
        X : array-like, shape (n_samples, n_features)
            待预测的数据
            
        返回值:
        probabilities : array-like, shape (n_samples, n_classes)
            每个样本属于各类别的概率
        """
        X = np.asarray(X)
        X_feature = X[:, self.feature_idx]
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # 初始化概率矩阵
        probabilities = np.zeros((n_samples, n_classes))
        
        # 计算每个类别的后验概率 P(y=c|x)
        for i, class_label in enumerate(self.classes_):
            # 计算似然 P(x|y=c)
            likelihood = self._gaussian_pdf(
                X_feature, self.class_means_[i], self.class_vars_[i]
            )
            
            # 计算后验概率（未归一化）P(y=c|x) ∝ P(x|y=c) * P(y=c)
            probabilities[:, i] = likelihood * self.class_priors_[i]
        
        # 归一化概率（贝叶斯定理的分母）
        prob_sum = np.sum(probabilities, axis=1, keepdims=True)
        probabilities = probabilities / prob_sum
        
        return probabilities
    
    def predict(self, X):
        """
        预测类别标签
        
        参数:
        X : array-like, shape (n_samples, n_features)
            待预测的数据
            
        返回值:
        predictions : array-like, shape (n_samples,)
            预测的类别标签
        """
        # 获取概率预测
        probabilities = self.predict_proba(X)
        
        # 选择概率最大的类别
        predicted_indices = np.argmax(probabilities, axis=1)
        
        # 返回对应的类别标签
        predictions = self.classes_[predicted_indices]
        
        return predictions
    
    def plot_distribution(self, X, y, X_test=None, y_test=None, title=None):
        """
        绘制一维高斯分布和数据点
        
        参数:
        X : array-like, shape (n_samples, n_features)
            训练数据特征矩阵
        y : array-like, shape (n_samples,)
            训练数据标签向量
        X_test : array-like, shape (n_test_samples, n_features), 可选
            测试数据特征矩阵
        y_test : array-like, shape (n_test_samples,), 可选
            测试数据标签向量
        title : str, 可选
            图表标题
        """
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']  # 设置支持中文的字体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
        plt.figure(figsize=(10, 6))
        
        # 提取指定特征
        X_feature = X[:, self.feature_idx]
        
        # 为每个类别绘制数据点和分布曲线
        colors = ['red', 'blue']
        labels = ['负例 (坏瓜)', '正例 (好瓜)']
        
        # 创建x轴范围
        x_min, x_max = X_feature.min() - 0.1, X_feature.max() + 0.1
        xx = np.linspace(x_min, x_max, 1000)
        
        # 绘制每个类别的高斯分布
        for i, class_label in enumerate(self.classes_):
            # 获取当前类别的样本索引
            class_mask = (y == class_label)
            X_class = X_feature[class_mask]
            
            # 绘制数据点
            plt.scatter(X_class, np.zeros_like(X_class) + 0.02 * i - 0.01, 
                       color=colors[i], alpha=0.6, label=f'{labels[i]} (训练)')
            
            # 绘制高斯分布曲线
            plt.plot(xx, self._gaussian_pdf(xx, self.class_means_[i], self.class_vars_[i]) * self.class_priors_[i] * 5,
                    color=colors[i], linestyle='-', linewidth=2)
        
        # 如果有测试数据，也绘制出来
        if X_test is not None and y_test is not None:
            X_test_feature = X_test[:, self.feature_idx]
            for i, class_label in enumerate(self.classes_):
                class_mask = (y_test == class_label)
                if np.any(class_mask):
                    X_test_class = X_test_feature[class_mask]
                    plt.scatter(X_test_class, np.zeros_like(X_test_class) + 0.02 * i - 0.01, 
                              color=colors[i], marker='x', s=100, alpha=0.8, label=f'{labels[i]} (测试)')
        
        # 绘制预测样本
        if X_pre is not None:
            X_pre_feature = X_pre[:, self.feature_idx]
            pred_class = self.predict(X_pre)[0]
            pred_color = colors[int(pred_class)]
            plt.scatter(X_pre_feature, 0, color=pred_color, marker='*', s=200, 
                       edgecolor='black', linewidth=1.5, label='预测样本')
            
            # 添加预测结果文本
            plt.text(X_pre_feature[0], 0.01, f'预测: {"好瓜" if pred_class == 1 else "坏瓜"}', 
                    fontsize=12, ha='center', va='bottom')
        
        # 设置图表属性
        plt.title(title or f'一维高斯判别分析 (特征: {self.feature_idx})', fontsize=14)
        plt.xlabel(f'特征值 ({"含糖率" if self.feature_idx == 7 else "密度"})', fontsize=12)
        plt.ylabel('概率密度', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
        
        # 显示决策边界
        # 找到两个高斯分布的交点
        def find_decision_boundary():
            # 求解方程: P(y=0|x) = P(y=1|x)
            a = 1/(2*self.class_vars_[0]) - 1/(2*self.class_vars_[1])
            b = self.class_means_[1]/self.class_vars_[1] - self.class_means_[0]/self.class_vars_[0]
            c = (self.class_means_[0]**2)/(2*self.class_vars_[0]) - (self.class_means_[1]**2)/(2*self.class_vars_[1]) - np.log(self.class_priors_[1]/self.class_priors_[0]) - 0.5*np.log(self.class_vars_[1]/self.class_vars_[0])
            
            # 解二次方程 ax^2 + bx + c = 0
            if abs(a) < 1e-10:  # 如果a接近0，则为线性方程
                return [-c/b]
            else:
                discriminant = b**2 - 4*a*c
                if discriminant < 0:
                    return []  # 无实数解
                elif discriminant == 0:
                    return [-b/(2*a)]  # 一个解
                else:
                    return [(-b + np.sqrt(discriminant))/(2*a), (-b - np.sqrt(discriminant))/(2*a)]  # 两个解
        
        boundaries = find_decision_boundary()
        for boundary in boundaries:
            if x_min <= boundary <= x_max:
                plt.axvline(x=boundary, color='green', linestyle='--', linewidth=2, label='决策边界')
                plt.text(boundary, 0.03, f'x={boundary:.3f}', fontsize=10, ha='center', va='bottom', color='green')
        
        plt.tight_layout()
        return plt.gcf()  # 返回当前图形对象


class GDA2D(GDA1D):
    """
    二维高斯判别分析分类器
    
    基于贝叶斯定理和高斯分布假设的生成式分类模型
    假设每个类别的特征服从二维高斯分布
    """
    
    def __init__(self, feature_idx1=6, feature_idx2=7):
        """
        初始化二维GDA分类器
        
        参数:
        feature_idx1 : int, 默认为6
            第一个特征的索引（从0开始计数）
        feature_idx2 : int, 默认为7
            第二个特征的索引（从0开始计数）
        """
        super().__init__()  # 调用父类初始化方法
        self.feature_idx1 = feature_idx1  # 第一个特征索引
        self.feature_idx2 = feature_idx2  # 第二个特征索引
        self.class_covs_ = None           # 各类别的协方差矩阵
        
    def fit(self, X, y):
        """
        训练二维GDA模型
        
        参数:
        X : array-like, shape (n_samples, n_features)
            训练数据特征矩阵
        y : array-like, shape (n_samples,)
            训练数据标签向量
            
        返回值:
        self : 返回自身实例，支持链式调用
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # 提取指定特征
        X_features = X[:, [self.feature_idx1, self.feature_idx2]]
        
        # 获取所有类别
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = len(y)
        
        # 初始化存储各类别统计量的数组
        self.class_means_ = np.zeros((n_classes, 2))
        self.class_covs_ = np.zeros((n_classes, 2, 2))
        self.class_priors_ = np.zeros(n_classes)
        
        # 计算每个类别的均值、协方差矩阵和先验概率
        for i, class_label in enumerate(self.classes_):
            # 获取当前类别的样本索引
            class_mask = (y == class_label)
            X_class = X_features[class_mask]
            n_class = len(X_class)
            
            # 计算类别先验概率 P(y=c)
            self.class_priors_[i] = n_class / n_samples
            
            # 计算类别均值向量 μ_c
            self.class_means_[i] = np.mean(X_class, axis=0)
            
            # 计算类别协方差矩阵 Σ_c (使用无偏估计)
            if n_class > 1:
                # 使用np.cov进行无偏估计 (ddof=1)
                self.class_covs_[i] = np.cov(X_class, rowvar=False, ddof=1)
            else:
                # 单个样本时的处理
                self.class_covs_[i] = np.eye(2) * 1e-6
            
            # 防止协方差矩阵奇异
            self.class_covs_[i] += np.eye(2) * 1e-6
        
        return self
    
    def _multivariate_gaussian_pdf(self, X, mean, cov):
        """
        计算二维高斯分布的概率密度函数
        
        参数:
        X : array-like, shape (n_samples, 2)
            输入数据
        mean : array-like, shape (2,)
            均值向量
        cov : array-like, shape (2, 2)
            协方差矩阵
            
        返回值:
        pdf : array-like, shape (n_samples,)
            概率密度值
        """
        n_dim = len(mean)
        det = np.linalg.det(cov)
        inv = np.linalg.inv(cov)
        
        # 计算归一化常数
        norm_const = 1.0 / (np.sqrt((2 * np.pi) ** n_dim * det))
        
        # 计算每个样本的概率密度
        X_centered = X - mean
        
        # 计算马哈拉诺比斯距离的平方
        mahalanobis_sq = np.sum(np.dot(X_centered, inv) * X_centered, axis=1)
        
        # 计算概率密度
        pdf = norm_const * np.exp(-0.5 * mahalanobis_sq)
        
        return pdf
    
    def predict_proba(self, X):
        """
        预测类别概率
        
        参数:
        X : array-like, shape (n_samples, n_features)
            待预测的数据
            
        返回值:
        probabilities : array-like, shape (n_samples, n_classes)
            每个样本属于各类别的概率
        """
        X = np.asarray(X)
        X_features = X[:, [self.feature_idx1, self.feature_idx2]]
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # 初始化概率矩阵
        probabilities = np.zeros((n_samples, n_classes))
        
        # 计算每个类别的后验概率 P(y=c|x)
        for i, class_label in enumerate(self.classes_):
            # 计算似然 P(x|y=c)
            likelihood = self._multivariate_gaussian_pdf(
                X_features, self.class_means_[i], self.class_covs_[i]
            )
            
            # 计算后验概率（未归一化）P(y=c|x) ∝ P(x|y=c) * P(y=c)
            probabilities[:, i] = likelihood * self.class_priors_[i]
        
        # 归一化概率（贝叶斯定理的分母）
        prob_sum = np.sum(probabilities, axis=1, keepdims=True)
        probabilities = probabilities / prob_sum
        
        return probabilities
    
    def plot_distribution(self, X, y, X_test=None, y_test=None, title=None, resolution=120, show_test=False, show_gaussian_contours=False):
        """
        绘制二维高斯分布与决策边界（简洁版）
        - 默认仅显示训练数据点、淡化的概率背景和 P=0.5 决策边界
        - 可选显示测试数据点与类别高斯等高线
        """
        # 设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 8))
        
        # 提取指定特征
        X_features = X[:, [self.feature_idx1, self.feature_idx2]]
        
        # 简化的配色与标签
        colors = ['red', 'blue']
        labels = ['负例 (坏瓜)', '正例 (好瓜)']
        
        # 创建网格
        x_min, x_max = X_features[:, 0].min() - 0.1, X_features[:, 0].max() + 0.1
        y_min, y_max = X_features[:, 1].min() - 0.1, X_features[:, 1].max() + 0.1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                             np.linspace(y_min, y_max, resolution))
        
        # 计算每个网格点的预测概率
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        temp_X = np.zeros((len(grid_points), X.shape[1]))
        temp_X[:, self.feature_idx1] = grid_points[:, 0]
        temp_X[:, self.feature_idx2] = grid_points[:, 1]
        Z = self.predict_proba(temp_X)[:, 1].reshape(xx.shape)
        
        # 背景概率热度图（淡化）与决策边界
        plt.contourf(xx, yy, Z, alpha=0.2, cmap=plt.cm.RdBu_r)
        plt.contour(xx, yy, Z, [0.5], linewidths=2.0, colors='green')
        
        # 训练数据点（去掉“训练”后缀，统一样式）
        for i, class_label in enumerate(self.classes_):
            class_mask = (y == class_label)
            plt.scatter(
                X_features[class_mask, 0], X_features[class_mask, 1],
                color=colors[i], s=60, alpha=0.8, label=labels[i]
            )
        
        # 可选：测试数据点（默认不显示，避免图例重复）
        if show_test and X_test is not None and y_test is not None:
            X_test_features = X_test[:, [self.feature_idx1, self.feature_idx2]]
            for i, class_label in enumerate(self.classes_):
                class_mask = (y_test == class_label)
                if np.any(class_mask):
                    plt.scatter(
                        X_test_features[class_mask, 0], X_test_features[class_mask, 1],
                        color=colors[i], marker='x', s=70, alpha=0.6
                    )
        
        # 可选：类别高斯分布等高线（默认不显示）
        if show_gaussian_contours:
            for i, class_label in enumerate(self.classes_):
                mean = self.class_means_[i]
                cov = self.class_covs_[i]
                gp = np.c_[xx.ravel(), yy.ravel()]
                ZZ = self._multivariate_gaussian_pdf(gp, mean, cov).reshape(xx.shape)
                plt.contour(xx, yy, ZZ, 3, colors=colors[i], alpha=0.4, linestyles='--')
        
        # 预测样本位置（仅星标，不加文字）
        if X_pre is not None:
            X_pre_features = X_pre[:, [self.feature_idx1, self.feature_idx2]]
            pred_class = self.predict(X_pre)[0]
            pred_color = colors[int(pred_class)]
            plt.scatter(
                X_pre_features[:, 0], X_pre_features[:, 1],
                color=pred_color, marker='*', s=220, edgecolor='black', linewidth=1.2,
                label='预测样本'
            )
        
        # 简洁的图表属性
        plt.title(title or '二维高斯判别分析 (密度和含糖率)', fontsize=14)
        plt.xlabel('密度', fontsize=12)
        plt.ylabel('含糖率', fontsize=12)
        plt.legend(loc='best', frameon=False)
        plt.tight_layout()
        
        return plt.gcf()  # 返回当前图形对象


def evaluate_model(model, X_test, y_test, model_name):
    """
    评估模型性能
    
    参数:
    model : 训练好的模型
    X_test : 测试集特征
    y_test : 测试集标签
    model_name : 模型名称
    """
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{model_name} 测试集准确率: {accuracy*100:.2f}%")
    
    # 输出分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=["坏瓜", "好瓜"]))
    
    # 输出混淆矩阵
    print("混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))


# 主函数
if __name__ == "__main__":
    # 1. 数据预处理
    # 提取连续特征（密度和含糖率）
    continuous_features = X_data[:, [6, 7]]
    
    # 标准化连续特征
    scaler = StandardScaler()
    continuous_features_scaled = scaler.fit_transform(continuous_features)
    
    # 创建新的特征矩阵（只包含标准化后的连续特征）
    X_data_scaled = continuous_features_scaled.copy()
    
    # 标准化预测样本
    X_pre_scaled = scaler.transform(X_pre[:, [6, 7]])
    
    # 2. 数据集划分（使用随机分层划分）
    X_train, X_test, y_train, y_test = train_test_split(
        X_data_scaled, y_data, test_size=0.25, random_state=42, stratify=y_data
    )
    
    print("=== 数据集划分 ===")
    print(f"训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")
    print(f"训练集类别分布: {np.unique(y_train, return_counts=True)}")
    print(f"测试集类别分布: {np.unique(y_test, return_counts=True)}")
    
    # 3. 一维高斯判别分析（只考虑含糖率）
    print("\n=== 一维高斯判别分析（含糖率）===")
    gda_1d = GDA1D(feature_idx=1)  # 标准化后含糖率是第2个特征（索引为1）
    gda_1d.fit(X_train, y_train)
    
    # 输出模型参数
    print("模型参数:")
    for i, class_label in enumerate(gda_1d.classes_):
        print(f"类别 {int(class_label)} ({'好瓜' if class_label == 1 else '坏瓜'}):")
        print(f"  先验概率: {gda_1d.class_priors_[i]:.4f}")
        print(f"  均值: {gda_1d.class_means_[i]:.4f}")
        print(f"  方差: {gda_1d.class_vars_[i]:.4f}")
    
    # 评估模型
    evaluate_model(gda_1d, X_test, y_test, "一维GDA（含糖率）")
    
    # 预测新样本
    y_pred_new = gda_1d.predict(X_pre_scaled)
    print("\n新样本预测结果:")
    print(f"预测标签={int(y_pred_new[0])} ({'好瓜' if y_pred_new[0] == 1 else '坏瓜'})")
    
    # 绘制一维高斯分布
    fig1 = gda_1d.plot_distribution(X_train, y_train, X_test, y_test, "一维高斯判别分析 (含糖率)")
    plt.savefig('GDA_1D_sugar.png')
    
    # 4. 二维高斯判别分析（考虑密度和含糖率）
    print("\n=== 二维高斯判别分析（密度和含糖率）===")
    gda_2d = GDA2D(feature_idx1=0, feature_idx2=1)  # 标准化后密度是第1个特征（索引为0），含糖率是第2个特征（索引为1）
    gda_2d.fit(X_train, y_train)
    
    # 输出模型参数
    print("模型参数:")
    for i, class_label in enumerate(gda_2d.classes_):
        print(f"类别 {int(class_label)} ({'好瓜' if class_label == 1 else '坏瓜'}):")
        print(f"  先验概率: {gda_2d.class_priors_[i]:.4f}")
        print(f"  均值向量: {gda_2d.class_means_[i]}")
        print(f"  协方差矩阵:\n{gda_2d.class_covs_[i]}")
    
    # 评估模型
    evaluate_model(gda_2d, X_test, y_test, "二维GDA（密度和含糖率）")
    
    # 预测新样本
    y_pred_new_2d = gda_2d.predict(X_pre_scaled)
    print("\n新样本预测结果:")
    print(f"预测标签={int(y_pred_new_2d[0])} ({'好瓜' if y_pred_new_2d[0] == 1 else '坏瓜'})")
    
    # 绘制二维高斯分布
    fig2 = gda_2d.plot_distribution(X_train, y_train, X_test, y_test, "二维高斯判别分析 (密度和含糖率)")
    plt.savefig('GDA_2D.png')
    
    # 结果分析
    print("\n=== 结果分析 ===")
    print("1. 一维GDA (仅含糖率):")
    if y_pred_new[0] == 1:
        print("   预测样本为好瓜。从含糖率特征来看，该样本的含糖率较高，更接近好瓜的分布。")
    else:
        print("   预测样本为坏瓜。从含糖率特征来看，该样本的含糖率较低，更接近坏瓜的分布。")
    
    print("\n2. 二维GDA (密度和含糖率):")
    if y_pred_new_2d[0] == 1:
        print("   预测样本为好瓜。综合考虑密度和含糖率两个特征，该样本更接近好瓜的二维分布。")
    else:
        print("   预测样本为坏瓜。综合考虑密度和含糖率两个特征，该样本更接近坏瓜的二维分布。")
    
    print("\n3. 对比分析:")
    if y_pred_new[0] == y_pred_new_2d[0]:
        print("   一维和二维GDA的预测结果一致，说明含糖率是一个较强的判别特征。")
    else:
        print("   一维和二维GDA的预测结果不一致，说明加入密度特征后改变了分类结果。")
        print("   这表明单一特征可能不足以准确分类，多特征模型能提供更全面的判断。")
    
    # 显示图形
    plt.show()