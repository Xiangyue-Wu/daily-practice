import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from collections import Counter
import math

class DecisionTree(ABC):
    """决策树基类"""
    
    def __init__(self, max_depth=None, min_samples_split=2):
        # 1.属性初始化
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.feature_names = None
        self.target_name = None
        return None
    
    def calculate_entropy(self, y):
        """计算信息熵"""
        # 1.统计各类别的数量
        counts = Counter(y)
        total = len(y)
        
        # 2.计算信息熵
        entropy = 0
        for count in counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        return entropy
    
    def build_tree(self, X, y, depth=0, feature_names=None):
        """递归构建决策树"""
        # 1.初始化特征名称
        if feature_names is None:
            if isinstance(X, pd.DataFrame):
                feature_names = list(X.columns)
            else:
                feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 2.检查终止条件
        if (self.max_depth is not None and depth >= self.max_depth) or \
           len(set(y)) == 1 or \
           len(y) < self.min_samples_split:
            # 返回叶节点，使用统一的字典结构
            most_common_class = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'class': most_common_class}
        
        if X.shape[1] == 0:  # 没有特征可用
            most_common_class = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'class': most_common_class}
        
        # 3.选择最优特征
        best_feature, best_threshold = self.split_criterion(X, y)
        
        if best_feature is None:
            most_common_class = Counter(y).most_common(1)[0][0]
            return {'type': 'leaf', 'class': most_common_class}
        
        best_feature_name = feature_names[best_feature]
        
        # 4.创建内部节点
        node = {
            'type': 'internal',
            'feature': best_feature_name,
            'feature_index': best_feature,
            'threshold': best_threshold,
            'children': {}
        }
        
        # 5.根据最优特征划分数据
        if isinstance(X, pd.DataFrame):
            feature_values = X.iloc[:, best_feature]
        else:
            feature_values = X[:, best_feature]
        unique_values = np.unique(feature_values)
        
        for value in unique_values:
            mask = feature_values == value
            if isinstance(X, pd.DataFrame):
                X_subset = X[mask].drop(X.columns[best_feature], axis=1)
                y_subset = y[mask]
            else:
                X_subset = np.delete(X[mask], best_feature, axis=1)
                y_subset = y[mask]
            
            if len(y_subset) > 0:
                # 更新特征名称列表
                new_feature_names = [name for i, name in enumerate(feature_names) if i != best_feature]
                
                # 递归构建子树
                node['children'][value] = self.build_tree(X_subset, y_subset, depth + 1, new_feature_names)
            else:
                # 子集为空，创建叶节点
                most_common_class = Counter(y).most_common(1)[0][0]
                node['children'][value] = {'type': 'leaf', 'class': most_common_class}
        
        return node
    
    def fit(self, X, y):
        """训练决策树"""
        # 保存特征名称
        if isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
        else:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 构建决策树
        self.tree = self.build_tree(X, y)
        return self
    
    def cross_validate_evaluate(self, X, y):
        """五折交叉验证评估"""
        # 1.初始化KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        accuracies = []
        
        # 2.进行五折交叉验证
        for train_idx, val_idx in kf.split(X):
            # 分割数据
            if isinstance(X, pd.DataFrame):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
            
            # 训练模型
            self.fit(X_train, y_train)
            
            # 预测并计算准确率
            y_pred = self.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            accuracies.append(accuracy)
        
        return accuracies
    
    def predict(self, X):
        """预测方法"""
        # 1.对每个样本进行预测
        predictions = []
        if isinstance(X, pd.DataFrame):
            for _, sample in X.iterrows():
                pred = self._predict_single(sample, self.tree, self.feature_names)
                predictions.append(pred)
        else:
            for sample in X:
                pred = self._predict_single(sample, self.tree, self.feature_names)
                predictions.append(pred)
        return np.array(predictions)
    
    def _predict_single(self, sample, node, feature_names=None):
        """预测单个样本"""
        # 1.如果是叶节点，返回预测结果
        if node['type'] == 'leaf':
            return node['class']
        
        # 2.根据特征值选择子树
        if isinstance(sample, pd.Series):
            # 使用特征名称进行预测（更稳健）
            feature_name = node['feature']
            if feature_name in sample.index:
                feature_value = sample[feature_name]
            else:
                # 如果特征名称不存在，使用索引
                feature_idx = node['feature_index']
                feature_value = sample.iloc[feature_idx] if feature_idx < len(sample) else None
        else:
            # numpy数组，使用索引
            feature_idx = node['feature_index']
            feature_value = sample[feature_idx] if feature_idx < len(sample) else None
        
        if feature_value is not None and feature_value in node['children']:
            return self._predict_single(sample, node['children'][feature_value], feature_names)
        else:
            # 如果特征值不在训练集中，返回最常见的类别
            return 'e'  # 默认返回可食用
    
    @abstractmethod
    def split_criterion(self, X, y):
        """抽象方法：划分标准，子类必须实现"""
        pass

class ID3Tree(DecisionTree):
    """ID3决策树"""
    
    def split_criterion(self, X, y):
        """计算信息增益"""
        # 1.计算原始信息熵
        base_entropy = self.calculate_entropy(y)
        best_gain = 0
        best_feature = None
        
        # 2.遍历所有特征
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            if isinstance(X, pd.DataFrame):
                feature_values = X.iloc[:, feature_idx]
            else:
                feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # 计算条件熵
            weighted_entropy = 0
            for value in unique_values:
                mask = feature_values == value
                if isinstance(y, pd.Series):
                    subset_y = y[mask]
                else:
                    subset_y = y[mask]
                if len(subset_y) > 0:
                    weight = len(subset_y) / len(y)
                    weighted_entropy += weight * self.calculate_entropy(subset_y)
            
            # 计算信息增益
            gain = base_entropy - weighted_entropy
            
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_idx
        
        return best_feature, None

class C45Tree(DecisionTree):
    """C4.5决策树"""
    
    def split_criterion(self, X, y):
        """计算信息增益率"""
        # 1.计算原始信息熵
        base_entropy = self.calculate_entropy(y)
        best_gain_ratio = 0
        best_feature = None
        
        # 2.遍历所有特征
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            if isinstance(X, pd.DataFrame):
                feature_values = X.iloc[:, feature_idx]
            else:
                feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # 计算条件熵
            weighted_entropy = 0
            split_info = 0
            
            for value in unique_values:
                mask = feature_values == value
                if isinstance(y, pd.Series):
                    subset_y = y[mask]
                else:
                    subset_y = y[mask]
                if len(subset_y) > 0:
                    weight = len(subset_y) / len(y)
                    weighted_entropy += weight * self.calculate_entropy(subset_y)
                    # 计算分裂信息
                    if weight > 0:
                        split_info -= weight * math.log2(weight)
            
            # 计算信息增益
            gain = base_entropy - weighted_entropy
            
            # 计算信息增益率
            if split_info > 0:
                gain_ratio = gain / split_info
                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feature_idx
        
        return best_feature, None

class CARTTree(DecisionTree):
    """CART决策树"""
    
    def __init__(self, max_depth=None, min_samples_split=2, task='classification'):
        super().__init__(max_depth, min_samples_split)
        # 1.任务类型属性
        self.task = task
        return None
    
    def split_criterion(self, X, y):
        """根据task选择基尼系数或均方误差"""
        # 1.根据任务类型选择划分标准
        if self.task == 'classification':
            return self._best_split_gini(X, y)
        else:  # regression
            return self._best_split_mse(X, y)
    
    def _best_split_gini(self, X, y):
        """使用基尼系数进行分类任务的划分"""
        # 1.初始化最佳参数
        best_gini = float('inf')
        best_feature = None
        
        # 2.遍历所有特征
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            if isinstance(X, pd.DataFrame):
                feature_values = X.iloc[:, feature_idx]
            else:
                feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # 计算加权基尼系数
            weighted_gini = 0
            for value in unique_values:
                mask = feature_values == value
                if isinstance(y, pd.Series):
                    subset_y = y[mask]
                else:
                    subset_y = y[mask]
                if len(subset_y) > 0:
                    weight = len(subset_y) / len(y)
                    weighted_gini += weight * self.calculate_gini(subset_y)
            
            if weighted_gini < best_gini:
                best_gini = weighted_gini
                best_feature = feature_idx
        
        return best_feature, None
    
    def _best_split_mse(self, X, y):
        """使用均方误差进行回归任务的划分"""
        # 1.初始化最佳参数
        best_mse = float('inf')
        best_feature = None
        
        # 2.遍历所有特征
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            if isinstance(X, pd.DataFrame):
                feature_values = X.iloc[:, feature_idx]
            else:
                feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # 计算加权均方误差
            weighted_mse = 0
            for value in unique_values:
                mask = feature_values == value
                if isinstance(y, pd.Series):
                    subset_y = y[mask]
                else:
                    subset_y = y[mask]
                if len(subset_y) > 0:
                    weight = len(subset_y) / len(y)
                    mse = self.calculate_mse(subset_y)
                    weighted_mse += weight * mse
            
            if weighted_mse < best_mse:
                best_mse = weighted_mse
                best_feature = feature_idx
        
        return best_feature, None
    
    def calculate_gini(self, y):
        """计算基尼系数，用于分类任务"""
        # 1.统计各类别的数量
        counts = Counter(y)
        total = len(y)
        
        # 2.计算基尼系数
        gini = 1.0
        for count in counts.values():
            p = count / total
            gini -= p ** 2
        return gini
    
    def calculate_mse(self, y):
        """计算均方误差，用于回归任务"""
        # 1.计算均值
        mean_y = np.mean(y)
        
        # 2.计算均方误差
        mse = np.mean((y - mean_y) ** 2)
        return mse

def load_and_split_data(data_path):
    """读取数据并划分训练集和测试集"""
    # 1.读取数据
    data = pd.read_csv(data_path)
    
    # 2.分离特征和标签（第一列是标签）
    X = data.iloc[:, 1:]  # 特征从第二列开始
    y = data.iloc[:, 0]   # 标签是第一列
    
    # 3.划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def create_comparison_chart(accuracy_list):
    """通过matplotlib库输出一个柱状图对比"""
    # 1.设置中文字体
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    except:
        # 如果中文字体设置失败，使用默认字体
        pass
    
    # 2.准备数据
    algorithms = ['ID3', 'C4.5', 'CART']
    accuracies = accuracy_list
    
    # 3.创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    
    # 4.添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom')
    
    # 5.设置图表属性
    plt.title('三种决策树算法准确率对比')
    plt.ylabel('准确率')
    plt.xlabel('算法')
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # 6.显示图表
    plt.tight_layout()
    plt.show()
    return None

def main():
    # 1.读取数据集，并划分训练集和测试集
    data_path = r'data/mushrooms.csv'
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)
    
    print("数据集加载完成！")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 2.实例化ID3算法
    print("\n=== ID3算法 ===")
    id3_model = ID3Tree(max_depth=10, min_samples_split=5)
    
    # 2.1 得到ID3模型的五折交叉验证结果
    id3_cv_accuracies = id3_model.cross_validate_evaluate(X_train, y_train)
    print("五折交叉验证准确率:")
    for i, acc in enumerate(id3_cv_accuracies, 1):
        print(f"  第{i}折: {acc:.4f}")
    print(f"平均准确率: {np.mean(id3_cv_accuracies):.4f}")
    
    # 2.2 重新训练完整模型并测试
    id3_model.fit(X_train, y_train)
    id3_pred = id3_model.predict(X_test)
    id3_test_accuracy = accuracy_score(y_test, id3_pred)
    print(f"测试集准确率: {id3_test_accuracy:.4f}")
    
    # 3.实例化C4.5算法
    print("\n=== C4.5算法 ===")
    c45_model = C45Tree(max_depth=10, min_samples_split=5)
    
    # 3.1 得到C4.5模型的五折交叉验证结果
    c45_cv_accuracies = c45_model.cross_validate_evaluate(X_train, y_train)
    print("五折交叉验证准确率:")
    for i, acc in enumerate(c45_cv_accuracies, 1):
        print(f"  第{i}折: {acc:.4f}")
    print(f"平均准确率: {np.mean(c45_cv_accuracies):.4f}")
    
    # 3.2 重新训练完整模型并测试
    c45_model.fit(X_train, y_train)
    c45_pred = c45_model.predict(X_test)
    c45_test_accuracy = accuracy_score(y_test, c45_pred)
    print(f"测试集准确率: {c45_test_accuracy:.4f}")
    
    # 4.实例化CART算法
    print("\n=== CART算法 ===")
    cart_model = CARTTree(max_depth=10, min_samples_split=5, task='classification')
    
    # 4.1 得到CART模型的五折交叉验证结果
    cart_cv_accuracies = cart_model.cross_validate_evaluate(X_train, y_train)
    print("五折交叉验证准确率:")
    for i, acc in enumerate(cart_cv_accuracies, 1):
        print(f"  第{i}折: {acc:.4f}")
    print(f"平均准确率: {np.mean(cart_cv_accuracies):.4f}")
    
    # 4.2 重新训练完整模型并测试
    cart_model.fit(X_train, y_train)
    cart_pred = cart_model.predict(X_test)
    cart_test_accuracy = accuracy_score(y_test, cart_pred)
    print(f"测试集准确率: {cart_test_accuracy:.4f}")
    
    # 5.可视化对比结果
    print("\n=== 结果对比 ===")
    test_accuracies = [id3_test_accuracy, c45_test_accuracy, cart_test_accuracy]
    create_comparison_chart(test_accuracies)
    
    return None

if __name__ == "__main__":
    main()
