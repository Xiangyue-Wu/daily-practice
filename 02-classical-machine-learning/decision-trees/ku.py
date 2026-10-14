import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class SklearnDecisionTree:
    """使用sklearn库实现的决策树包装类"""
    
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, random_state=42):
        # 1.属性初始化
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = None
        self.label_encoders = {}
        return None
    
    def _encode_features(self, X, fit=True):
        """对分类特征进行编码"""
        # 1.复制数据避免修改原始数据
        X_encoded = X.copy()
        
        # 2.对每个特征进行编码
        for column in X_encoded.columns:
            if X_encoded[column].dtype == 'object':
                if fit:
                    # 训练时创建编码器
                    le = LabelEncoder()
                    X_encoded[column] = le.fit_transform(X_encoded[column])
                    self.label_encoders[column] = le
                else:
                    # 预测时使用已有编码器
                    if column in self.label_encoders:
                        le = self.label_encoders[column]
                        # 处理未见过的类别
                        unique_values = set(X_encoded[column].unique())
                        known_values = set(le.classes_)
                        unknown_values = unique_values - known_values
                        
                        if unknown_values:
                            # 将未知值替换为第一个已知类别
                            first_class = le.classes_[0]
                            X_encoded[column] = X_encoded[column].replace(list(unknown_values), first_class)
                        
                        X_encoded[column] = le.transform(X_encoded[column])
        
        return X_encoded
    
    def fit(self, X, y):
        """训练决策树模型"""
        # 1.对特征进行编码
        X_encoded = self._encode_features(X, fit=True)
        
        # 2.创建并训练模型
        self.model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )
        self.model.fit(X_encoded, y)
        return self
    
    def predict(self, X):
        """预测方法"""
        # 1.对特征进行编码
        X_encoded = self._encode_features(X, fit=False)
        
        # 2.进行预测
        return self.model.predict(X_encoded)
    
    def 五折交叉验证评估(self, X, y):
        """五折交叉验证评估"""
        # 1.对特征进行编码
        X_encoded = self._encode_features(X, fit=True)
        
        # 2.创建模型
        model = DecisionTreeClassifier(
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state
        )
        
        # 3.进行五折交叉验证
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_encoded, y, cv=kf, scoring='accuracy')
        
        return scores

class ID3Tree(SklearnDecisionTree):
    """ID3决策树（使用信息增益，近似实现）"""
    
    def __init__(self, max_depth=None, min_samples_split=2, random_state=42):
        # ID3使用信息增益，sklearn中用entropy近似
        super().__init__(criterion='entropy', max_depth=max_depth, 
                        min_samples_split=min_samples_split, random_state=random_state)
        return None

class C45Tree(SklearnDecisionTree):
    """C4.5决策树（使用信息增益率，近似实现）"""
    
    def __init__(self, max_depth=None, min_samples_split=2, random_state=42):
        # C4.5使用信息增益率，sklearn中用entropy近似
        super().__init__(criterion='entropy', max_depth=max_depth, 
                        min_samples_split=min_samples_split, random_state=random_state)
        return None

class CARTTree(SklearnDecisionTree):
    """CART决策树（使用基尼系数）"""
    
    def __init__(self, max_depth=None, min_samples_split=2, random_state=42):
        # CART使用基尼系数
        super().__init__(criterion='gini', max_depth=max_depth, 
                        min_samples_split=min_samples_split, random_state=random_state)
        return None

def train_test(数据路径):
    """读取数据并划分训练集和测试集"""
    # 1.读取数据
    data = pd.read_csv(数据路径)
    
    # 2.分离特征和标签（第一列是标签）
    X = data.iloc[:, 1:]  # 特征从第二列开始
    y = data.iloc[:, 0]   # 标签是第一列
    
    # 3.划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return (X_train, y_train), (X_test, y_test)

def zzt_visualize(传入三种方法的准确率):
    """通过matplotlib库输出一个柱状图对比"""
    # 1.设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 2.准备数据
    algorithms = ['ID3', 'C4.5', 'CART']
    accuracies = 传入三种方法的准确率
    
    # 3.创建柱状图
    plt.figure(figsize=(10, 6))
    bars = plt.bar(algorithms, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
    
    # 4.添加数值标签
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.4f}', ha='center', va='bottom')
    
    # 5.设置图表属性
    plt.title('三种决策树算法准确率对比（sklearn实现）')
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
    数据路径 = r'data/mushrooms.csv'
    train, test = train_test(数据路径)
    X_train, y_train = train
    X_test, y_test = test
    
    print("数据集加载完成！")
    print(f"训练集大小: {X_train.shape}")
    print(f"测试集大小: {X_test.shape}")
    
    # 2.实例化ID3算法（使用sklearn的entropy）
    print("\n=== ID3算法（sklearn实现） ===")
    id3_model = ID3Tree(max_depth=10, min_samples_split=5)
    
    # 2.1 得到ID3模型的五折交叉验证结果
    id3_cv_accuracies = id3_model.五折交叉验证评估(X_train, y_train)
    print("五折交叉验证准确率:")
    for i, acc in enumerate(id3_cv_accuracies, 1):
        print(f"  第{i}折: {acc:.4f}")
    print(f"平均准确率: {np.mean(id3_cv_accuracies):.4f}")
    
    # 2.2 重新训练完整模型并测试
    id3_model.fit(X_train, y_train)
    id3_pred = id3_model.predict(X_test)
    id3_test_accuracy = accuracy_score(y_test, id3_pred)
    print(f"测试集准确率: {id3_test_accuracy:.4f}")
    
    # 3.实例化C4.5算法（使用sklearn的entropy）
    print("\n=== C4.5算法（sklearn实现） ===")
    c45_model = C45Tree(max_depth=10, min_samples_split=5)
    
    # 3.1 得到C4.5模型的五折交叉验证结果
    c45_cv_accuracies = c45_model.五折交叉验证评估(X_train, y_train)
    print("五折交叉验证准确率:")
    for i, acc in enumerate(c45_cv_accuracies, 1):
        print(f"  第{i}折: {acc:.4f}")
    print(f"平均准确率: {np.mean(c45_cv_accuracies):.4f}")
    
    # 3.2 重新训练完整模型并测试
    c45_model.fit(X_train, y_train)
    c45_pred = c45_model.predict(X_test)
    c45_test_accuracy = accuracy_score(y_test, c45_pred)
    print(f"测试集准确率: {c45_test_accuracy:.4f}")
    
    # 4.实例化CART算法（使用sklearn的gini）
    print("\n=== CART算法（sklearn实现） ===")
    cart_model = CARTTree(max_depth=10, min_samples_split=5)
    
    # 4.1 得到CART模型的五折交叉验证结果
    cart_cv_accuracies = cart_model.五折交叉验证评估(X_train, y_train)
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
    zzt_visualize(test_accuracies)
    
    return None

if __name__ == "__main__":
    数据路径 = r'data/mushrooms.csv'
    main()