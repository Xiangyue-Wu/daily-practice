import math
import numpy as np
from collections import defaultdict
from collections import Counter

class NaiveBayesClassifier:
    def __init__(self): 
        self.pri = None # 存储先验概率
        self.p_x_c = None # 存储离散特征的条件概率 P(某种特征的某个取值|某个类别)
        self.continuous_params = None # 存储连续特征的参数（均值和方差）
        self.feature_types = None # 存储特征类型（离散或连续）
    
    def identify_feature_types(self, X_train):# 判断每个特征是离散型还是连续型
        feature_types = []
        n_features = X_train.shape[1]
        
        for i in range(n_features):
            # 获取该特征列的所有值
            feature_values = X_train[:, i]
            
            # 检查是否所有值都是整数
            is_integer_valued = True
            for val in feature_values:
                if val != int(val):
                    is_integer_valued = False
                    break
            
            # 检查唯一值的数量
            unique_values = np.unique(feature_values)
            
            # 判断特征类型
            # 如果值都是整数且唯一值数量较少，则为离散型
            # 否则为连续型
            if is_integer_valued and len(unique_values) <= 5:  # 假设离散特征最多有5个不同取值
                feature_types.append('discrete')
            else:
                feature_types.append('continuous')
        
        print('特征类型判断结果：', feature_types)
        return feature_types
    
    def div_pri(self, c_t, X_train, y_train):  # 分割和先验计算
        pri = {}  # {A类:先验概率，B类:先验概率}
        div = {}  # {A类:np[特征组合1的列表，特征组合2的列表],...}
        for c, t in c_t.items():
            pri[c] = (t + 1) / (len(y_train) + len(c_t))

            div[c] = []  # div以当前类为键，值为空列表
            for i in range(0, len(y_train)):
                if y_train[i] == c:
                    div[c].append(X_train[i])
            div[c] = np.array(div[c])
        print('分割完成，div：')
        print(div)

        return div, pri

    def get_attrs(self, X_train):  # 得到每种特征的取值数{特征索引：counter{取值1：数量，……}，……}
        print('len(X_train[0,:]):', len(X_train[0, :]))
        attrs = {}
        for i in range(len(X_train[0, :])):
            if self.feature_types[i] == 'discrete':  # 只对离散特征计算取值数
                attrs[i] = Counter(X_train[:, i])
        return attrs

    def get_p_x_c(self, div, attrs, y_train):  # 得到离散特征的条件概率 P(某种特征的某个取值|某个类别)
        # div {A类:np[特征组合1的列表，特征组合2的列表],...}
        c_t = Counter(y_train)  # 某类样本数

        p_x_c = {}  # 记录P(某种特征的某个取值|某个类别) {A类:  {特征1:  {取值1:概率，取值2:概率}  ，特征2……}   ，B类:……}
        c_x_t = {}  # 记录该类时，该特征样本数 {A类:  {特征1:  {取值1:样本数，取值2:样本数}  ，特征2……}   ，B类:……}
        # A类时，第i个特征取值1的样本数：c_x_t[A][i][1]
        for c, x in div.items():  # 每次取出某类及所有特征组合np[特征组合1的列表，特征组合2的列表]
            xn = np.array(x)
            p_x_c[c] = {}
            c_x_t[c] = {}
            for i in range(0, len(x[0])):  # i为每个特征的索引
                if self.feature_types[i] == 'discrete':  # 只处理离散特征
                    c_x_t[c][i] = Counter(xn[:, i])  # 取出第i列并counter，得到第i个特征的{取值1:样本数，取值2:样本数}
                    p_x_c[c][i] = {}  # 初始化该类别下该特征的概率字典
                    for xx, n in c_x_t[c][i].items():
                        # 拉普拉斯平滑
                        p_x_c[c][i][xx] = (n + 1) / (c_t[c] + len(attrs[i]))

        return p_x_c
    
    def calculate_continuous_params(self, div):# 计算连续特征的参数（均值和方差）
        continuous_params = {}  # {A类: {特征1: {'mean': 均值, 'var': 方差}, 特征2: ...}, B类: ...}
        
        for c, x in div.items():
            xn = np.array(x)
            continuous_params[c] = {}
            
            for i in range(len(x[0])):
                if self.feature_types[i] == 'continuous':  # 只处理连续特征
                    feature_values = xn[:, i]
                    # 计算均值和方差（加小值避免方差为0）
                    mean = np.mean(feature_values)
                    var = np.var(feature_values) + 1e-9  # 加一个小值避免方差为0
                    continuous_params[c][i] = {'mean': mean, 'var': var}
        
        return continuous_params
    
    def gaussian_probability(self, x, mean, var):# 计算高斯分布概率密度
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * var)))
        return (1 / (math.sqrt(2 * math.pi * var))) * exponent

    def fit(self, X_train, y_train):  # 后验概率=先验概率*条件概率/P(X),fit需要算出先验概率及P(某种特征的某个取值|某个类别)
        # 判断特征类型
        self.feature_types = self.identify_feature_types(X_train)
        
        c_t = Counter(y_train)  # 某类样本数

        div, pri = self.div_pri(c_t, X_train, y_train)  # 分割及先验概率
        
        attrs = self.get_attrs(X_train)  # 每种特征的取值数{特征索引：counter{取值1：数量，……}，……}

        # 计算离散特征的条件概率
        p_x_c = self.get_p_x_c(div, attrs, y_train)  
        
        # 计算连续特征的参数
        continuous_params = self.calculate_continuous_params(div)

        self.pri = pri
        self.p_x_c = p_x_c
        self.continuous_params = continuous_params
        return pri, p_x_c, continuous_params

    def predict(self, X_pre):  # 由于对同一组特征而言，分母一样，故只比较分子，即先验条件
        y_pre = []
        for x in X_pre:  # 如[0, 1, 0, 0]
            test_post = {}  # 存储后验概率{A类:后验概率，b类...}
            for c, pr in self.pri.items():  # {A类:先验概率}
                # 从先验概率开始
                posterior = math.log(pr)  # 使用对数避免数值下溢
                
                # 累加离散特征的条件概率（对数形式）
                for i in range(len(x)):
                    if self.feature_types[i] == 'discrete':
                        # 离散特征
                        if i in self.p_x_c[c] and x[i] in self.p_x_c[c][i]:
                            posterior += math.log(self.p_x_c[c][i][x[i]])
                        else:
                            # 如果特征值在训练时未见过，使用很小的概率
                            posterior += math.log(1e-6)
                    else:
                        # 连续特征
                        if i in self.continuous_params[c]:
                            mean = self.continuous_params[c][i]['mean']
                            var = self.continuous_params[c][i]['var']
                            prob = self.gaussian_probability(x[i], mean, var)
                            # 避免概率为0导致对数计算错误
                            prob = max(prob, 1e-10)
                            posterior += math.log(prob)
                
                # 转换回概率空间（为了便于比较，实际上不需要）
                test_post[c] = posterior
            
            # 找出概率最大的类别
            max_v = max(test_post.values())
            max_c = [c for c, p in test_post.items() if p == max_v]
            if len(max_c) > 1:
                print(f'{x}可能的类别有{len(max_c)}种，分别是{max_c}')
            else:
                y_pre.append(max_c[0])
                if y_pre[-1] == 1:
                    print('待测瓜是好瓜')
                else:
                    print('待测瓜不是好瓜')
        print(f"y_pre为{y_pre}")
        return y_pre


X_train = np.array([
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

y_train = np.array([1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) 

X_pre = np.array([
    [0, 0, 0, 0, 0, 0, 0.697, 0.460]  
])

classifier = NaiveBayesClassifier()
pri, p_x_c, continuous_params = classifier.fit(X_train, y_train)
y_pre = classifier.predict(X_pre)
