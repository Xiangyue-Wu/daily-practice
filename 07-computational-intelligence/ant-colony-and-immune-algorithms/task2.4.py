import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler

# ===============================
# 1. 数据加载与预处理
# ===============================
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 归一化
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# 定义：
# Setosa(0) → 正常
# 其他 → 异常
normal_data = X[y == 0]
abnormal_data = X[y != 0]

# ===============================
# 2. 参数设置
# ===============================
N_ANTIBODIES = 50
MAX_ITER = 30
CLONE_FACTOR = 5
MUTATION_RATE = 0.1
SUPPRESSION_THRESHOLD = 0.1

# ===============================
# 3. 亲和度函数（欧氏距离）
# ===============================
def affinity(ab, x):
    center = ab[:-1]
    return np.linalg.norm(center - x)

# ===============================
# 4. 判断是否匹配（异常检测）
# ===============================
def is_match(ab, x):
    return affinity(ab, x) < ab[-1]

# ===============================
# 5. 初始化抗体（负选择）
# ===============================
def initialize_antibodies():
    antibodies = []

    while len(antibodies) < N_ANTIBODIES:
        center = np.random.rand(4)
        theta = random.uniform(0.1, 0.5)

        ab = np.append(center, theta)

        # 不能匹配正常数据（Self）
        match_self = False
        for x in normal_data:
            if is_match(ab, x):
                match_self = True
                break

        if not match_self:
            antibodies.append(ab)

    return np.array(antibodies)

# ===============================
# 6. 适应度函数
# ===============================
def fitness(ab):
    # 检测异常能力（DR）
    detected = 0
    for x in abnormal_data:
        if is_match(ab, x):
            detected += 1
    DR = detected / len(abnormal_data)

    # 误报率（FPR）
    false_alarm = 0
    for x in normal_data:
        if is_match(ab, x):
            false_alarm += 1
    FPR = false_alarm / len(normal_data)

    return DR - FPR

# ===============================
# 7. 克隆与变异
# ===============================
def clone_and_mutate(antibodies):
    new_population = []

    for ab in antibodies:
        fit = fitness(ab)
        num_clones = int(CLONE_FACTOR * (fit + 1))  # 防止负数

        for _ in range(num_clones):
            clone = ab.copy()

            # 变异
            for i in range(len(clone) - 1):
                if random.random() < MUTATION_RATE:
                    clone[i] += np.random.normal(0, 0.1)

            # 限制范围
            clone[:-1] = np.clip(clone[:-1], 0, 1)
            clone[-1] = np.clip(clone[-1], 0.05, 1)

            new_population.append(clone)

    return np.array(new_population)

# ===============================
# 8. 免疫抑制（去重）
# ===============================
def suppression(antibodies):
    filtered = []

    for ab in antibodies:
        keep = True
        for other in filtered:
            if np.linalg.norm(ab[:-1] - other[:-1]) < SUPPRESSION_THRESHOLD:
                keep = False
                break
        if keep:
            filtered.append(ab)

    return np.array(filtered)

# ===============================
# 9. 训练过程
# ===============================
antibodies = initialize_antibodies()

# ⭐ 新增：记录收敛曲线
fitness_history = []

for iteration in range(MAX_ITER):
    # 克隆 + 变异
    new_pop = clone_and_mutate(antibodies)

    # 合并
    combined = np.vstack((antibodies, new_pop))

    # 计算适应度并排序
    fitness_values = np.array([fitness(ab) for ab in combined])
    sorted_idx = np.argsort(-fitness_values)

    combined = combined[sorted_idx]

    # 免疫抑制
    combined = suppression(combined)

    # 保留前N个
    antibodies = combined[:N_ANTIBODIES]

    # ⭐ 记录最优适应度
    best_fit = fitness_values[sorted_idx[0]]
    fitness_history.append(best_fit)

    print(f"Iteration {iteration+1}, Best Fitness: {best_fit:.4f}")

# ===============================
# 10. 测试（异常检测）
# ===============================
def detect(x):
    for ab in antibodies:
        if is_match(ab, x):
            return 1  # 异常
    return 0  # 正常

y_true = np.array([0]*len(normal_data) + [1]*len(abnormal_data))
X_test = np.vstack((normal_data, abnormal_data))

y_pred = np.array([detect(x) for x in X_test])

# ===============================
# 11. 评估
# ===============================
accuracy = np.mean(y_pred == y_true)
print("Accuracy:", accuracy)

# ===============================
# 12. 可视化（分类结果）
# ===============================
plt.figure()

for i, x in enumerate(X_test):
    if y_pred[i] == 1:
        plt.scatter(x[0], x[1], marker='x')  # 异常
    else:
        plt.scatter(x[0], x[1], marker='o')  # 正常

plt.title("AIS Anomaly Detection (Iris)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")


# ===============================
# 13. 收敛曲线
# ===============================
plt.figure()
plt.plot(fitness_history)
plt.title("Fitness Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Fitness")
plt.grid()
plt.show()