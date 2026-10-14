import numpy as np
import matplotlib.pyplot as plt

# ======================
# 1. 数据准备
# ======================
cities = np.array([
    [41, 94], [37, 84], [54, 67], [25, 62], [7, 64], [2, 99],
    [68, 58], [71, 44], [54, 62], [83, 69], [64, 63], [18, 54],
    [22, 60], [83, 46], [91, 38], [25, 38], [24, 42], [58, 69],
    [71, 71], [74, 78], [87, 76], [18, 40], [13, 40], [82, 7],
    [62, 32], [58, 35], [45, 21], [41, 26], [44, 35], [3, 50]
])
n_cities = len(cities)

# 距离矩阵
dist_matrix = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        dist_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])


# 辅助函数：计算路径长度
def calc_length(path):
    return sum(dist_matrix[path[i], path[i + 1]] for i in range(len(path) - 1))

# 2. 蚁群算法 —— 经典蚁密模型 (Ant-Density)
print("【正在运行蚁密模型 (Ant-Density ACO)...】")
n_ants = 30
n_iterations = 200
alpha = 1.0
beta = 2.0
rho = 0.5
Q = 1.0  # 固定增量

# 初始化
tau = np.full((n_cities, n_cities), 0.1)  # 初始信息素
eta = 1.0 / (dist_matrix + np.eye(n_cities) * 1e-10)

best_length_aco = float('inf')
best_path_aco = None
history_aco = []

for it in range(n_iterations):
    # 全局挥发
    tau *= (1 - rho)

    current_best_len = float('inf')

    for k in range(n_ants):
        # 初始化蚂蚁路径
        start_city = np.random.randint(0, n_cities)
        path = [start_city]
        visited = np.zeros(n_cities, dtype=bool)
        visited[start_city] = True
        current = start_city

        # 构建路径（n_cities - 1 步）
        for step in range(n_cities - 1):
            unvisited = np.where(~visited)[0]
            # 计算转移概率
            probs = (tau[current, unvisited] ** alpha) * (eta[current, unvisited] ** beta)
            probs /= np.sum(probs)
            next_city = np.random.choice(unvisited, p=probs)

            # 【蚁密模型核心】：每走一步，立即增加固定信息素
            tau[current, next_city] += Q
            tau[next_city, current] += Q  # 对称TSP

            path.append(next_city)
            visited[next_city] = True
            current = next_city

        path.append(path[0])  # 闭合回路
        length = calc_length(path)

        if length < current_best_len:
            current_best_len = length
        if length < best_length_aco:
            best_length_aco = length
            best_path_aco = path[:]

    history_aco.append(best_length_aco)

# 打印结果
best_path_aco_clean = [int(x) for x in best_path_aco]
print(f"【蚁群算法 (蚁密模型)】")
print(f"最优路径长度: {best_length_aco:.4f}")
print(f"最优路径: {best_path_aco_clean}")

# 3. 免疫算法 + 疫苗接种 (2-opt 作为疫苗)
print("\n【正在运行免疫算法 (IPA + 2-opt 疫苗)...】")

pop_size = 100
max_gen = 200
pc = 0.8
pm = 0.2
pv = 0.5  # 疫苗接种概率（较高，因2-opt有效）


# 2-opt 局部搜索（作为疫苗）
def two_opt(route, dist):
    best_route = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route) - 1):
            for j in range(i + 1, len(route)):
                if j - i == 1:
                    continue
                new_route = route[:i] + route[i:j][::-1] + route[j:]
                if calc_length(new_route + [new_route[0]]) < calc_length(best_route + [best_route[0]]):
                    best_route = new_route
                    improved = True
        route = best_route
    return best_route

# 初始化种群
def create_population(size):
    pop = []
    for _ in range(size):
        if np.random.rand() < 0.3:
            # 贪心+随机
            path = [np.random.randint(0, n_cities)]
            visited = set(path)
            while len(visited) < n_cities:
                cur = path[-1]
                candidates = [(i, dist_matrix[cur, i]) for i in range(n_cities) if i not in visited]
                candidates.sort(key=lambda x: x[1])
                idx = np.random.randint(0, min(3, len(candidates)))
                path.append(candidates[idx][0])
                visited.add(candidates[idx][0])
            path.append(path[0])
            pop.append(path)
        else:
            p = list(np.random.permutation(n_cities))
            p.append(p[0])
            pop.append(p)
    return pop

# 交叉 (OX)
def crossover(p1, p2):
    if np.random.rand() > pc:
        return p1[:], p2[:]
    size = n_cities
    start, end = sorted(np.random.choice(size, 2, replace=False))
    child1 = [-1] * (size + 1)
    child2 = [-1] * (size + 1)
    child1[start:end] = p1[start:end]
    child2[start:end] = p2[start:end]

    def fill_child(child, parent):
        ptr = 0
        for i in range(size):
            if child[i] == -1:
                while ptr < size and parent[ptr] in child:
                    ptr += 1
                if ptr < size:
                    child[i] = parent[ptr]
                else:
                    # 安全兜底：填充剩余城市
                    remaining = [c for c in parent if c not in child]
                    for j in range(size):
                        if child[j] == -1:
                            child[j] = remaining.pop(0)
                    break
                ptr += 1
        child[-1] = child[0]
        return child

    return fill_child(child1, p2), fill_child(child2, p1)

# 变异 (Swap)
def mutate(path):
    if np.random.rand() > pm:
        return path[:]
    new_path = path[:]
    i, j = np.random.choice(range(n_cities), 2, replace=False)
    new_path[i], new_path[j] = new_path[j], new_path[i]
    new_path[-1] = new_path[0]
    return new_path

# 疫苗接种：使用 2-opt
def vaccinate(pop):
    new_pop = []
    for path in pop:
        if np.random.rand() < pv:
            # 提取无重复路径（去掉最后的起点）
            core_path = path[:-1]
            optimized = two_opt(core_path, dist_matrix)
            optimized.append(optimized[0])
            new_pop.append(optimized)
        else:
            new_pop.append(path[:])
    return new_pop

# 免疫选择
def immune_selection(old_pop, new_pop):
    selected = []
    for old_p, new_p in zip(old_pop, new_pop):
        if calc_length(new_p) <= calc_length(old_p):
            selected.append(new_p)
        else:
            # 小概率保留变差个体（维持多样性）
            if np.random.rand() < 0.05:
                selected.append(new_p)
            else:
                selected.append(old_p)
    return selected

# 主循环
population = create_population(pop_size)
best_length_ia = float('inf')
best_path_ia = None
history_ia = []

for gen in range(max_gen):
    lengths = [calc_length(p) for p in population]
    min_len = min(lengths)
    if min_len < best_length_ia:
        best_length_ia = min_len
        best_path_ia = population[lengths.index(min_len)]
    history_ia.append(best_length_ia)

    # 交叉
    next_gen = []
    for i in range(0, pop_size, 2):
        p1 = population[i]
        p2 = population[(i + 1) % pop_size]
        c1, c2 = crossover(p1, p2)
        next_gen.extend([c1, c2])

    # 变异
    next_gen = [mutate(p) for p in next_gen]

    # 疫苗接种（2-opt）
    vaccinated = vaccinate(next_gen)

    # 免疫选择
    population = immune_selection(next_gen, vaccinated)

best_path_ia_clean = [int(x) for x in best_path_ia]
print(f"\n【免疫算法 (含疫苗接种)】")
print(f"最优路径长度: {best_length_ia:.4f}")
print(f"最优路径: {best_path_ia_clean}")

# 4. 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history_aco, label='ACO (Ant-Density)')
plt.title(f'ACO Convergence\nBest: {best_length_aco:.2f}')
plt.xlabel('Iteration')
plt.ylabel('Path Length')
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_ia, color='orange', label='IA + 2-opt Vaccine')
plt.title(f'IA Convergence\nBest: {best_length_ia:.2f}')
plt.xlabel('Generation')
plt.ylabel('Path Length')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# 绘制路径
def plot_path(path, title):
    plt.figure(figsize=(8, 6))
    coords = cities[path[:-1]]  # 去掉重复起点
    x, y = coords[:, 0], coords[:, 1]
    plt.plot(x, y, 'o-', linewidth=1.5, markersize=6)
    plt.plot(x[0], y[0], 'rs', markersize=10, label='Start/End')
    # 标注城市编号（真实ID）
    for i in range(len(coords)):
        city_id = path[i]
        plt.annotate(str(city_id), (x[i], y[i]), textcoords="offset points", xytext=(5, 5), fontsize=8)
    plt.title(f"{title}\nLength: {calc_length(path):.4f}")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    plt.show()

plot_path(best_path_aco, "ACO (Ant-Density) Optimal Path")
plot_path(best_path_ia, "IA + 2-opt Vaccine Optimal Path")