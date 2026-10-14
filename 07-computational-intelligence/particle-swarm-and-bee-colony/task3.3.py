import numpy as np
import matplotlib.pyplot as plt
import random
import copy

# 1. 数据准备与初始化
cities = np.array([
    [41, 94], [37, 84], [54, 67], [25, 62], [7, 64], [2, 99],
    [68, 58], [71, 44], [54, 62], [83, 69], [64, 63], [18, 54],
    [22, 60], [83, 46], [91, 38], [25, 38], [24, 42], [58, 69],
    [71, 71], [74, 78], [87, 76], [18, 40], [13, 40], [82, 7],
    [62, 32], [58, 35], [45, 21], [41, 26], [44, 35], [3, 50]
])
n_cities = len(cities)

# 计算距离矩阵
dist_matrix = np.zeros((n_cities, n_cities))
for i in range(n_cities):
    for j in range(n_cities):
        dist_matrix[i, j] = np.linalg.norm(cities[i] - cities[j])

# 2. 粒子群算法 (PSO)
class Particle:
    def __init__(self, n):
        # 随机初始化位置 (排列)
        self.position = list(np.random.permutation(n))
        # 初始化速度 (空列表，表示交换操作)
        self.velocity = []
        # 个体最优
        self.best_position = self.position[:]
        self.best_cost = float('inf')

def calc_cost(path):
    cost = 0
    for i in range(len(path)):
        cost += dist_matrix[path[i], path[(i + 1) % len(path)]]
    return cost

def apply_velocity(pos, vel):
    # 在位置 pos 上执行交换序列 vel
    new_pos = pos[:]
    for swap in vel:
        i, j = swap
        new_pos[i], new_pos[j] = new_pos[j], new_pos[i]
    return new_pos

def get_swap_sequence(p1, p2):
    # 计算从 p1 变到 p2 需要的交换操作
    swaps = []
    current = p1[:]
    pos_map = {v: i for i, v in enumerate(current)}  # 值到索引的映射

    for i in range(len(p1)):
        if current[i] != p2[i]:
            # 找到 p2[i] 在 current 中的位置
            target_val = p2[i]
            j = pos_map[target_val]

            # 交换 current[i] 和 current[j]
            swaps.append((i, j))

            # 更新 current 和 map
            val_i = current[i]
            current[i], current[j] = current[j], current[i]

            pos_map[val_i] = j
            pos_map[target_val] = i

    return swaps

def two_opt_swap(path):
    # 2-opt 局部搜索，用于提升解的质量
    best = path[:]
    improved = True
    while improved:
        improved = False
        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                if j - i == 1: continue  # 相邻点交换无意义
                new_path = best[:]
                new_path[i:j] = reversed(best[i:j])

                if calc_cost(new_path) < calc_cost(best):
                    best = new_path
                    improved = True
        if improved:
            break  # 找到一个改进就跳出，重新开始（First Improvement）
    return best

n_particles = 50
n_iterations = 200
w = 0.6  # 惯性权重
c1 = 1.5  # 个体学习因子
c2 = 1.5  # 全局学习因子

# 初始化种群
swarm = [Particle(n_cities) for _ in range(n_particles)]
global_best_position = None
global_best_cost = float('inf')
history_pso = []

# 计算初始适应度
for p in swarm:
    cost = calc_cost(p.position)
    p.best_cost = cost
    if cost < global_best_cost:
        global_best_cost = cost
        global_best_position = p.position[:]

# --- 迭代优化 ---
for it in range(n_iterations):
    for p in swarm:
        # 1. 计算当前成本
        current_cost = calc_cost(p.position)

        # 2. 更新个体最优
        if current_cost < p.best_cost:
            p.best_cost = current_cost
            p.best_position = p.position[:]

        # 3. 更新全局最优
        if current_cost < global_best_cost:
            global_best_cost = current_cost
            global_best_position = p.position[:]

    # 记录历史
    history_pso.append(global_best_cost)

    # 4. 更新速度和位置
    for p in swarm:
        # --- 速度更新逻辑 (基于概率的交换序列组合) ---
        new_velocity = []

        # 惯性部分: w * V
        for swap in p.velocity:
            if random.random() < w:
                new_velocity.append(swap)

        # 认知部分: c1 * (Pbest - X)
        seq_pbest = get_swap_sequence(p.position, p.best_position)
        for swap in seq_pbest:
            if random.random() < c1:
                new_velocity.append(swap)

        # 社会部分: c2 * (Gbest - X)
        seq_gbest = get_swap_sequence(p.position, global_best_position)
        for swap in seq_gbest:
            if random.random() < c2:
                new_velocity.append(swap)

        p.velocity = new_velocity

        # --- 位置更新 ---
        p.position = apply_velocity(p.position, p.velocity)

    # --- 局部搜索增强 ---
    if it % 10 == 0:
        temp_best = two_opt_swap(global_best_position)
        temp_cost = calc_cost(temp_best)
        if temp_cost < global_best_cost:
            global_best_cost = temp_cost
            global_best_position = temp_best
            # 同步给所有粒子，引导搜索
            for particle in swarm:
                particle.best_position = temp_best[:]
                # 可选：重置粒子位置到全局最优附近

print(f"\n【粒子群算法 (PSO + 2-opt)】")
print(f"最优路径长度: {global_best_cost:.4f}")
print(f"最优路径: {global_best_position}")

# 3. 结果可视化
plt.figure(figsize=(12, 5))

# 绘制收敛曲线
plt.subplot(1, 2, 1)
plt.plot(history_pso, color='green', label='PSO Best')
plt.title(f'PSO Convergence (Best: {global_best_cost:.2f})')
plt.xlabel('Iteration')
plt.ylabel('Path Length')
plt.grid(True)
plt.legend()


# 绘制最佳路径图
def plot_path(path, title):
    plt.figure(figsize=(8, 6))
    # 提取坐标
    x = [cities[i][0] for i in path] + [cities[path[0]][0]]
    y = [cities[i][1] for i in path] + [cities[path[0]][1]]

    plt.plot(x, y, 'o-', linewidth=1.5, markersize=6, color='green')
    plt.scatter(x[0], y[0], c='red', s=100, zorder=5, label='Start/End')

    for i, txt in enumerate(path):
        plt.annotate(txt, (cities[txt][0], cities[txt][1]),
                     textcoords="offset points", xytext=(5, 5), fontsize=9)

    plt.title(f"{title}\nLength: {global_best_cost:.4f}")
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.show()


plot_path(global_best_position, "PSO Optimal Path")