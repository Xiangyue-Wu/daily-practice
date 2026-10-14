import numpy as np
import random
import matplotlib.pyplot as plt

# -----------------------------
# 1. 城市坐标（30个城市的TSP问题）
# -----------------------------
city = np.array([
    [41,94],[37,84],[54,67],[25,62],[7,64],[2,99],
    [68,58],[71,44],[54,62],[83,69],[64,63],[18,54],
    [22,60],[83,46],[91,38],[25,38],[24,42],[58,69],
    [71,71],[74,78],[87,76],[18,40],[13,40],[82,7],
    [62,32],[58,35],[45,21],[41,26],[44,35],[3,50]
])
city_num = len(city)

# -----------------------------
# 2. 核心参数
# -----------------------------
pop_size = 150       # 种群规模
generations = 500    # 迭代代数
pc = 0.9             # 交叉概率
pm = 0.01            # 变异概率

# -----------------------------
# 3. 计算路径长度（闭合路径）
# -----------------------------
def path_distance(path):
    dist = 0
    # 计算相邻城市的距离
    for i in range(city_num - 1):
        a = city[path[i]]
        b = city[path[i+1]]
        dist += np.linalg.norm(a - b)
    # 回到起点，形成闭合路径
    dist += np.linalg.norm(city[path[-1]] - city[path[0]])
    return dist

# -----------------------------
# 4. 初始化种群（无重复城市的随机路径）
# -----------------------------
def init_population():
    pop = []
    for _ in range(pop_size):
        p = list(range(city_num))
        random.shuffle(p)
        pop.append(p)
    return pop

# -----------------------------
# 5. 轮盘赌选择（基于适应度的概率选择）
# -----------------------------
def selection(pop, fitness, k=3):
    new_pop = []
    for _ in range(pop_size):
        # 随机选k个个体
        candidates = random.sample(list(zip(pop, fitness)), k)
        # 选适应度最高的
        winner = max(candidates, key=lambda x: x[1])
        new_pop.append(winner[0][:])
    return new_pop

# -----------------------------
# 6. 顺序交叉OX（TSP专用，保证路径无重复）
# -----------------------------
def order_crossover(p1, p2):
    # 随机选择交叉区间
    start = random.randint(0, city_num-2)
    end = random.randint(start+1, city_num-1)
    
    child = [-1]*city_num
    # 复制父代1的交叉区间
    child[start:end+1] = p1[start:end+1]
    # 填充父代2的剩余基因（无重复）
    ptr = 0
    for gene in p2:
        if gene not in child:
            while child[ptr] != -1:
                ptr += 1
            child[ptr] = gene
    return child

# -----------------------------
# 7. 交换变异（避免同一位置交换）
# -----------------------------
def mutation(path):
    i = random.randint(0, city_num-2)
    j = random.randint(i+1, city_num-1)
    path[i:j] = reversed(path[i:j])
    return path

# -----------------------------
# 8. 单轮GA求解函数（含精英保留）
# -----------------------------
def run_single_ga():
    population = init_population()
    best_distance = float("inf")
    best_path = None
    history = []  # 记录每代最优距离
    
    for g in range(generations):
        # 计算所有路径长度和适应度（1/距离，避免除0加极小值）
        distances = [path_distance(p) for p in population]
        fitness = [1/(d + 1e-8) for d in distances]
        
        # 更新全局最优解
        current_min_dist = min(distances)
        current_min_idx = distances.index(current_min_dist)
        current_best_path = population[current_min_idx].copy()
        
        if current_min_dist < best_distance:
            best_distance = current_min_dist
            best_path = current_best_path.copy()
        
        history.append(best_distance)
        
        # 步骤1：选择操作
        new_pop = selection(population, fitness)
        
        # 步骤2：交叉操作
        for i in range(0, pop_size-1, 2):
            if random.random() < pc:
                child1 = order_crossover(new_pop[i], new_pop[i+1])
                child2 = order_crossover(new_pop[i+1], new_pop[i])
                new_pop[i] = child1
                new_pop[i+1] = child2
        
        # 步骤3：变异操作
        for i in range(pop_size):
            if random.random() < pm:
                new_pop[i] = mutation(new_pop[i])
        
        # 步骤4：精英保留（将全局最优解保留到下一代）
        new_pop[0] = best_path.copy()
        
        # 更新种群
        population = new_pop
    
    return best_distance, best_path, history

# -----------------------------
# 9. 多次运行取最优（消除随机性影响）
# -----------------------------
def run_multiple_ga(run_times=10):
    print(f"开始运行{run_times}次遗传算法（pm={pm}）...\n")
    best_total_dist = float("inf")
    best_total_path = None
    best_total_history = None
    
    for i in range(run_times):
        dist, path, history = run_single_ga()
        print(f"第{i+1}次运行 - 最优路径长度：{dist:.2f}")
        
        # 更新全局最优（多次运行的最优）
        if dist < best_total_dist:
            best_total_dist = dist
            best_total_path = path
            best_total_history = history
    
    print(f"\n{run_times}次运行的最优结果：")
    print(f"最短路径长度：{best_total_dist:.2f}")
    print(f"最优路径：{best_total_path}")
    
    return best_total_dist, best_total_path, best_total_history

# -----------------------------
# 10. 可视化结果
# -----------------------------
def plot_results(best_dist, best_path, history):
    plt.figure(figsize=(12, 5))
    
    # 子图1：收敛曲线
    plt.subplot(1, 2, 1)
    plt.plot(history, color='blue', linewidth=1.5)
    plt.xlabel("Generation", fontsize=10)
    plt.ylabel("Best Distance)", fontsize=10)
    plt.title(f"GA TSP Convergence(pm={pm})", fontsize=12)
    plt.grid(alpha=0.3)
    
    # 子图2：最优路径图
    plt.subplot(1, 2, 2)
    # 闭合路径（回到起点）
    path = best_path + [best_path[0]]
    # 绘制路径
    for i in range(len(path)-1):
        a = city[path[i]]
        b = city[path[i+1]]
        plt.plot([a[0], b[0]], [a[1], b[1]], 'b-', alpha=0.6, linewidth=1)
    # 绘制城市节点
    plt.scatter(city[:,0], city[:,1], c='red', s=50, alpha=0.8)
    # 标注城市编号
    for i, (x, y) in enumerate(city):
        plt.text(x + 0.5, y + 0.5, str(i), fontsize=8)
    plt.title(f"Best Path{best_dist:.2f}", fontsize=12)
    plt.xlabel("X", fontsize=10)
    plt.ylabel("Y", fontsize=10)
    
    # 调整布局，避免重叠
    plt.tight_layout()
    plt.show()

# -----------------------------
# 主程序入口
# -----------------------------
if __name__ == "__main__":
    # 运行多次GA并取最优
    best_dist, best_path, best_history = run_multiple_ga(run_times=10)
    # 可视化结果
    plot_results(best_dist, best_path, best_history)