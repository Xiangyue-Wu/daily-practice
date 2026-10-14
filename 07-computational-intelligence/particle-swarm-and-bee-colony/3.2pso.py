import numpy as np
import matplotlib.pyplot as plt

# 问题定义
class KnapsackProblem:
    def __init__(self):
        self.num_items = 10
        self.capacity = 300
        self.volumes = np.array([95, 75, 23, 73, 50, 22, 6, 57, 89, 98])
        self.values = np.array([89, 59, 19, 43, 100, 72, 44, 16, 7, 64])
        self.penalty_alpha = 2.0  # 惩罚系数

    def calculate_volume(self, solution):
        """计算解的总体积"""
        return np.dot(solution, self.volumes)

    def calculate_value(self, solution):
        """计算解的总价值"""
        return np.dot(solution, self.values)

    def fitness(self, solution):
        """计算适应度（带惩罚）"""
        total_vol = self.calculate_volume(solution)
        total_val = self.calculate_value(solution)

        if total_vol <= self.capacity:
            return total_val
        else:
            # 超出容量，施加惩罚
            penalty = self.penalty_alpha * (total_vol - self.capacity)
            return total_val - penalty

# 粒子群优化 (PSO) 算法
class BPSO:
    def __init__(self, problem, pop_size=100, max_iter=200):
        self.problem = problem
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.dim = problem.num_items

        # PSO 参数
        self.c1 = 1.5  # 个体学习因子
        self.c2 = 1.5  # 社会学习因子
        self.w_max = 0.8  # 最大惯性权重
        self.w_min = 0.4  # 最小惯性权重
        self.v_max = 10.0  # 最大速度
        self.v_min = -10.0  # 最小速度

        # 初始化种群
        self.positions = np.random.randint(0, 2, size=(self.pop_size, self.dim))  # 二进制位置
        self.velocities = np.random.uniform(self.v_min, self.v_max, size=(self.pop_size, self.dim))  # 连续速度

        # 计算初始适应度
        self.fitnesses = np.array([self.problem.fitness(p) for p in self.positions])

        # 个体历史最优 (pbest)
        self.pbest_positions = self.positions.copy()
        self.pbest_fitnesses = self.fitnesses.copy()

        # 全局最优 (gbest)
        gbest_idx = np.argmax(self.fitnesses)
        self.gbest_position = self.positions[gbest_idx].copy()
        self.gbest_fitness = self.fitnesses[gbest_idx]

        # 记录进化历史
        self.history = []

    def sigmoid(self, x):
        """Sigmoid 函数，用于将速度映射到 [0, 1] 概率"""
        # 为防止溢出，对输入进行裁剪
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def optimize(self):
        """执行PSO优化主循环"""
        for iter in range(self.max_iter):
            # 动态更新惯性权重
            w = self.w_max - (self.w_max - self.w_min) * iter / self.max_iter

            # 更新速度
            r1 = np.random.rand(self.pop_size, self.dim)
            r2 = np.random.rand(self.pop_size, self.dim)
            cognitive = self.c1 * r1 * (self.pbest_positions - self.positions)
            social = self.c2 * r2 * (self.gbest_position - self.positions)
            self.velocities = w * self.velocities + cognitive + social

            # 限制速度范围
            self.velocities = np.clip(self.velocities, self.v_min, self.v_max)

            # 使用 Sigmoid 将速度转换为概率，并更新二进制位置
            prob = self.sigmoid(self.velocities)
            rand_matrix = np.random.rand(self.pop_size, self.dim)
            self.positions = (rand_matrix < prob).astype(int)

            # 评估新位置
            new_fitnesses = np.array([self.problem.fitness(p) for p in self.positions])

            # 更新个体历史最优 (pbest)
            improved = new_fitnesses > self.pbest_fitnesses
            self.pbest_positions[improved] = self.positions[improved]
            self.pbest_fitnesses[improved] = new_fitnesses[improved]

            # 更新全局最优 (gbest)
            current_best_idx = np.argmax(self.pbest_fitnesses)
            if self.pbest_fitnesses[current_best_idx] > self.gbest_fitness:
                self.gbest_fitness = self.pbest_fitnesses[current_best_idx]
                self.gbest_position = self.pbest_positions[current_best_idx].copy()

            # 记录当前全局最优适应度
            self.history.append(self.gbest_fitness)

            print(f"PSO Iteration {iter + 1}/{self.max_iter}, Best Fitness: {self.gbest_fitness:.2f}")

        return self.gbest_position, self.gbest_fitness

    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, 'b-', linewidth=2, label='PSO')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title('PSO Convergence Curve for 0-1 Knapsack Problem')
        plt.grid(True)
        plt.legend()
        plt.show()

# 主程序入口
if __name__ == "__main__":
    # 创建背包问题实例
    knapsack = KnapsackProblem()

    # 运行 PSO
    pso = BPSO(knapsack, pop_size=100, max_iter=200)
    best_solution_pso, best_fitness_pso = pso.optimize()

    # 输出结果
    print("\n" + "=" * 50)
    print("PSO Algorithm Result:")
    print("=" * 50)
    print(f"Best Solution (Binary): {best_solution_pso}")
    print(f"Selected Items (1-based): {np.where(best_solution_pso == 1)[0] + 1}")
    print(f"Total Volume: {knapsack.calculate_volume(best_solution_pso)}")
    print(f"Total Value: {knapsack.calculate_value(best_solution_pso)}")
    print(f"Fitness (with penalty): {best_fitness_pso:.2f}")
    print(f"Feasible? {'Yes' if knapsack.calculate_volume(best_solution_pso) <= knapsack.capacity else 'No'}")

    # 绘制收敛图
    pso.plot_convergence()