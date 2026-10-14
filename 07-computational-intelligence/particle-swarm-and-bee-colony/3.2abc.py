import numpy as np
import matplotlib.pyplot as plt

# ===========================
# 问题定义 (与PSO共享)
# ===========================
# 假设 KnapsackProblem 类已在上面定义，此处不再重复。
# 如果单独运行ABC，请取消注释下面的类定义。


class KnapsackProblem:
    def __init__(self):
        self.num_items = 10
        self.capacity = 300
        self.volumes = np.array([95, 75, 23, 73, 50, 22, 6, 57, 89, 98])
        self.values = np.array([89, 59, 19, 43, 100, 72, 44, 16, 7, 64])
        self.penalty_alpha = 2.0

    def calculate_volume(self, solution):
        return np.dot(solution, self.volumes)

    def calculate_value(self, solution):
        return np.dot(solution, self.values)

    def fitness(self, solution):
        total_vol = self.calculate_volume(solution)
        total_val = self.calculate_value(solution)
        if total_vol <= self.capacity:
            return total_val
        else:
            penalty = self.penalty_alpha * (total_vol - self.capacity)
            return total_val - penalty


# ===========================
# 人工蜂群 (ABC) 算法
# ===========================
class DiscreteABC:
    def __init__(self, problem, colony_size=100, max_iter=200, limit_factor=0.5):
        """
        colony_size: 蜂群规模 (等于雇佣蜂数量 SN)
        limit_factor: 侦察蜂触发阈值的比例因子 (limit = limit_factor * D)
        """
        self.problem = problem
        self.colony_size = colony_size  # 雇佣蜂数 = 观察蜂数 = SN
        self.max_iter = max_iter
        self.dim = problem.num_items
        self.limit = int(limit_factor * self.dim)  # 侦察蜂阈值

        # 初始化食物源 (解)
        self.food_sources = np.random.randint(0, 2, size=(self.colony_size, self.dim))
        self.fitnesses = np.array([self.problem.fitness(fs) for fs in self.food_sources])
        self.trial_counts = np.zeros(self.colony_size)  # 记录每个解未改进的次数

        # 全局最优
        best_idx = np.argmax(self.fitnesses)
        self.global_best_solution = self.food_sources[best_idx].copy()
        self.global_best_fitness = self.fitnesses[best_idx]

        # 记录历史
        self.history = []

    def generate_neighbor(self, current_solution, phi=None):
        """
        生成邻域解。对于离散问题，采用随机翻转一个维度的策略。
        phi 参数在此策略中未使用，保留以符合ABC框架。
        """
        neighbor = current_solution.copy()
        # 随机选择一个维度进行翻转
        j = np.random.randint(self.dim)
        neighbor[j] = 1 - neighbor[j]  # 0->1, 1->0
        return neighbor

    def optimize(self):
        """执行ABC优化主循环"""
        for iter in range(self.max_iter):
            # --- 雇佣蜂阶段 ---
            for i in range(self.colony_size):
                # 为第i个食物源生成一个邻域解
                candidate = self.generate_neighbor(self.food_sources[i])
                candidate_fitness = self.problem.fitness(candidate)

                # 贪婪选择
                if candidate_fitness > self.fitnesses[i]:
                    self.food_sources[i] = candidate
                    self.fitnesses[i] = candidate_fitness
                    self.trial_counts[i] = 0  # 重置尝试次数
                else:
                    self.trial_counts[i] += 1

            # --- 计算选择概率 (用于观察蜂) ---
            # 为了避免负适应度导致概率计算错误，进行平移
            # 标准做法是使用 fit_i = 1 / (1 + f_i) 如果 f_i >= 0, 否则 fit_i = 1 + abs(f_i)
            # 但为了简化并与PSO对比，我们直接使用适应度值，并确保其为正
            adjusted_fitness = self.fitnesses - np.min(self.fitnesses) + 1e-8
            probabilities = adjusted_fitness / np.sum(adjusted_fitness)

            # --- 观察蜂阶段 ---
            # 轮盘赌选择 colony_size 次
            for _ in range(self.colony_size):
                # 轮盘赌选择一个食物源
                selected_index = np.random.choice(self.colony_size, p=probabilities)

                # 为选中的食物源生成邻域解
                candidate = self.generate_neighbor(self.food_sources[selected_index])
                candidate_fitness = self.problem.fitness(candidate)

                # 贪婪选择
                if candidate_fitness > self.fitnesses[selected_index]:
                    self.food_sources[selected_index] = candidate
                    self.fitnesses[selected_index] = candidate_fitness
                    self.trial_counts[selected_index] = 0

            # --- 侦察蜂阶段 ---
            # 检查是否有食物源需要被放弃
            for i in range(self.colony_size):
                if self.trial_counts[i] > self.limit:
                    # 放弃该食物源，随机生成一个新的
                    self.food_sources[i] = np.random.randint(0, 2, size=self.dim)
                    self.fitnesses[i] = self.problem.fitness(self.food_sources[i])
                    self.trial_counts[i] = 0

            # --- 更新全局最优 ---
            current_best_idx = np.argmax(self.fitnesses)
            if self.fitnesses[current_best_idx] > self.global_best_fitness:
                self.global_best_fitness = self.fitnesses[current_best_idx]
                self.global_best_solution = self.food_sources[current_best_idx].copy()

            # 记录历史
            self.history.append(self.global_best_fitness)

            print(f"ABC Iteration {iter + 1}/{self.max_iter}, Best Fitness: {self.global_best_fitness:.2f}")

        return self.global_best_solution, self.global_best_fitness

    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history, 'r--', linewidth=2, label='ABC')
        plt.xlabel('Iteration')
        plt.ylabel('Best Fitness')
        plt.title('ABC Convergence Curve for 0-1 Knapsack Problem')
        plt.grid(True)
        plt.legend()
        plt.show()


# ===========================
# 主程序入口
# ===========================
if __name__ == "__main__":
    # 创建背包问题实例
    knapsack = KnapsackProblem()

    # 运行 ABC
    abc = DiscreteABC(knapsack, colony_size=100, max_iter=200)
    best_solution_abc, best_fitness_abc = abc.optimize()

    # 输出结果
    print("\n" + "=" * 50)
    print("ABC Algorithm Result:")
    print("=" * 50)
    print(f"Best Solution (Binary): {best_solution_abc}")
    print(f"Selected Items (1-based): {np.where(best_solution_abc == 1)[0] + 1}")
    print(f"Total Volume: {knapsack.calculate_volume(best_solution_abc)}")
    print(f"Total Value: {knapsack.calculate_value(best_solution_abc)}")
    print(f"Fitness (with penalty): {best_fitness_abc:.2f}")
    print(f"Feasible? {'Yes' if knapsack.calculate_volume(best_solution_abc) <= knapsack.capacity else 'No'}")

    # 绘制收敛图
    abc.plot_convergence()