import numpy as np
import matplotlib.pyplot as plt
import time

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class ArtificialBeeColony:    
    def __init__(self):
        """初始化参数"""
        # 基本参数
        self.SN = 25           # 食物源数量（雇佣蜂=观察蜂=25）
        self.MCN = 200         # 最大循环次数
        self.limit = 50        # 放弃阈值
        self.D = 2             # 问题维度
        self.X_min, self.X_max = -4.0, 4.0   # 搜索边界
        
        # 记录数据
        self.convergence_curve = []   # 收敛曲线
        self.positions_history = []   # 位置历史
        
        print("="*60)
        print("人工蜂群算法（ABC）")
        print("="*60)
        print(f"蜂群规模: 雇佣蜂={self.SN}, 观察蜂={self.SN}")
        print(f"最大循环: {self.MCN}, 放弃阈值: {self.limit}")
        
    def objective_function(self, x, y):
        """目标函数：f(x,y) = 3*cos(x*y) + x + y^2"""
        return 3 * np.cos(x * y) + x + y**2
    
    def calculate_fitness(self, position):
        """计算适应度（目标函数值越小，适应度越大）"""
        f = self.objective_function(position[0], position[1])
        return 1 / (1 + f) if f >= 0 else 1 + abs(f)
    
    def initialize(self):
        """步骤1：初始化食物源"""
        # 随机初始化位置
        self.positions = np.random.uniform(self.X_min, self.X_max, (self.SN, self.D))
        # 计算适应度
        self.fitness = np.array([self.calculate_fitness(p) for p in self.positions])
        # 尝试次数计数器
        self.trials = np.zeros(self.SN)
        # 全局最优
        best_idx = np.argmax(self.fitness)
        self.gbest_pos = self.positions[best_idx].copy()
        self.gbest_val = self.objective_function(self.gbest_pos[0], self.gbest_pos[1])
        
        self.convergence_curve.append(self.gbest_val)
        self.positions_history.append(self.positions.copy())
        
        print(f"初始最优值: {self.gbest_val:.6f}")
    
    def generate_new_solution(self, i):
        """在当前解i的邻域生成新解"""
        k = np.random.randint(self.SN)
        while k == i:
            k = np.random.randint(self.SN)
        
        j = np.random.randint(self.D)  # 随机选一个维度
        phi = np.random.uniform(-1, 1)
        
        new_pos = self.positions[i].copy()
        new_pos[j] = self.positions[i, j] + phi * (self.positions[i, j] - self.positions[k, j])
        # 边界处理
        new_pos = np.clip(new_pos, self.X_min, self.X_max)
        
        return new_pos
    
    def employed_bee_phase(self):
        """雇佣蜂阶段：每个食物源附近搜索"""
        for i in range(self.SN):
            new_pos = self.generate_new_solution(i)
            new_fit = self.calculate_fitness(new_pos)
            
            if new_fit > self.fitness[i]:  # 贪婪选择
                self.positions[i] = new_pos
                self.fitness[i] = new_fit
                self.trials[i] = 0
            else:
                self.trials[i] += 1
    
    def onlooker_bee_phase(self):
        """观察蜂阶段：按概率选择优质食物源搜索"""
        prob = self.fitness / np.sum(self.fitness)
        
        for _ in range(self.SN):
            i = np.random.choice(self.SN, p=prob)  # 轮盘赌选择
            
            new_pos = self.generate_new_solution(i)
            new_fit = self.calculate_fitness(new_pos)
            
            if new_fit > self.fitness[i]:
                self.positions[i] = new_pos
                self.fitness[i] = new_fit
                self.trials[i] = 0
            else:
                self.trials[i] += 1
    
    def scout_bee_phase(self):
        """侦察蜂阶段：放弃长期未改进的食物源"""
        max_trial_idx = np.argmax(self.trials)
        if self.trials[max_trial_idx] >= self.limit:
            self.positions[max_trial_idx] = np.random.uniform(self.X_min, self.X_max, self.D)
            self.fitness[max_trial_idx] = self.calculate_fitness(self.positions[max_trial_idx])
            self.trials[max_trial_idx] = 0
    
    def update_global_best(self):
        """更新全局最优"""
        best_idx = np.argmax(self.fitness)
        current_best = self.objective_function(self.positions[best_idx, 0], self.positions[best_idx, 1])
        if current_best < self.gbest_val:
            self.gbest_val = current_best
            self.gbest_pos = self.positions[best_idx].copy()
    
    def optimize(self):
        """执行优化"""
        self.initialize()
        
        print("\n开始迭代...")
        start_time = time.time()
        
        for cycle in range(self.MCN):
            self.positions_history.append(self.positions.copy())
            
            self.employed_bee_phase()    # 阶段1
            self.onlooker_bee_phase()    # 阶段2
            self.scout_bee_phase()       # 阶段3
            self.update_global_best()    # 更新全局最优
            
            self.convergence_curve.append(self.gbest_val)
            
            if (cycle + 1) % 20 == 0:
                print(f"循环 {cycle+1:3d}: 最优值 = {self.gbest_val:.8f}")
        
        print(f"\n优化完成！用时: {time.time() - start_time:.4f}秒")
        return self.gbest_pos, self.gbest_val
    
    def plot_convergence(self):
        """绘制收敛曲线"""
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(self.convergence_curve)), self.convergence_curve, 'b-', linewidth=2)
        plt.xlabel('循环次数', fontsize=12)
        plt.ylabel('全局最优值', fontsize=12)
        plt.title('ABC收敛曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 标注最终值
        final_val = self.convergence_curve[-1]
        plt.axhline(y=final_val, color='r', linestyle='--', alpha=0.5)
        plt.text(len(self.convergence_curve)//2, final_val + 0.2, 
                 f'最终值: {final_val:.6f}', fontsize=11)
        plt.show()
    
    def plot_search_process(self):
        """绘制搜索过程（初始和最终对比）"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.linspace(self.X_min, self.X_max, 100)
        y = np.linspace(self.X_min, self.X_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.objective_function(X, Y)
        
        for idx, (ax, iter_num, title) in enumerate([
            (axes[0], 0, '初始食物源分布'),
            (axes[1], -1, f'最终食物源分布 (循环{self.MCN})')
        ]):
            ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
            ax.scatter(self.positions_history[iter_num][:, 0], 
                      self.positions_history[iter_num][:, 1],
                      c='white', s=30, edgecolors='black', linewidths=0.5)
            ax.scatter(self.gbest_pos[0], self.gbest_pos[1], 
                      c='red', s=120, marker='*', edgecolors='k')
            ax.set_xlabel('x', fontsize=12)
            ax.set_ylabel('y', fontsize=12)
            ax.set_title(title, fontsize=13)
            ax.set_xlim([self.X_min, self.X_max])
            ax.set_ylim([self.X_min, self.X_max])
        
        plt.tight_layout()
        plt.show()
    
    def print_results(self):
        """输出结果"""
        print("\n" + "="*60)
        print("ABC算法最终结果")
        print("="*60)
        print(f"最优解 x = {self.gbest_pos[0]:.8f}")
        print(f"最优解 y = {self.gbest_pos[1]:.8f}")
        print(f"最优值 f(x,y) = {self.gbest_val:.8f}")
        print("="*60)


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n目标函数：f(x,y) = 3*cos(x*y) + x + y^2")
    print("搜索范围：x ∈ [-4, 4], y ∈ [-4, 4]\n")
    
    # 运行ABC算法
    abc = ArtificialBeeColony()
    best_pos, best_val = abc.optimize()
    
    # 输出结果
    abc.print_results()
    
    # 绘制收敛曲线
    abc.plot_convergence()
    
    # 绘制搜索过程
    abc.plot_search_process()