"""
粒子群优化算法（PSO）求解函数极小值问题
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ParticleSwarmOptimization:
    def __init__(self):
        """步骤1：初始化算法参数"""
        # 基本参数设置（严格按照图片要求）
        self.N = 100          # 初始种群粒子数
        self.D = 2            # 粒子维数（x和y两个维度）
        self.T = 200          # 最大迭代次数
        
        # 学习因子
        self.C1 = 1.5         # 个体学习因子
        self.C2 = 1.5         # 社会学习因子
        
        # 惯性权重
        self.W_max = 0.8      # 惯性权重最大值
        self.W_min = 0.4      # 惯性权重最小值
        
        # 位置边界
        self.X_max = 4.0      # 位置最大值
        self.X_min = -4.0     # 位置最小值
        
        # 速度边界
        self.V_max = 1.0      # 速度最大值
        self.V_min = -1.0     # 速度最小值
        
        # 用于存储的数据结构
        self.convergence_curve = []      # 收敛曲线（每代的最优值）
        self.gbest_history = []          # 全局最优位置历史
        self.positions_history = []      # 粒子位置历史
        
        print("="*60)
        print("步骤1：初始化算法参数")
        print("="*60)
        print(f"粒子数 N = {self.N}")
        print(f"粒子维数 D = {self.D}")
        print(f"最大迭代次数 T = {self.T}")
        print(f"学习因子 C1 = C2 = {self.C1}")
        print(f"惯性权重范围: W_min = {self.W_min}, W_max = {self.W_max}")
        print(f"位置范围: X ∈ [{self.X_min}, {self.X_max}]")
        print(f"速度范围: V ∈ [{self.V_min}, {self.V_max}]")
        
    def objective_function(self, x, y):
        """
        目标函数：f(x,y) = 3*cos(x*y) + x + y^2
        
        参数：
        - x, y: 决策变量
        
        返回：
        - 函数值
        """
        return 3 * np.cos(x * y) + x + y**2
    
    def calculate_dynamic_inertia_weight(self, t):
        w = self.W_max - (self.W_max - self.W_min) * (t / self.T)
        return w
    
    def initialize_population(self):
        """
        步骤2：初始化种群粒子的位置和速度
               初始化粒子个体的最优位置和最优值
               初始化粒子群全局的最优位置和最优值
        """
        print("\n" + "="*60)
        print("步骤2：初始化种群")
        print("="*60)
        
        # 初始化粒子的位置 x（在[X_min, X_max]范围内随机生成）
        self.x = np.random.uniform(self.X_min, self.X_max, (self.N, self.D))
        
        # 初始化粒子的速度 v（在[V_min, V_max]范围内随机生成）
        self.v = np.random.uniform(self.V_min, self.V_max, (self.N, self.D))
        
        # 初始化粒子个体的最优位置 p（初始时等于当前位置）
        self.p = self.x.copy()
        
        # 初始化粒子个体的最优值 pbest
        self.pbest = np.array([self.objective_function(self.x[i, 0], self.x[i, 1]) 
                               for i in range(self.N)])
        
        # 初始化粒子群全局的最优位置 g
        best_idx = np.argmin(self.pbest)
        self.g = self.p[best_idx].copy()
        
        # 初始化粒子群全局的最优值 gbest
        self.gbest = self.pbest[best_idx]
        
        print(f"初始全局最优值 gbest = {self.gbest:.6f}")
        print(f"初始全局最优位置 g = ({self.g[0]:.6f}, {self.g[1]:.6f})")
        
        # 记录初始状态
        self.convergence_curve.append(self.gbest)
        self.gbest_history.append(self.g.copy())
        self.positions_history.append(self.x.copy())
    
    def boundary_check(self, x, v):
        """
        边界条件处理
        
        参数：
        - x: 位置向量
        - v: 速度向量
        
        返回：
        - x_new: 处理后的位置向量
        - v_new: 处理后的速度向量
        """
        x_new = x.copy()
        v_new = v.copy()
        
        for d in range(self.D):
            # 位置边界处理（吸收边界）
            if x_new[d] > self.X_max:
                x_new[d] = self.X_max
                v_new[d] = 0  # 碰到边界时速度置零
            elif x_new[d] < self.X_min:
                x_new[d] = self.X_min
                v_new[d] = 0  # 碰到边界时速度置零
            
            # 速度边界处理
            if v_new[d] > self.V_max:
                v_new[d] = self.V_max
            elif v_new[d] < self.V_min:
                v_new[d] = self.V_min
        
        return x_new, v_new
    
    def update_velocity_and_position(self, w):
        """
        根据式（6.9）更新位置 x 和速度 v
        
        速度更新公式：
        v_i = w * v_i + C1 * r1 * (p_i - x_i) + C2 * r2 * (g - x_i)
        
        位置更新公式：
        x_i = x_i + v_i
        
        参数：
        - w: 当前惯性权重
        """
        for i in range(self.N):
            # 生成[0,1]范围内的随机数 r1, r2
            r1 = np.random.rand(self.D)
            r2 = np.random.rand(self.D)
            
            # 速度更新（式6.9）
            cognitive_component = self.C1 * r1 * (self.p[i] - self.x[i])
            social_component = self.C2 * r2 * (self.g - self.x[i])
            
            self.v[i] = w * self.v[i] + cognitive_component + social_component
            
            # 位置更新
            self.x[i] = self.x[i] + self.v[i]
            
            # 边界条件处理
            self.x[i], self.v[i] = self.boundary_check(self.x[i], self.v[i])
    
    def update_individual_best(self):
        """
        判断是否替换粒子个体最优位置 p 和最优值 pbest
        """
        for i in range(self.N):
            # 计算当前位置的目标函数值
            current_value = self.objective_function(self.x[i, 0], self.x[i, 1])
            
            # 如果当前值优于个体历史最优值，则更新
            if current_value < self.pbest[i]:
                self.pbest[i] = current_value
                self.p[i] = self.x[i].copy()
    
    def update_global_best(self):
        """
        判断是否替换粒子群全局最优位置 g 和最优值 gbest
        """
        # 找出当前种群中的最优个体
        current_best_idx = np.argmin(self.pbest)
        current_best_value = self.pbest[current_best_idx]
        
        # 如果当前最优值优于全局历史最优值，则更新
        if current_best_value < self.gbest:
            self.gbest = current_best_value
            self.g = self.p[current_best_idx].copy()
            return True
        return False
    
    def optimize(self, verbose=True):
        """
        执行PSO优化过程
        
        步骤3：迭代更新
        步骤4：判断终止条件
        """
        # 执行步骤2：初始化
        self.initialize_population()
        
        print("\n" + "="*60)
        print("步骤3：开始迭代优化")
        print("="*60)
        
        start_time = time.time()
        
        # 迭代优化
        for t in range(self.T):
            # 记录当前位置
            self.positions_history.append(self.x.copy())
            
            # 根据式（6.14）计算动态惯性权值 w
            w = self.calculate_dynamic_inertia_weight(t)
            
            # 根据式（6.9）更新位置 x 和速度 v，并进行边界条件处理
            self.update_velocity_and_position(w)
            
            # 判断是否替换粒子个体最优位置 p 和最优值 pbest
            self.update_individual_best()
            
            # 判断是否替换粒子群全局最优位置 g 和最优值 gbest
            updated = self.update_global_best()
            
            # 记录当前代的最优值
            self.convergence_curve.append(self.gbest)
            self.gbest_history.append(self.g.copy())
            
            # 打印迭代信息
            if verbose and (t + 1) % 20 == 0:
                print(f"迭代 {t+1:3d}: w = {w:.4f}, "
                      f"gbest = {self.gbest:.8f}, "
                      f"位置 = ({self.g[0]:.6f}, {self.g[1]:.6f})")
        
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("步骤4：终止条件判断")
        print("="*60)
        print(f"已达到最大迭代次数 T = {self.T}，搜索结束")
        print(f"优化用时：{elapsed_time:.4f} 秒")
        print("="*60)
        
        return self.g, self.gbest
    
    def plot_convergence_curve(self, save_path=None):
        """
        绘制收敛曲线
        """
        plt.figure(figsize=(12, 6))
        
        # 主图：收敛曲线
        plt.subplot(1, 2, 1)
        iterations = range(len(self.convergence_curve))
        plt.plot(iterations, self.convergence_curve, 'b-', linewidth=2)
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('全局最优值 gbest', fontsize=12)
        plt.title('PSO收敛曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        # 标注最终结果
        final_value = self.convergence_curve[-1]
        plt.axhline(y=final_value, color='r', linestyle='--', alpha=0.5)
        plt.text(len(self.convergence_curve)//2, final_value + 0.2, 
                 f'最终值: {final_value:.6f}', fontsize=11, color='r')
        
        # 副图：惯性权重变化曲线
        plt.subplot(1, 2, 2)
        w_values = [self.calculate_dynamic_inertia_weight(t) for t in range(self.T)]
        plt.plot(range(self.T), w_values, 'g-', linewidth=2)
        plt.xlabel('迭代次数', fontsize=12)
        plt.ylabel('惯性权重 w', fontsize=12)
        plt.title('惯性权重动态变化曲线', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.ylim([self.W_min - 0.05, self.W_max + 0.05])
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_search_process(self, save_path=None):
        """
        绘制搜索过程可视化
        包括函数等高线和粒子搜索轨迹
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 创建背景等高线数据
        x = np.linspace(self.X_min, self.X_max, 100)
        y = np.linspace(self.X_min, self.X_max, 100)
        X, Y = np.meshgrid(x, y)
        Z = self.objective_function(X, Y)
        
        # 1. 初始种群分布（左上）
        ax1 = axes[0, 0]
        ax1.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        ax1.scatter(self.positions_history[0][:, 0], self.positions_history[0][:, 1], 
                   c='white', s=20, edgecolors='black', linewidths=0.5)
        ax1.scatter(self.gbest_history[0][0], self.gbest_history[0][1], 
                   c='red', s=100, marker='*', edgecolors='k', label='初始最优')
        ax1.set_xlabel('x', fontsize=12)
        ax1.set_ylabel('y', fontsize=12)
        ax1.set_title(f'初始种群分布 (迭代0)', fontsize=13)
        ax1.set_xlim([self.X_min, self.X_max])
        ax1.set_ylim([self.X_min, self.X_max])
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 中间种群分布（右上，迭代100次）
        mid_idx = min(100, len(self.positions_history) - 1)
        ax2 = axes[0, 1]
        ax2.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        ax2.scatter(self.positions_history[mid_idx][:, 0], self.positions_history[mid_idx][:, 1], 
                   c='white', s=20, edgecolors='black', linewidths=0.5)
        ax2.scatter(self.gbest_history[mid_idx][0], self.gbest_history[mid_idx][1], 
                   c='red', s=100, marker='*', edgecolors='k', label=f'迭代{mid_idx}最优')
        ax2.set_xlabel('x', fontsize=12)
        ax2.set_ylabel('y', fontsize=12)
        ax2.set_title(f'中间种群分布 (迭代{mid_idx})', fontsize=13)
        ax2.set_xlim([self.X_min, self.X_max])
        ax2.set_ylim([self.X_min, self.X_max])
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. 最终种群分布（左下）
        ax3 = axes[1, 0]
        ax3.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        ax3.scatter(self.positions_history[-1][:, 0], self.positions_history[-1][:, 1], 
                   c='white', s=20, edgecolors='black', linewidths=0.5)
        ax3.scatter(self.g, self.g, c='red', s=150, marker='*', edgecolors='k', 
                   label=f'最终最优')
        ax3.set_xlabel('x', fontsize=12)
        ax3.set_ylabel('y', fontsize=12)
        ax3.set_title(f'最终种群分布 (迭代{self.T})', fontsize=13)
        ax3.set_xlim([self.X_min, self.X_max])
        ax3.set_ylim([self.X_min, self.X_max])
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 4. 全局最优搜索轨迹（右下）
        ax4 = axes[1, 1]
        ax4.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.6)
        gbest_array = np.array(self.gbest_history)
        ax4.plot(gbest_array[:, 0], gbest_array[:, 1], 'r-', linewidth=2, 
                label='全局最优轨迹', alpha=0.8)
        ax4.scatter(gbest_array[0, 0], gbest_array[0, 1], c='yellow', s=80, 
                   marker='o', edgecolors='k', label='起点', zorder=5)
        ax4.scatter(gbest_array[-1, 0], gbest_array[-1, 1], c='red', s=120, 
                   marker='*', edgecolors='k', label='终点', zorder=5)
        ax4.set_xlabel('x', fontsize=12)
        ax4.set_ylabel('y', fontsize=12)
        ax4.set_title('全局最优搜索轨迹', fontsize=13)
        ax4.set_xlim([self.X_min, self.X_max])
        ax4.set_ylim([self.X_min, self.X_max])
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def print_results(self):
        """
        输出最终优化结果
        """
        print("\n" + "="*60)
        print("PSO算法最终优化结果")
        print("="*60)
        print(f"最优解 x = {self.g[0]:.8f}")
        print(f"最优解 y = {self.g[1]:.8f}")
        print(f"最优函数值 f(x,y) = {self.gbest:.8f}")
        print(f"总迭代次数 = {self.T}")
        print(f"初始最优值 = {self.convergence_curve[0]:.6f}")
        print(f"最终最优值 = {self.convergence_curve[-1]:.6f}")
        print(f"改进幅度 = {self.convergence_curve[0] - self.convergence_curve[-1]:.6f}")
        print("="*60)


# ==================== 主程序 ====================
if __name__ == "__main__":
    print("\n目标函数：f(x,y) = 3*cos(x*y) + x + y^2")
    print("搜索范围：x ∈ [-4, 4], y ∈ [-4, 4]")
    print()
    
    # 创建PSO实例
    pso = ParticleSwarmOptimization()
    
    # 执行优化
    best_position, best_value = pso.optimize(verbose=True)
    
    # 输出最终结果
    pso.print_results()
    
    # 绘制收敛曲线
    print("\n正在生成收敛曲线...")
    pso.plot_convergence_curve(save_path='PSO_convergence.png')
    
    # 绘制搜索过程可视化
    print("正在生成搜索过程可视化...")
    pso.plot_search_process(save_path='PSO_search_process.png')
    
    print("\n程序执行完毕！")