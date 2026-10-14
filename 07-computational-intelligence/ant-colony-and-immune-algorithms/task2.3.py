import numpy as np
import matplotlib.pyplot as plt

# ==================== 目标函数 ====================
def f(x):
    """多峰函数：求最大值"""
    if x <= 0:
        x = 1e-6
    return 10 + np.sin(1/x) / ((x - 0.16)**2 + 0.1)


# ==================== 二进制编码 ====================
L = 20  # 编码长度
PRECISION = 1 / (2**L - 1)  # 精度 ≈ 9.5e-7

def encode(x):
    """实数 -> 二进制串"""
    decimal = int(x / PRECISION)
    return format(decimal, 'b').zfill(L)

def decode(binary):
    """二进制串 -> 实数"""
    x = int(binary, 2) * PRECISION
    return max(1e-6, min(1.0, x))


# ==================== IPA算法 ====================
def run_ipa():
    # 参数设置
    POP_SIZE = 50      # 种群大小
    PC = 0.8           # 交叉概率
    PM = 0.1           # 变异概率
    PV = 0.5           # 接种概率
    TOP_K = 5          # 疫苗提取个数
    T0 = 100           # 初始温度
    ALPHA = 0.95       # 温度衰减
    MAX_GEN = 100      # 最大代数
    
    # 1. 初始化种群
    pop = [''.join(np.random.choice(['0','1'], L)) for _ in range(POP_SIZE)]
    T = T0
    
    best_fitness_history = []
    best_x_history = []
    
    for gen in range(MAX_GEN):
        # 解码并计算适应度
        x_vals = [decode(ind) for ind in pop]
        fitness = np.array([f(x) for x in x_vals])
        
        # 记录最优
        best_idx = np.argmax(fitness)
        best_x = x_vals[best_idx]
        best_fit = fitness[best_idx]
        best_fitness_history.append(best_fit)
        best_x_history.append(best_x)
        
        if gen % 20 == 0:
            print(f"Gen {gen}: best_x = {best_x:.6f}, best_f = {best_fit:.6f}")
        
        # 2. 疫苗提取（取前TOP_K个最优个体的多数表决）
        sorted_idx = np.argsort(fitness)[::-1]
        top_inds = [pop[i] for i in sorted_idx[:TOP_K]]
        vaccine = ''
        for i in range(L):
            ones = sum(1 for ind in top_inds if ind[i] == '1')
            vaccine += '1' if ones > TOP_K/2 else '0'
        
        # 3. 生成下一代
        new_pop = []
        
        # 精英保留
        new_pop.append(pop[best_idx])
        
        while len(new_pop) < POP_SIZE:
            # 选择父代（随机选择）
            p1 = pop[np.random.randint(POP_SIZE)]
            p2 = pop[np.random.randint(POP_SIZE)]
            
            # 交叉
            if np.random.rand() < PC:
                pos = np.random.randint(1, L)
                c1 = p1[:pos] + p2[pos:]
                c2 = p2[:pos] + p1[pos:]
            else:
                c1, c2 = p1, p2
            
            # 变异
            for i in range(L):
                if np.random.rand() < PM:
                    c1 = c1[:i] + ('1' if c1[i]=='0' else '0') + c1[i+1:]
                if np.random.rand() < PM:
                    c2 = c2[:i] + ('1' if c2[i]=='0' else '0') + c2[i+1:]
            
            # 疫苗接种
            if np.random.rand() < PV:
                # 随机选择一半位置接种
                positions = np.random.choice(L, L//2, replace=False)
                c1_list = list(c1)
                for pos in positions:
                    c1_list[pos] = vaccine[pos]
                c1 = ''.join(c1_list)
            
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)
        
        # 4. 免疫选择（退火选择）
        for i in range(POP_SIZE):
            x_old = decode(pop[i])
            x_new = decode(new_pop[i])
            f_old = f(x_old)
            f_new = f(x_new)
            
            if f_new > f_old:
                pop[i] = new_pop[i]  # 更好，直接接受
            else:
                delta = f_old - f_new
                if np.random.rand() < np.exp(-delta / T):
                    pop[i] = new_pop[i]  # 退火接受劣质个体
        
        # 温度衰减
        T *= ALPHA
    
    return best_x_history, best_fitness_history


# ==================== 运行 ====================
if __name__ == "__main__":
    best_x_history, best_fitness_history = run_ipa()
    
    # 最终结果
    print(f"\n最优解: x = {best_x_history[-1]:.8f}")
    print(f"最大值: f(x) = {best_fitness_history[-1]:.8f}")
    
    # 画图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 左图：收敛曲线
    ax1.plot(best_fitness_history)
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Best Fitness')
    ax1.set_title('Convergence Curve')
    ax1.grid(True)
    
    # 右图：函数图像
    x_vals = np.linspace(0.01, 1, 1000)
    y_vals = [f(x) for x in x_vals]
    ax2.plot(x_vals, y_vals)
    ax2.scatter(best_x_history[-1], best_fitness_history[-1], 
                color='red', s=100, label='Best Found')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.set_title('Objective Function')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()