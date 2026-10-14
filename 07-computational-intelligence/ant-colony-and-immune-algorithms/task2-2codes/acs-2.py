import time
import numpy as np
import matplotlib.pyplot as plt


def g2d(G):
    l = G.shape[0]
    D = np.zeros((l * l, l * l), dtype=float)
    for i in range(1, l + 1):
        for j in range(1, l + 1):
            if G[i - 1, j - 1] == 0:
                for m in range(1, l + 1):
                    for n in range(1, l + 1):
                        if G[m - 1, n - 1] == 0:
                            im = abs(i - m)
                            jn = abs(j - n)
                            if im + jn == 1 or (im == 1 and jn == 1):
                                D[(i - 1) * l + j - 1, (m - 1) * l + n - 1] = (im + jn) ** 0.5
    return D


def grid_to_xy(grid_idx, MM, a=1):
    ix = a * ((grid_idx % MM) - 0.5)
    if ix == -0.5:
        ix = MM - 0.5
    iy = a * (MM + 0.5 - np.ceil(grid_idx / MM))
    return ix, iy


def build_eta(N, E, MM, a):
    E_x, E_y = grid_to_xy(E, MM, a)
    Eta = np.zeros(N, dtype=float)
    for i in range(1, N + 1):
        ix, iy = grid_to_xy(i, MM, a)
        if i != E:
            Eta[i - 1] = 1 / (((ix - E_x) ** 2 + (iy - E_y) ** 2) ** 0.5)
        else:
            Eta[i - 1] = 100.0
    return Eta


def get_feasible_nodes(D, current_node, visited):
    row = D[current_node - 1, :]
    neighbors = np.where(row > 0)[0] + 1
    feasible = []
    for node in neighbors:
        if visited[node - 1] == 0:
            feasible.append(int(node))
    return feasible


def roulette_selection(probabilities):
    cumulative = np.cumsum(probabilities)
    r = np.random.rand()
    idx = np.where(cumulative >= r)[0]
    if len(idx) == 0:
        return len(probabilities) - 1
    return int(idx[0])


def acs_select_next_node(current_node, feasible_nodes, Tau, Eta, Alpha, Beta, q0):
    desirability = np.zeros(len(feasible_nodes), dtype=float)
    for i in range(len(feasible_nodes)):
        node = feasible_nodes[i]
        desirability[i] = (Tau[current_node - 1, node - 1] ** Alpha) * (Eta[node - 1] ** Beta)
    if np.sum(desirability) == 0:
        probabilities = np.ones(len(feasible_nodes), dtype=float) / len(feasible_nodes)
    else:
        probabilities = desirability / np.sum(desirability)
    q = np.random.rand()
    if q <= q0:
        best_idx = int(np.argmax(desirability))
        return feasible_nodes[best_idx]
    chosen_idx = roulette_selection(probabilities)
    return feasible_nodes[chosen_idx]


def local_pheromone_update(Tau, i, j, rho_local, tau0):
    Tau[i - 1, j - 1] = (1 - rho_local) * Tau[i - 1, j - 1] + rho_local * tau0
    Tau[j - 1, i - 1] = Tau[i - 1, j - 1]
    return Tau  # 明确返回Tau


def global_pheromone_update(Tau, best_path, best_length, rho_global, Q):
    if best_length > 0 and len(best_path) >= 2:
        delta_tau = Q / best_length
        for s in range(len(best_path) - 1):
            x = best_path[s]
            y = best_path[s + 1]
            Tau[x - 1, y - 1] = (1 - rho_global) * Tau[x - 1, y - 1] + rho_global * delta_tau
            Tau[y - 1, x - 1] = Tau[x - 1, y - 1]
    return Tau


def run_acs(D, Eta, Tau, S, E, K, M, Alpha, Beta, Rho, Q):
    N = D.shape[0]
    ROUTES = [[[] for _ in range(M)] for _ in range(K)]
    PL = np.zeros((K, M), dtype=float)
    q0 = 0.9
    rho_local = 0.1
    tau0 = 1.0 / (N * np.mean(Eta))  # 修改：基于启发式信息计算初始信息素
    
    global_best_length = float("inf")
    global_best_path = []
    convergence = np.full(K, np.nan, dtype=float)
    
    for k in range(K):
        for m in range(M):
            current = S
            path = [S]
            path_length = 0.0
            visited = np.zeros(N, dtype=int)
            visited[S - 1] = 1
            feasible_nodes = get_feasible_nodes(D, current, visited)
            
            while current != E and len(feasible_nodes) >= 1:
                next_node = acs_select_next_node(current, feasible_nodes, Tau, Eta, Alpha, Beta, q0)
                path.append(next_node)
                path_length = path_length + D[current - 1, next_node - 1]
                
                # 关键修改：确保tau0正确传递给局部信息素更新函数
                Tau = local_pheromone_update(Tau, current, next_node, rho_local, tau0)
                
                current = next_node
                visited[current - 1] = 1
                feasible_nodes = get_feasible_nodes(D, current, visited)
            
            ROUTES[k][m] = path
            if path[-1] == E:
                PL[k, m] = path_length
            else:
                PL[k, m] = 0.0
        
        # 找到本次迭代的最优解
        valid_idx = np.where(PL[k, :] > 0)[0]
        if len(valid_idx) > 0:
            iter_best_ant = int(valid_idx[np.argmin(PL[k, valid_idx])])
            iter_best_length = PL[k, iter_best_ant]
            iter_best_path = ROUTES[k][iter_best_ant]
            
            if iter_best_length < global_best_length:
                global_best_length = iter_best_length
                global_best_path = iter_best_path.copy()
        
        # 关键修改：重新设计全局信息素更新逻辑
        if np.isfinite(global_best_length) and len(global_best_path) >= 2:
            # 只对最优路径进行信息素增强
            Tau = global_pheromone_update(Tau, global_best_path, global_best_length, Rho, Q)
            
            # 信息素挥发（避免过早收敛）
            evaporation_rate = 0.05
            Tau = (1 - evaporation_rate) * Tau
            Tau[Tau < 0.1] = 0.1  # 设置信息素下限
        
        convergence[k] = global_best_length if np.isfinite(global_best_length) else np.nan
    
    return ROUTES, PL, Tau, global_best_path, global_best_length, convergence


def plot_results(G, MM, best_path, convergence, a):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 收敛曲线图
    plt.figure(1, figsize=(12, 5))
    plt.subplot(1, 2, 1)
    valid_indices = ~np.isnan(convergence)
    if np.any(valid_indices):
        plt.plot(np.where(valid_indices)[0], convergence[valid_indices], 'b-', linewidth=2)
    plt.grid(True)
    plt.title('ACS收敛曲线变化趋势')
    plt.xlabel('迭代次数')
    plt.ylabel('全局最优路径长度')
    plt.xlim(0, len(convergence)-1)
    
    # 添加最终值标注
    if len(convergence) > 0 and not np.isnan(convergence[-1]):
        plt.text(len(convergence)-1, convergence[-1], 
                f'最终: {convergence[-1]:.2f}', 
                verticalalignment='bottom', horizontalalignment='right')
    
    # 路径图
    plt.subplot(1, 2, 2)
    plt.xlim(0, MM)
    plt.ylim(0, MM)
    
    for i in range(1, MM + 1):
        for j in range(1, MM + 1):
            x1 = j - 1
            y1 = MM - i
            x2 = j
            y2 = MM - i
            x3 = j
            y3 = MM - i + 1
            x4 = j - 1
            y4 = MM - i + 1
            if G[i - 1, j - 1] == 1:
                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], color='black', edgecolor='gray', linewidth=0.5)
            else:
                plt.fill([x1, x2, x3, x4], [y1, y2, y3, y4], color='white', edgecolor='gray', linewidth=0.5)
    
    ax = plt.gca()
    ax.set_yticks(np.linspace(0, MM, 11))
    ax.set_yticklabels([20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0])
    plt.title('ACS机器人最优运动轨迹')
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    
    if len(best_path) >= 2:
        Rx = np.zeros(len(best_path))
        Ry = np.zeros(len(best_path))
        for idx in range(len(best_path)):
            Rx[idx] = a * ((best_path[idx] % MM) - 0.5)
            if Rx[idx] == -0.5:
                Rx[idx] = MM - 0.5
            Ry[idx] = a * (MM + 0.5 - np.ceil(best_path[idx] / MM))
        
        plt.plot(Rx, Ry, 'r-', linewidth=2, label='最优路径')
        plt.scatter(Rx[0], Ry[0], c='g', s=100, marker='o', label='起点')
        plt.scatter(Rx[-1], Ry[-1], c='b', s=100, marker='s', label='终点')
        plt.legend()
    
    plt.tight_layout()
    plt.show()
    return None


def main():
    # 1.操作描述
    G = np.array([
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ], dtype=int)
    
    MM = G.shape[0]
    S = 1
    E = MM * MM
    a = 1
    D = g2d(G)
    N = D.shape[0]
    Eta = build_eta(N, E, MM, a)
    
    # 参数调整（避免过早收敛）
    K = 100
    M = 50
    Alpha = 1.0
    Beta = 2.0  # 调整启发式信息权重
    Rho = 0.5  # 增加全局信息素更新强度
    Q = 100.0  # 增加信息素增量
    
    # 调整初始信息素
    tau_mean = 1.0 / (N * np.mean(Eta[Eta < 100]))
    Tau = np.ones((N, N), dtype=float) * tau_mean
    
    # 运行ACS算法
    start_time = time.time()
    ROUTES, PL, Tau, best_path, best_length, convergence = run_acs(D, Eta, Tau, S, E, K, M, Alpha, Beta, Rho, Q)
    elapsed_time = time.time() - start_time
    
    print("ACS最优路径长度:", best_length)
    print("ACS最优路径节点数:", len(best_path))
    print("ACS运行时间(秒):", elapsed_time)
    print("最终收敛值:", convergence[-1])
    
    plot_results(G, MM, best_path, convergence, a)
    return None


if __name__ == "__main__":
    main()