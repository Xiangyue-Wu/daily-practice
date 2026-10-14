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


def select_next_node(current_node, feasible_nodes, Tau, Eta, Alpha, Beta):
    desirability = np.zeros(len(feasible_nodes), dtype=float)
    for i in range(len(feasible_nodes)):
        node = feasible_nodes[i]
        desirability[i] = (Tau[current_node - 1, node - 1] ** Alpha) * (Eta[node - 1] ** Beta)
    if np.sum(desirability) == 0:
        probabilities = np.ones(len(feasible_nodes), dtype=float) / len(feasible_nodes)
    else:
        probabilities = desirability / np.sum(desirability)
    chosen_idx = roulette_selection(probabilities)
    return feasible_nodes[chosen_idx]


def update_tau_mmas(Tau, best_path, best_length, Rho, Q, tau_min, tau_max):
    Tau = (1 - Rho) * Tau
    if best_length > 0 and len(best_path) >= 2:
        delta_tau = Q / best_length
        for s in range(len(best_path) - 1):
            x = best_path[s]
            y = best_path[s + 1]
            Tau[x - 1, y - 1] = Tau[x - 1, y - 1] + delta_tau
            Tau[y - 1, x - 1] = Tau[x - 1, y - 1]
    Tau = np.clip(Tau, tau_min, tau_max)
    return Tau


def run_mmas(D, Eta, Tau, S, E, K, M, Alpha, Beta, Rho, Q):
    N = D.shape[0]
    ROUTES = [[[] for _ in range(M)] for _ in range(K)]
    PL = np.zeros((K, M), dtype=float)
    global_best_path = []
    global_best_length = float("inf")
    convergence = np.full(K, np.nan, dtype=float)
    tau_max = np.max(Tau)
    tau_min = tau_max / 10.0
    for k in range(K):
        for m in range(M):
            current = S
            path = [S]
            path_length = 0.0
            visited = np.zeros(N, dtype=int)
            visited[S - 1] = 1
            feasible_nodes = get_feasible_nodes(D, current, visited)
            while current != E and len(feasible_nodes) >= 1:
                next_node = select_next_node(current, feasible_nodes, Tau, Eta, Alpha, Beta)
                path.append(next_node)
                path_length = path_length + D[current - 1, next_node - 1]
                current = next_node
                visited[current - 1] = 1
                feasible_nodes = get_feasible_nodes(D, current, visited)
            ROUTES[k][m] = path
            if path[-1] == E:
                PL[k, m] = path_length
            else:
                PL[k, m] = 0.0
        valid_idx = np.where(PL[k, :] > 0)[0]
        if len(valid_idx) > 0:
            iter_best_ant = int(valid_idx[np.argmin(PL[k, valid_idx])])
            iter_best_length = PL[k, iter_best_ant]
            iter_best_path = ROUTES[k][iter_best_ant]
            if iter_best_length < global_best_length:
                global_best_length = iter_best_length
                global_best_path = iter_best_path.copy()
                tau_max = 1.0 / (Rho * global_best_length)
                tau_min = tau_max / 10.0
        if np.isfinite(global_best_length):
            Tau = update_tau_mmas(Tau, global_best_path, global_best_length, Rho, Q, tau_min, tau_max)
            convergence[k] = global_best_length
        else:
            Tau = np.clip((1 - Rho) * Tau, tau_min, tau_max)
    return ROUTES, PL, Tau, global_best_path, global_best_length, convergence


def plot_results(G, MM, best_path, convergence, a):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(1)
    plt.plot(convergence)
    plt.grid(True)
    plt.title('MMAS收敛曲线变化趋势')
    plt.xlabel('迭代次数')
    plt.ylabel('全局最优路径长度')
    plt.figure(2)
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
    plt.title('MMAS机器人最优运动轨迹')
    plt.xlabel('坐标x')
    plt.ylabel('坐标y')
    if len(best_path) >= 2:
        Rx = np.array(best_path, dtype=float)
        Ry = np.array(best_path, dtype=float)
        for idx in range(len(best_path)):
            Rx[idx] = a * ((best_path[idx] % MM) - 0.5)
            if Rx[idx] == -0.5:
                Rx[idx] = MM - 0.5
            Ry[idx] = a * (MM + 0.5 - np.ceil(best_path[idx] / MM))
        plt.plot(Rx, Ry, 'r-', linewidth=2)
        plt.scatter(Rx[0], Ry[0], c='g', s=50)
        plt.scatter(Rx[-1], Ry[-1], c='b', s=50)
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
    K = 100
    M = 50
    Alpha = 1
    Beta = 7
    Rho = 0.3
    Q = 1
    Tau = np.ones((N, N), dtype=float) * 8.0
    # 2.操作描述
    start_time = time.time()
    ROUTES, PL, Tau, best_path, best_length, convergence = run_mmas(D, Eta, Tau, S, E, K, M, Alpha, Beta, Rho, Q)
    _ = ROUTES
    _ = PL
    _ = Tau
    elapsed_time = time.time() - start_time
    print("MMAS最优路径长度:", best_length)
    print("MMAS最优路径节点数:", len(best_path))
    print("MMAS运行时间(秒):", elapsed_time)
    plot_results(G, MM, best_path, convergence, a)
    return None


if __name__ == "__main__":
    main()
