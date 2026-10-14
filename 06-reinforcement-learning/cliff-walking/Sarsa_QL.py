import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 1. 环境参数（与 DP 完全一致）
# ==========================
W = 3          # 行数
L = 12         # 列数
R_CLIFF = -100
R_STEP = -1
gamma = 1.0
start = (2, 0)
goal = (2, 11)
cliff = set((2, c) for c in range(1, 11))

# 动作定义
actions = ['up', 'down', 'left', 'right']
action_to_idx = {a: i for i, a in enumerate(actions)}
idx_to_action = {i: a for i, a in enumerate(actions)}

# 所有合法状态（用于初始化 Q 表）
states = [(r, c) for r in range(W) for c in range(L)]

# ==========================
# 2. 复用 DP 的 step 函数（关键！）
# ==========================
def step(state, action):
    r, c = state
    nr, nc = r, c
    if action == 'up':
        nr = max(0, r - 1)
    elif action == 'down':
        nr = min(W - 1, r + 1)
    elif action == 'left':
        nc = max(0, c - 1)
    elif action == 'right':
        nc = min(L - 1, c + 1)

    next_state = (nr, nc)

    if next_state in cliff:
        reward = R_CLIFF
        next_state = start
    else:
        reward = R_STEP

    return next_state, reward

# ==========================
# 3. 工具函数
# ==========================
def init_q_table():
    """初始化 Q 表：每个状态有 4 个动作的 Q 值"""
    q_table = {}
    for s in states:
        q_table[s] = np.zeros(len(actions))
    return q_table

def choose_action(state, q_table, epsilon):
    """ε-greedy 策略"""
    if np.random.rand() < epsilon:
        return np.random.choice(len(actions))
    else:
        return np.argmax(q_table[state])

def extract_v_and_policy(q_table):
    """从 Q 表提取 V(s) 和策略 π(s)"""
    V = {}
    policy = {}
    for s in states:
        if s == goal:
            V[s] = 0.0
            policy[s] = 'end'
        else:
            V[s] = np.max(q_table[s])
            best_a_idx = np.argmax(q_table[s])
            policy[s] = idx_to_action[best_a_idx]
    return V, policy

def plot_results(q_sarsa, q_ql, rewards_sarsa, rewards_ql, title_suffix=""):
    """绘制策略图、价值函数、奖励曲线"""
    V_sarsa, pol_sarsa = extract_v_and_policy(q_sarsa)
    V_ql, pol_ql = extract_v_and_policy(q_ql)

    # --- 构建网格 ---
    V_grid_sarsa = np.full((W, L), np.nan)
    V_grid_ql = np.full((W, L), np.nan)
    policy_grid_sarsa = np.full((W, L), '', dtype='<U2')
    policy_grid_ql = np.full((W, L), '', dtype='<U2')

    action_arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

    for r in range(W):
        for c in range(L):
            s = (r, c)

            # --- 处理价值函数：悬崖设为 NaN ---
            if s in cliff:
                V_grid_sarsa[r, c] = np.nan
                V_grid_ql[r, c] = np.nan
            else:
                V_grid_sarsa[r, c] = V_sarsa.get(s, np.nan)
                V_grid_ql[r, c] = V_ql.get(s, np.nan)

            # --- 处理策略显示 ---
            if s in cliff:
                policy_grid_sarsa[r, c] = 'C'
                policy_grid_ql[r, c] = 'C'
            elif s == goal:
                policy_grid_sarsa[r, c] = 'G'
                policy_grid_ql[r, c] = 'G'
            elif s == start:
                policy_grid_sarsa[r, c] = 'S'
                policy_grid_ql[r, c] = 'S'
            else:
                policy_grid_sarsa[r, c] = action_arrows[pol_sarsa[s]]
                policy_grid_ql[r, c] = action_arrows[pol_ql[s]]

    # --- 绘图 ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Cliff Walking: SARSA vs Q-Learning {title_suffix}", fontsize=16)

    # SARSA Value
    im1 = axes[0, 0].imshow(V_grid_sarsa, cmap='viridis', origin='upper')
    axes[0, 0].set_title('SARSA: Value Function')
    for r in range(W):
        for c in range(L):
            if not np.isnan(V_grid_sarsa[r, c]):
                axes[0, 0].text(c, r,str(int(round(V_grid_sarsa[r, c]))),
                                ha='center', va='center', color='white', fontsize=9)
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # QL Value
    im2 = axes[0, 1].imshow(V_grid_ql, cmap='viridis', origin='upper')
    axes[0, 1].set_title('Q-Learning: Value Function')
    for r in range(W):
        for c in range(L):
            if not np.isnan(V_grid_ql[r, c]):
                axes[0, 1].text(c, r, str(int(round(V_grid_ql[r, c]))),
                                ha='center', va='center', color='white', fontsize=9)
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Reward Comparison
    window = 50
    rew_s_smooth = np.convolve(rewards_sarsa, np.ones(window)/window, mode='valid')
    rew_q_smooth = np.convolve(rewards_ql, np.ones(window)/window, mode='valid')
    axes[0, 2].plot(rewards_sarsa, alpha=0.3, label='SARSA')
    axes[0, 2].plot(rewards_ql, alpha=0.3, label='Q-Learning')
    axes[0, 2].plot(rew_s_smooth, label='SARSA (smoothed)')
    axes[0, 2].plot(rew_q_smooth, label='QL (smoothed)')
    axes[0, 2].axhline(y=-13, color='k', linestyle='--', label='Optimal (-13)')
    axes[0, 2].set_title('Total Reward per Episode')
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].legend()
    axes[0, 2].grid(True)

    # --- 新增：滑动窗口标准差曲线 ---
    window_std = 20  # 滑动窗口大小

    # 计算滑动窗口标准差
    def sliding_std(rewards, window):
        return [np.std(rewards[i:i + window]) for i in range(len(rewards) - window + 1)]

    sarsa_std = sliding_std(rewards_sarsa, window_std)
    ql_std = sliding_std(rewards_ql, window_std)
    episodes_std = range(window_std, len(rewards_sarsa) + 1)

    # 绘制标准差曲线
    ax_std = fig.add_subplot(2, 3, 6)  # 利用原画布的空白子图
    ax_std.plot(episodes_std, sarsa_std, label='SARSA Std', color='blue')
    ax_std.plot(episodes_std, ql_std, label='Q-Learning Std', color='orange')
    ax_std.set_title('Reward Std (Sliding Window)')
    ax_std.set_xlabel('Episode')
    ax_std.set_ylabel('Standard Deviation')
    ax_std.legend()
    ax_std.grid(True)

    # SARSA Policy
    axes[1, 0].set_title('SARSA Policy')
    _draw_policy(axes[1, 0], policy_grid_sarsa)

    # QL Policy
    axes[1, 1].set_title('Q-Learning Policy')
    _draw_policy(axes[1, 1], policy_grid_ql)

    # Hide unused subplot
    axes[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # --- 新增：最后50轮奖励的箱线图 ---
    fig, ax = plt.subplots(figsize=(8, 4))
    # 提取最后50轮的奖励
    sarsa_last50 = rewards_sarsa[-50:]
    ql_last50 = rewards_ql[-50:]
    # 绘制箱线图
    ax.boxplot([sarsa_last50, ql_last50], labels=['SARSA', 'Q-Learning'])
    ax.set_title('Reward Distribution (Last 50 Episodes)')
    ax.set_ylabel('Total Reward per Episode')
    ax.axhline(y=-13, color='k', linestyle='--', label='Optimal')
    ax.legend()
    plt.tight_layout()
    plt.show()


def _draw_policy(ax, grid):
    ax.set_xlim(-0.5, L - 0.5)
    ax.set_ylim(-0.5, W - 0.5)
    ax.invert_yaxis()
    ax.set_xticks(range(L))
    ax.set_yticks(range(W))
    ax.grid(True, color='gray', linestyle='--', linewidth=0.5)

    for r in range(W):
        for c in range(L):
            char = grid[r, c]
            if char == 'C':
                ax.text(c, r, '■', ha='center', va='center', fontsize=16, color='red')
            elif char == 'S':
                ax.text(c, r, 'S', ha='center', va='center', fontsize=14, color='blue')
            elif char == 'G':
                ax.text(c, r, 'G', ha='center', va='center', fontsize=14, color='green')
            elif char in ['↑', '↓', '←', '→']:
                ax.text(c, r, char, ha='center', va='center', fontsize=16, color='black')

# ==========================
# 4. SARSA 算法
# ==========================
def run_sarsa(episodes=20000, alpha=0.1, epsilon=0.2):
    q_table = init_q_table()
    rewards = []

    for ep in range(episodes):
        state = start
        action_idx = choose_action(state, q_table, epsilon)
        total_reward = 0

        while state != goal:
            action = idx_to_action[action_idx]
            next_state, reward = step(state, action)
            total_reward += reward

            if next_state == goal:
                # 终止状态：无后续动作
                target = reward
                q_table[state][action_idx] += alpha * (target - q_table[state][action_idx])
                break
            else:
                next_action_idx = choose_action(next_state, q_table, epsilon)
                target = reward + gamma * q_table[next_state][next_action_idx]
                q_table[state][action_idx] += alpha * (target - q_table[state][action_idx])

                state = next_state
                action_idx = next_action_idx

        rewards.append(total_reward)

    return q_table, rewards

# ==========================
# 5. Q-Learning 算法
# ==========================
def run_q_learning(episodes=20000, alpha=0.1, epsilon=0.2):
    q_table = init_q_table()
    rewards = []

    for ep in range(episodes):
        state = start
        total_reward = 0

        while state != goal:
            action_idx = choose_action(state, q_table, epsilon)
            action = idx_to_action[action_idx]
            next_state, reward = step(state, action)
            total_reward += reward

            if next_state == goal:
                target = reward
            else:
                target = reward + gamma * np.max(q_table[next_state])

            q_table[state][action_idx] += alpha * (target - q_table[state][action_idx])
            state = next_state

        rewards.append(total_reward)

    return q_table, rewards

# ==========================
# 6. 主程序
# ==========================
if __name__ == "__main__":
    print("Running SARSA...")
    q_sarsa, rewards_sarsa = run_sarsa(episodes=500, alpha=0.1, epsilon=0.1)

    print("Running Q-Learning...")
    q_ql, rewards_ql = run_q_learning(episodes=500, alpha=0.1, epsilon=0.1)

    print("Plotting results...")
    plot_results(q_sarsa, q_ql, rewards_sarsa, rewards_ql)

    # 最终性能
    print(f"\nFinal 50-episode average reward:")
    print(f"SARSA:      {np.mean(rewards_sarsa[-50]):.2f}")
    print(f"Q-Learning: {np.mean(rewards_ql[-50]):.2f}")
    print(f"Optimal:    -13.00")

    print("Q-values near goal:")
    print(f"(5,11): {q_ql[(5, 11)]} → V = {np.max(q_ql[(5, 11)])}")
    print(f"(6,11): {q_ql[(6, 11)]} → V = {np.max(q_ql[(6, 11)])}")
    print(f"Start (6,0): V = {np.max(q_ql[(6, 0)])}")