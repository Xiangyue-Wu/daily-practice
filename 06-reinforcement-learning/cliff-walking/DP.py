import numpy as np
import matplotlib.pyplot as plt

# ==========================
# 1. 环境参数
# ==========================
W = 3  # 高度（行数）
L = 12  # 长度（列数）
R_CLIFF = -100  # 掉落悬崖的惩罚
R_STEP = -1  # 每步正常奖励
gamma = 1.0
theta = 1e-6
actions = ['up', 'down', 'left', 'right']
action_to_idx = {'up': 0, 'down': 1, 'left': 2, 'right': 3}
action_arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}

# 悬崖区域：第2行（索引2），列1~10
cliff = set((2, c) for c in range(1, 11))
start = (2, 0)
goal = (2, 11)

# 所有安全状态（可停留的状态）
states = []
for r in range(W):
    for c in range(L):
        if (r, c) == goal or (r, c) not in cliff:
            states.append((r, c))

# 初始化价值函数 V(s)
V = {s: 0.0 for s in states}


# ==========================
# 2. 环境转移函数
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

    # 判断是否踩到悬崖
    if next_state in cliff:
        reward = R_CLIFF
        next_state = start  # 重置到起点
    else:
        reward = R_STEP

    return next_state, reward


# ==========================
# 3. 值迭代
# ==========================
iteration = 0
while True:
    delta = 0
    for s in states:
        if s == goal:
            continue  # 终点价值固定为0

        v_old = V[s]
        q_values = []
        for a in actions:
            next_s, reward = step(s, a)
            # next_s 一定是安全状态（因为踩崖会重置到 start ∈ states）
            q = reward + gamma * V[next_s]
            q_values.append(q)

        V[s] = max(q_values)
        delta = max(delta, abs(v_old - V[s]))

    iteration += 1
    if delta < theta:
        print(f"值迭代收敛！共迭代 {iteration} 次")
        break

# ==========================
# 4. 提取最优策略
# ==========================
policy = {}
for s in states:
    if s == goal:
        policy[s] = 'end'
        continue

    best_action = None
    best_q = -float('inf')
    for a in actions:
        next_s, reward = step(s, a)
        q = reward + gamma * V[next_s]
        if q > best_q:
            best_q = q
            best_action = a
    policy[s] = best_action

# ==========================
# 5. 可视化
# ==========================
# 创建网格用于绘图
V_grid = np.full((W, L), np.nan)  # 价值函数
policy_grid = np.full((W, L), '', dtype='<U2')  # 策略箭头

# 填充数据
for r in range(W):
    for c in range(L):
        if (r, c) in cliff:
            V_grid[r, c] = np.nan  # 悬崖不显示
            policy_grid[r, c] = 'C'  # C 表示 cliff
        elif (r, c) in V:
            V_grid[r, c] = V[(r, c)]
            if (r, c) == goal:
                policy_grid[r, c] = 'G'
            elif (r, c) == start:
                policy_grid[r, c] = 'S'
            else:
                policy_grid[r, c] = action_arrows[policy[(r, c)]]

# --- 绘图 ---
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# (a) 价值函数热力图
im1 = axes[0].imshow(V_grid, cmap='viridis', origin='upper')
axes[0].set_title('Value Function (V(s))', fontsize=14)
axes[0].set_xlabel('Column')
axes[0].set_ylabel('Row')

# 标注数值（可选）
for r in range(W):
    for c in range(L):
        if not np.isnan(V_grid[r, c]):
            axes[0].text(c, r, f'{V_grid[r, c]:.0f}',
                         ha='center', va='center', color='white', fontsize=9)

plt.colorbar(im1, ax=axes[0])

# (b) 策略图（箭头）
axes[1].set_title('Optimal Policy', fontsize=14)
axes[1].set_xlim(-0.5, L - 0.5)
axes[1].set_ylim(-0.5, W - 0.5)
axes[1].invert_yaxis()  # 使 (0,0) 在左上角

for r in range(W):
    for c in range(L):
        char = policy_grid[r, c]
        if char == 'C':
            axes[1].text(c, r, '■', ha='center', va='center', fontsize=16, color='red')
        elif char == 'S':
            axes[1].text(c, r, 'S', ha='center', va='center', fontsize=14, color='blue')
        elif char == 'G':
            axes[1].text(c, r, 'G', ha='center', va='center', fontsize=14, color='green')
        elif char in ['↑', '↓', '←', '→']:
            axes[1].text(c, r, char, ha='center', va='center', fontsize=16, color='black')

axes[1].set_xticks(range(L))
axes[1].set_yticks(range(W))
axes[1].grid(True, color='gray', linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig('cliff_walking_dp_result.png', dpi=150)
plt.show()

# ==========================
# 6. 模拟一次轨迹（验证策略）
# ==========================
print("\n模拟从起点按最优策略行走：")
state = start
path = [state]
total_reward = 0
steps = 0
while state != goal and steps < 100:  # 防止无限循环
    action = policy[state]
    if action == 'end':
        break
    next_state, reward = step(state, action)
    total_reward += reward
    path.append(next_state)
    state = next_state
    steps += 1

print("路径:", " → ".join([str(s) for s in path]))
print(f"总步数: {steps}, 总奖励: {total_reward}")