from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import os

# 价值迭代实现：包含环境定义、智能体、可视化（含变体规则）
# 注释遵循 `vtry5.py` 风格，为关键参数和步骤提供中文说明

class CarRentalEnv:
    def __init__(self):
        # 环境参数配置
        # MAX_NUM：每个停车场最大可容纳车辆数
        self.MAX_NUM = 20
        self.lam1_rent = 3
        self.lam1_return = 3
        self.lam2_rent = 4
        self.lam2_return = 2
        # MAX_ACTION：单次最大可调度车辆数；DISCOUNT：折扣因子γ
        # CAR_COST/CAR_EARNING：调度成本/租赁收益
        self.MAX_ACTION = 5
        self.DISCOUNT = 0.9
        self.CAR_COST = 2
        self.CAR_EARNING = 10
        self.actions = np.arange(-self.MAX_ACTION, self.MAX_ACTION + 1)
        self.poisson_max = 11
        self.poisson_cache = dict()
        self.state_dimension = (self.MAX_NUM + 1, self.MAX_NUM + 1)

        # 问题变体：1→2方向可免费搬运前2辆；每地点超过10辆需付每车4美元
        # 修改：增加免费正向移动额度与停车超载罚金相关参数，用于问题变体
        # 修改原因：满足“1→2 方向最多2辆免费移动；每地点超过10辆需支付每车4美元停车费”的业务规则
        self.FREE_MOVE = 2           # 每晚从地点1到地点2的免费移动额度
        self.PARKING_LIMIT = 10      # 每个地点的免费停车车辆上限
        self.PARKING_PENALTY = 4     # 超出上限的每车每晚停车罚金

    # 计算泊松概率并做缓存（避免重复计算）
    def poisson_prob(self, n, lam):
        key = n * (self.poisson_max - 1) + lam
        if key not in self.poisson_cache:
            self.poisson_cache[key] = poisson.pmf(n, lam)
        return self.poisson_cache[key]

    # 计算状态-动作的期望回报：租赁收益 + 折扣的下一状态价值 − 调度/停车成本
    def state_value(self, state, action, state_value, constant_returned_cars):
        # returns = - self.CAR_COST * abs(action)  # 原代码：不区分方向与免费额度，所有移动按每车2美元计费
        # 修改：按照变体规则计算移动成本——1→2 方向前2辆免费，其余与反向2→1按每车2美元计费
        # 修改原因：体现“有2个员工可免费搬运一辆车从地点1到地点2”的业务变更
        if action > 0:
            paid_moves = max(0, action - self.FREE_MOVE)
            move_cost = self.CAR_COST * paid_moves
        else:
            move_cost = self.CAR_COST * abs(action)
        returns = - move_cost
        NUM_OF_CARS_1 = min(state[0] - action, self.MAX_NUM)
        NUM_OF_CARS_2 = min(state[1] + action, self.MAX_NUM)
        for rent_1 in range(self.poisson_max):
            for rent_2 in range(self.poisson_max):
                prob = self.poisson_prob(rent_1, self.lam1_rent) * self.poisson_prob(rent_2, self.lam2_rent)
                num_of_cars_1 = NUM_OF_CARS_1
                num_of_cars_2 = NUM_OF_CARS_2
                valid_rent_1 = min(num_of_cars_1, rent_1)
                valid_rent_2 = min(num_of_cars_2, rent_2)
                reward = (valid_rent_1 + valid_rent_2) * self.CAR_EARNING
                num_of_cars_1 -= valid_rent_1
                num_of_cars_2 -= valid_rent_2
                if constant_returned_cars:
                    returned_cars_1 = self.lam1_return
                    returned_cars_2 = self.lam2_return
                    num_of_cars_first_loc = min(num_of_cars_1 + returned_cars_1, self.MAX_NUM)
                    num_of_cars_second_loc = min(num_of_cars_2 + returned_cars_2, self.MAX_NUM)
                    # 停车罚金：每天结束时，超过 PARKING_LIMIT 的部分需支付 PARKING_PENALTY/车
                    penalty_1 = max(0, num_of_cars_first_loc - self.PARKING_LIMIT) * self.PARKING_PENALTY
                    penalty_2 = max(0, num_of_cars_second_loc - self.PARKING_LIMIT) * self.PARKING_PENALTY
                    total_penalty = penalty_1 + penalty_2
                    # returns += prob * (reward + self.DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])  # 原代码：未考虑停车超载罚金
                    returns += prob * (reward - total_penalty + self.DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])  # 修改原因：扣除超载停车罚金
                else:
                    for returned_cars_first_loc in range(self.poisson_max):
                        for returned_cars_second_loc in range(self.poisson_max):
                            prob_return = self.poisson_prob(returned_cars_first_loc, self.lam1_return) * self.poisson_prob(returned_cars_second_loc, self.lam2_return)
                            num_of_cars_first_loc_ = min(num_of_cars_1 + returned_cars_first_loc, self.MAX_NUM)
                            num_of_cars_second_loc_ = min(num_of_cars_2 + returned_cars_second_loc, self.MAX_NUM)
                            prob_ = prob_return * prob
                            # 停车罚金：每天结束时，超过 PARKING_LIMIT 的部分需支付 PARKING_PENALTY/车
                            penalty_1 = max(0, num_of_cars_first_loc_ - self.PARKING_LIMIT) * self.PARKING_PENALTY
                            penalty_2 = max(0, num_of_cars_second_loc_ - self.PARKING_LIMIT) * self.PARKING_PENALTY
                            total_penalty = penalty_1 + penalty_2
                            # returns += prob_ * (reward + self.DISCOUNT * state_value[num_of_cars_first_loc_, num_of_cars_second_loc_])  # 原代码：未考虑停车超载罚金
                            returns += prob_ * (reward - total_penalty + self.DISCOUNT * state_value[num_of_cars_first_loc_, num_of_cars_second_loc_])  # 修改原因：扣除超载停车罚金
        return returns

class Agent:
    # 智能体：维护状态价值 `v` 与策略 `policy`
    def __init__(self, gamma=0.9, precision=1e-4):
        # gamma：折扣因子；precision：收敛阈值
        self.gamma = gamma
        self.precision = precision
        # v：状态价值函数；policy：每个状态的最优动作
        self.v = None
        self.policy = None

    # 价值迭代：按 Bellman 最优方程同步更新价值与策略
    def value_iteration(self, env: CarRentalEnv, constant_returned_cars=True, result_dir: str = None, cmap: str = "magma"):
        self.v = np.ones(env.state_dimension, float)
        self.policy = np.zeros(env.state_dimension, int)
        iteration = 0
        # 单层循环：每轮对所有状态进行价值更新与策略导出
        while True:
            old_value = self.v.copy()
            for i in range(env.state_dimension[0]):
                for j in range(env.state_dimension[1]):
                    action_returns = []
                    for action in env.actions:
                        if (0 <= action <= i) or (-j <= action <= 0):
                            action_returns.append(env.state_value([i, j], action, self.v, constant_returned_cars))
                        else:
                            action_returns.append(-np.inf)
                    max_idx = int(np.argmax(action_returns))
                    max_action = env.actions[max_idx]
                    self.policy[i, j] = max_action
                    self.v[i, j] = action_returns[max_idx]
            max_value_change = abs(old_value - self.v).max()
            print('max value change {}'.format(max_value_change))
            if result_dir:
                visualize(env, self, result_dir, cmap, iteration)
            iteration += 1
            if max_value_change < self.precision:
                break
        return

    def optimal_action(self, env: CarRentalEnv, state, constant_returned_cars=True):
        # 根据当前价值函数评估并返回给定状态的最优动作
        i, j = state
        action_returns = []
        for action in env.actions:
            if (0 <= action <= i) or (-j <= action <= 0):
                action_returns.append(env.state_value([i, j], action, self.v, constant_returned_cars))
            else:
                action_returns.append(-np.inf)
        return env.actions[int(np.argmax(action_returns))]

def visualize(env: CarRentalEnv, agent: Agent, result_dir: str, cmap: str = "magma", iteration: int = None):
    os.makedirs(result_dir, exist_ok=True)
    # 文件名前缀指示算法与内容（策略/价值），便于区分
    
    policy_filename = f'policy_iteration_{iteration}.png' if iteration is not None else 'policy_final.png'
    value_filename = f'value_iteration_value_{iteration}.png' if iteration is not None else 'value_final.png'

    ax1 = sns.heatmap(np.flipud(agent.policy), cmap=cmap, vmin=-5, vmax=5)
    ax1.set_ylabel('cars at parking1', fontsize=18)
    ax1.set_yticks(list(reversed(range(env.MAX_NUM + 1))))
    ax1.set_xlabel('cars at parking2', fontsize=18)
    ax1.set_title(f'policy (Iteration {iteration})' if iteration is not None else 'policy (Final)', fontsize=20)
    plt.savefig(os.path.join(result_dir, policy_filename))
    plt.close()

    vmin = float(np.min(agent.v))
    vmax = float(np.max(agent.v))
    ax2 = sns.heatmap(np.flipud(agent.v), cmap=cmap, vmin=vmin, vmax=vmax)
    ax2.set_ylabel('cars at parking1', fontsize=18)
    ax2.set_yticks(list(reversed(range(env.MAX_NUM + 1))))
    ax2.set_xlabel('cars at parking2', fontsize=18)
    ax2.set_title(f'optimal value (Iteration {iteration})' if iteration is not None else 'optimal value (Final)', fontsize=20)
    plt.savefig(os.path.join(result_dir, value_filename))
    plt.close()

if __name__ == '__main__':
    env = CarRentalEnv()
    agent = Agent(gamma=env.DISCOUNT, precision=1e-4)
    result_dir_path = os.path.join(os.path.dirname(__file__), 'result2')
    # 运行价值迭代，并将可视化结果输出到 `result2` 目录
    agent.value_iteration(env, constant_returned_cars=False, result_dir=result_dir_path, cmap="magma")
    print("状态最优价值：")
    print(agent.v)
    print("V最小/最大:", float(agent.v.min()), float(agent.v.max()))
    print("V(0,0):", float(agent.v[0,0]))
    print("V(10,10):", float(agent.v[10,10]))
    print("V(20,20):", float(agent.v[20,20]))
    print("最优策略：")
    print(agent.policy)
