from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import poisson
import seaborn as sns
import os
# 策略迭代实现：包含环境定义、智能体、可视化
# 注释遵循 `ls.py` 风格，为关键参数和步骤提供中文说明

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

    # 计算泊松概率并做缓存（避免重复计算）
    def poisson_prob(self, n, lam):
        key = n * (self.poisson_max - 1) + lam
        if key not in self.poisson_cache:
            self.poisson_cache[key] = poisson.pmf(n, lam)
        return self.poisson_cache[key]

    # 计算状态-动作的期望回报：租赁收益 + 折扣的下一状态价值 − 调度成本
    def state_value(self, state, action, state_value, constant_returned_cars):
        returns = - self.CAR_COST * abs(action)
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
                    returns += prob * (reward + self.DISCOUNT * state_value[num_of_cars_first_loc, num_of_cars_second_loc])
                else:
                    for returned_cars_first_loc in range(self.poisson_max):
                        for returned_cars_second_loc in range(self.poisson_max):
                            prob_return = self.poisson_prob(returned_cars_first_loc, self.lam1_return) * self.poisson_prob(returned_cars_second_loc, self.lam2_return)
                            num_of_cars_first_loc_ = min(num_of_cars_1 + returned_cars_first_loc, self.MAX_NUM)
                            num_of_cars_second_loc_ = min(num_of_cars_2 + returned_cars_second_loc, self.MAX_NUM)
                            prob_ = prob_return * prob
                            returns += prob_ * (reward + self.DISCOUNT * state_value[num_of_cars_first_loc_, num_of_cars_second_loc_])
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

    # 策略迭代：交替进行策略评估与策略提升，直到策略稳定
    def policy_iteration(self, env: CarRentalEnv, constant_returned_cars=True, result_dir: str = None, cmap: str = "magma"):
        self.v = np.zeros(env.state_dimension, float)
        self.policy = np.zeros(env.state_dimension, int)
        iteration = 0
        # 外层循环：策略提升迭代
        while True:
            # 内层循环：在固定策略下评估价值直至收敛
            while True:
                old_value = self.v.copy()
                for i in range(env.state_dimension[0]):
                    for j in range(env.state_dimension[1]):
                        action = self.policy[i, j]
                        self.v[i, j] = env.state_value([i, j], action, self.v, constant_returned_cars)
                max_value_change = abs(old_value - self.v).max()
                print('policy evaluation max value change {}'.format(max_value_change))
                if max_value_change < self.precision:
                    break
            old_policy = self.policy.copy()
            for i in range(env.state_dimension[0]):
                for j in range(env.state_dimension[1]):
                    action_returns = []
                    for action in env.actions:
                        if (0 <= action <= i) or (-j <= action <= 0):
                            action_returns.append(env.state_value([i, j], action, self.v, constant_returned_cars))
                        else:
                            action_returns.append(-np.inf)
                    max_idx = int(np.argmax(action_returns))
                    self.policy[i, j] = env.actions[max_idx]
            if result_dir:
                visualize(env, self, result_dir, cmap, iteration)
            iteration += 1
            if np.array_equal(old_policy, self.policy):
                break
        return

    # 给定状态，返回基于当前价值函数计算的最优动作
    def optimal_action(self, env: CarRentalEnv, state, constant_returned_cars=True):
        i, j = state
        action_returns = []
        for action in env.actions:
            if (0 <= action <= i) or (-j <= action <= 0):
                action_returns.append(env.state_value([i, j], action, self.v, constant_returned_cars))
            else:
                action_returns.append(-np.inf)
        return env.actions[int(np.argmax(action_returns))]

def visualize(env: CarRentalEnv, agent: Agent, result_dir: str, cmap: str = "viridis", iteration: int = None):
    os.makedirs(result_dir, exist_ok=True)
    # 文件名前缀指示算法与内容（策略/价值），便于区分
    
    policy_filename = f'policy_iteration_policy_{iteration}.png' if iteration is not None else 'policy_final.png'
    value_filename = f'policy_iteration_value_{iteration}.png' if iteration is not None else 'value_final.png'

    ax1 = sns.heatmap(np.flipud(agent.policy), cmap=cmap, vmin=-5, vmax=5)
    ax1.set_ylabel('cars at parking1', fontsize=18)
    ax1.set_yticks(list(reversed(range(env.MAX_NUM + 1))))
    ax1.set_xlabel('cars at parking2', fontsize=18)
    ax1.set_title(f'policy_iteration_policy (Iteration {iteration})' if iteration is not None else 'policy_iteration_policy (Final)', fontsize=20)
    plt.savefig(os.path.join(result_dir, policy_filename))
    plt.close()

    ax2 = sns.heatmap(np.flipud(agent.v), cmap=cmap, vmin=0, vmax=625)
    ax2.set_ylabel('cars at parking1', fontsize=18)
    ax2.set_yticks(list(reversed(range(env.MAX_NUM + 1))))
    ax2.set_xlabel('cars at parking2', fontsize=18)
    ax2.set_title(f'policy_iteration_optimal value (Iteration {iteration})' if iteration is not None else 'policy_iteration_optimal value (Final)', fontsize=20)
    plt.savefig(os.path.join(result_dir, value_filename))
    plt.close()

if __name__ == '__main__':
    env = CarRentalEnv()
    agent = Agent(gamma=env.DISCOUNT, precision=1e-4)
    result_dir_path = os.path.join(os.path.dirname(__file__), 'presult')
    # 运行策略迭代，并将可视化结果输出到 `presult` 目录
    agent.policy_iteration(env, constant_returned_cars=False, result_dir=result_dir_path, cmap="magma")

    print("状态最优价值：")
    print(agent.v)
    print("最优策略：")
    print(agent.policy)
