"""
步骤2：提取特征 - 使用论文Table 1的20个特征
- 直接从step1处理好的数据读取放电和充电数据
- 无需重新读取原始.mat文件
- 放电和充电数据循环编号天然对齐
"""

import numpy as np
import pickle
from scipy.stats import skew
from pathlib import Path

BATTERIES = ['B0005', 'B0006', 'B0007', 'B0018']
INTERMEDIATE_DIR = r'outputs/intermediate'

# 论文Table 1的20个特征名称
FEATURE_NAMES = [
    'F1_放电电流方差',
    'F2_放电电压方差',
    'F3_放电电压中位数',
    'F4_放电电压偏度',
    'F5_放电负载电压偏度',
    'F6_放电电压斜率_50_500s',
    'F7_放电电压斜率_50_1000s',
    'F8_放电电压斜率_50_1500s',
    'F9_放电最高温度',
    'F10_放电平均温度',
    'F11_放电温度方差',
    'F12_放电温度偏度',
    'F13_放电最低温度',
    'F14_充电平均温度',
    'F15_充电温度方差',
    'F16_充电温度偏度',
    'F17_充电最低温度',
    'F18_CC充电时间',
    'F19_CV充电时间',
    'F20_总放电时间'
]


def extract_features_from_clean_data(battery_data):
    """
    直接从step1处理好的数据提取特征
    battery_data: 包含 'discharge' 和 'charge' 的字典
    """
    discharge_list = battery_data['discharge']
    charge_list = battery_data['charge']

    X = np.zeros((len(discharge_list), 20))

    for idx, d in enumerate(discharge_list):
        v = d['voltage']
        c = d['current']
        t = d['temperature']
        time_vec = d['time']

        # ========== F1-F5: 电流电压统计特征 ==========
        X[idx, 0] = np.var(c)                    # F1: 放电电流方差
        X[idx, 1] = np.var(v)                    # F2: 放电电压方差
        X[idx, 2] = np.median(v)                 # F3: 放电电压中位数
        X[idx, 3] = skew(v)                      # F4: 放电电压偏度
        X[idx, 4] = skew(v)                      # F5: 放电负载电压偏度（同F4）

        # ========== F6-F8: 放电电压斜率 ==========
        times = time_vec - time_vec[0]
        v_at = {}
        for t_target in [50, 500, 1000, 1500]:
            idx_t = np.argmin(np.abs(times - t_target))
            v_at[t_target] = v[idx_t]

        X[idx, 5] = (v_at[500] - v_at[50]) / (500 - 50)      # F6
        X[idx, 6] = (v_at[1000] - v_at[50]) / (1000 - 50)    # F7
        X[idx, 7] = (v_at[1500] - v_at[50]) / (1500 - 50)    # F8

        # ========== F9-F13: 放电温度特征 ==========
        X[idx, 8] = np.max(t)                    # F9: 放电最高温度
        X[idx, 9] = np.mean(t)                   # F10: 放电平均温度
        X[idx, 10] = np.var(t)                   # F11: 放电温度方差
        X[idx, 11] = skew(t)                     # F12: 放电温度偏度
        X[idx, 12] = np.min(t)                   # F13: 放电最低温度

        # ========== F14-F19: 充电阶段特征 ==========
        if idx < len(charge_list):
            ch = charge_list[idx]  # 循环编号自动对齐
            ch_v = ch['voltage']
            ch_c = ch['current']
            ch_t = ch['temperature']
            ch_time = ch['time']

            # 充电温度统计 (F14-F17)
            X[idx, 13] = np.mean(ch_t) if len(ch_t) > 0 else 0      # F14
            X[idx, 14] = np.var(ch_t) if len(ch_t) > 0 else 0       # F15
            X[idx, 15] = skew(ch_t) if len(ch_t) > 0 else 0         # F16
            X[idx, 16] = np.min(ch_t) if len(ch_t) > 0 else 0       # F17

            # 区分CC和CV阶段 (F18-F19)
            cc_mask = (np.abs(ch_c) > 0.1) & (ch_v < 4.19)
            cv_mask = (np.abs(ch_c) > 0.01) & (ch_v >= 4.19)

            cc_time = ch_time[cc_mask][-1] - ch_time[cc_mask][0] if cc_mask.any() else 0
            cv_time = ch_time[cv_mask][-1] - ch_time[cv_mask][0] if cv_mask.any() else 0

            X[idx, 17] = cc_time      # F18: CC充电时间
            X[idx, 18] = cv_time      # F19: CV充电时间

        # ========== F20: 时间特征 ==========
        X[idx, 19] = time_vec[-1] - time_vec[0]   # F20: 总放电时间

    return X


# ====== 主流程 ======
for battery in BATTERIES:
    print(f'提取特征 {battery}...')

    # 读取step1处理好的数据（包含放电和充电）
    with open(f'{INTERMEDIATE_DIR}/{battery}_clean.pkl', 'rb') as f:
        battery_data = pickle.load(f)

    discharge_list = battery_data['discharge']

    # 提取特征
    X = extract_features_from_clean_data(battery_data)

    # 计算SOH标签
    initial_capacity = discharge_list[0]['capacity']
    y = np.array([d['capacity'] / initial_capacity * 100 for d in discharge_list])

    # 保存
    np.save(f'{INTERMEDIATE_DIR}/{battery}_X.npy', X.astype(np.float32))
    np.save(f'{INTERMEDIATE_DIR}/{battery}_y.npy', y.astype(np.float32))

# 保存特征名称
np.save(f'{INTERMEDIATE_DIR}/feature_names.npy', np.array(FEATURE_NAMES))

print('步骤2完成！')