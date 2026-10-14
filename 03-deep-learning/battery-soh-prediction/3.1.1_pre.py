"""
步骤1：数据预处理
- 读取.mat文件，提取放电循环和充电循环
- 计算每个循环的容量和SOH
- 剔除异常循环（同时剔除放电和充电中对应的异常循环）
输出：intermediate/{电池名}_clean.pkl（包含放电和充电数据）
"""

import numpy as np
import scipy.io as sio
import pickle
from pathlib import Path
import matplotlib.pyplot as plt

# ====== 参数 ======
BATTERIES = ['B0005', 'B0006', 'B0007', 'B0018']
RAW_DATA_DIR = r'data/raw_data'
INTERMEDIATE_DIR = r'outputs/intermediate'
OUTLIER_SIGMA = 3.0
# ==================

Path(INTERMEDIATE_DIR).mkdir(exist_ok=True)


def load_mat(battery_name):
    """读取.mat文件，返回所有循环数据"""
    filepath = f'{RAW_DATA_DIR}/{battery_name}.mat'
    data = sio.loadmat(filepath)
    key = [k for k in data.keys() if not k.startswith('__')][-1]
    return data[key][0, 0]['cycle'][0]


def extract_cycles(cycles):
    """提取所有放电循环和充电循环，保持循环编号对齐"""
    discharge_list = []
    charge_list = []

    for i, cycle in enumerate(cycles):
        # 获取工况类型
        cycle_type = cycle['type'][0] if isinstance(cycle['type'], np.ndarray) else str(cycle['type'])
        cycle_type = cycle_type.lower()

        # 获取 data 结构体
        cycle_data = cycle['data'][0, 0]

        # 根据工况类型分别处理
        if 'discharge' in cycle_type:
            # 放电工况：读取电压、电流、温度、时间
            voltage = cycle_data['Voltage_measured'][0].flatten().astype(np.float64)
            current = cycle_data['Current_measured'][0].flatten().astype(np.float64)
            temp = cycle_data['Temperature_measured'][0].flatten().astype(np.float64)
            time_vec = cycle_data['Time'][0].flatten().astype(np.float64)

            # 直接读取官方 Capacity
            cap = cycle_data['Capacity'][0, 0]

            discharge_list.append({
                'cycle': i + 1,
                'voltage': voltage,
                'current': current,
                'temperature': temp,
                'time': time_vec,
                'capacity': cap
            })

        elif 'charge' in cycle_type:
            # 充电工况：读取电压、电流、温度、时间
            voltage = cycle_data['Voltage_measured'][0].flatten().astype(np.float64)
            current = cycle_data['Current_measured'][0].flatten().astype(np.float64)
            temp = cycle_data['Temperature_measured'][0].flatten().astype(np.float64)
            time_vec = cycle_data['Time'][0].flatten().astype(np.float64)

            charge_list.append({
                'cycle': i + 1,
                'voltage': voltage,
                'current': current,
                'temperature': temp,
                'time': time_vec
            })

        else:
            # 阻抗工况：跳过
            continue

    print(f'  找到 {len(discharge_list)} 个放电循环, {len(charge_list)} 个充电循环')

    # 对齐：充电和放电循环数量应该一致，如果不一致则截取
    min_len = min(len(discharge_list), len(charge_list))
    if len(discharge_list) != len(charge_list):
        print(f'  警告：循环数量不一致，将截取前{min_len}个')
        discharge_list = discharge_list[:min_len]
        charge_list = charge_list[:min_len]

    return discharge_list, charge_list


def remove_outliers(discharge_list, charge_list):
    """剔除异常循环（同时剔除放电和充电中对应的异常循环）"""
    if len(discharge_list) == 0:
        return discharge_list, charge_list

    capacities = np.array([d['capacity'] for d in discharge_list])
    x = np.arange(len(capacities))

    # 用直线拟合趋势
    if len(capacities) >= 2:
        trend = np.polyval(np.polyfit(x, capacities, 1), x)
    else:
        return discharge_list, charge_list

    residuals = capacities - trend
    std = np.std(residuals)
    if std == 0:
        return discharge_list, charge_list

    # 找出有效索引
    valid_idx = np.where(np.abs(residuals) < OUTLIER_SIGMA * std)[0]

    # 同时剔除放电和充电中的异常循环
    clean_discharge = [discharge_list[i] for i in valid_idx]
    clean_charge = [charge_list[i] for i in valid_idx]

    print(f'  剔除 {len(discharge_list) - len(clean_discharge)} 个异常循环')
    return clean_discharge, clean_charge


def plot_soh(battery_name, clean_data):
    """画SOH衰退曲线"""
    if len(clean_data) == 0:
        return

    cycles = [d['cycle'] for d in clean_data]
    soh = [d['capacity'] / clean_data[0]['capacity'] * 100 for d in clean_data]

    plt.figure(figsize=(10, 5))
    plt.plot(cycles, soh, 'o-', markersize=3)
    plt.xlabel('Cycle')
    plt.ylabel('SOH (%)')
    plt.title(f'{battery_name} SOH Degradation')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'{INTERMEDIATE_DIR}/{battery_name}_soh.png', dpi=150)
    plt.close()
    print(f'  初始容量: {clean_data[0]["capacity"]:.3f}Ah, '
          f'最终SOH: {soh[-1]:.2f}%, 总循环数: {len(clean_data)}')


# ====== 主流程 ======
for battery in BATTERIES:
    print(f'处理 {battery}...')
    cycles = load_mat(battery)
    discharge_list, charge_list = extract_cycles(cycles)
    clean_discharge, clean_charge = remove_outliers(discharge_list, charge_list)
    plot_soh(battery, clean_discharge)

    # 保存放电和充电数据
    battery_data = {
        'discharge': clean_discharge,
        'charge': clean_charge
    }

    with open(f'{INTERMEDIATE_DIR}/{battery}_clean.pkl', 'wb') as f:
        pickle.dump(battery_data, f)

print('步骤1完成！')


import pickle

with open('models/B0005_clean.pkl', 'rb') as f:
    data = pickle.load(f)

print('放电循环编号:', [d['cycle'] for d in data['discharge'][:5]])
print('充电循环编号:', [d['cycle'] for d in data['charge'][:5]])