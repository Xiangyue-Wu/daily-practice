"""
步骤3：数据划分（任务二）
- 先划分，保留全部20个候选特征
- 划分A：随机 60%/20%/20%
- 划分B：前60%循环训练，后40%测试
- 划分C：跨电池迁移学习
输出：prepared_data/{电池名}/split_*/  (此时还是20个特征)
"""
import numpy as np
import json
from pathlib import Path

# ====== 参数 ======
BATTERIES = ['B0005', 'B0006', 'B0007', 'B0018']
INTERMEDIATE_DIR = r'outputs/intermediate'
PREPARED_DATA_DIR = r'outputs/prepared_data'
RANDOM_SEED = 42
SPLIT_A_TRAIN = 0.6
SPLIT_A_VAL = 0.2
SPLIT_B_TRAIN = 0.6
SPLIT_B_VAL_RATIO = 0.2
SPLIT_C_FINETUNE = 0.1
# ==================

np.random.seed(RANDOM_SEED)


def save_split(save_dir, split_dict):
    """保存一份划分，此时还是20个特征"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for key, arr in split_dict.items():
        if key == 'info':
            with open(f'{save_dir}/split_info.json', 'w') as f:
                json.dump(arr, f, indent=2, default=int)
        else:
            np.save(f'{save_dir}/{key}.npy', arr.astype(np.float32))


def split_random(X, y, cycles):
    """划分A：随机60/20/20"""
    n = len(y)
    idx = np.random.permutation(n)
    train_end = int(n * SPLIT_A_TRAIN)
    val_end = int(n * (SPLIT_A_TRAIN + SPLIT_A_VAL))

    return {
        'train_X': X[idx[:train_end]], 'train_y': y[idx[:train_end]],
        'val_X': X[idx[train_end:val_end]], 'val_y': y[idx[train_end:val_end]],
        'test_X': X[idx[val_end:]], 'test_y': y[idx[val_end:]],
        'info': {
            'method': 'random', 'train_ratio': SPLIT_A_TRAIN,
            'val_ratio': SPLIT_A_VAL, 'random_seed': RANDOM_SEED,
            'train_cycles': cycles[idx[:train_end]].tolist(),
            'val_cycles': cycles[idx[train_end:val_end]].tolist(),
            'test_cycles': cycles[idx[val_end:]].tolist()
        }
    }


def split_sequential(X, y, cycles):
    """划分B：前60%训练，后40%测试（无验证集）"""
    n = len(y)
    train_end = int(n * 0.6)  # 严格按 60% 切分

    return {
        'train_X': X[:train_end], 'train_y': y[:train_end],
        'test_X': X[train_end:], 'test_y': y[train_end:],
        'info': {
            'method': 'sequential',
            'train_ratio': 0.6,
            'test_ratio': 0.4,
            'train_cycles': cycles[:train_end].tolist(),
            'test_cycles': cycles[train_end:].tolist()
        }
    }

def split_transfer(source_battery, other_batteries):
    """划分C：1个源电池全量 + 其余3个电池各前10%训练，其余3个电池各后90%测试"""
    # 1. 源电池全量作为训练集基础
    X_src = np.load(f'{INTERMEDIATE_DIR}/{source_battery}_X.npy')
    y_src = np.load(f'{INTERMEDIATE_DIR}/{source_battery}_y.npy')

    X_train_list, y_train_list = [X_src], [y_src]
    X_test_list, y_test_list = [], []

    # 2. 其余3个电池按 10% / 90% 划分
    for bat in other_batteries:
        X_bat = np.load(f'{INTERMEDIATE_DIR}/{bat}_X.npy')
        y_bat = np.load(f'{INTERMEDIATE_DIR}/{bat}_y.npy')
        split_idx = int(len(y_bat) * SPLIT_C_FINETUNE)

        X_train_list.append(X_bat[:split_idx])
        y_train_list.append(y_bat[:split_idx])

        X_test_list.append(X_bat[split_idx:])
        y_test_list.append(y_bat[split_idx:])

    # 3. 合并
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    return {
        'train_X': X_train, 'train_y': y_train,
        'test_X': X_test, 'test_y': y_test,
        'info': {
            'method': 'transfer_learning',
            'source_battery': source_battery,
            'target_batteries': other_batteries,
            'finetune_ratio': SPLIT_C_FINETUNE,
            'source_samples': len(y_src),
            'target_finetune_samples': len(y_train) - len(y_src),
            'target_test_samples': len(y_test),
        }
    }


# ====== 主流程 ======
# 加载所有电池的20个候选特征
data = {}
for battery in BATTERIES:
    X = np.load(f'{INTERMEDIATE_DIR}/{battery}_X.npy')  # 20个特征
    y = np.load(f'{INTERMEDIATE_DIR}/{battery}_y.npy')
    cycles = np.arange(1, len(y) + 1)
    data[battery] = (X, y, cycles)

# 划分A和B：每个电池单独做
for battery in BATTERIES:
    print(f'划分 {battery}（20个候选特征）...')
    X, y, cycles = data[battery]
    base = f'{PREPARED_DATA_DIR}/{battery}'

    save_split(f'{base}/split_A_random', split_random(X, y, cycles))
    save_split(f'{base}/split_B_sequential', split_sequential(X, y, cycles))

# 划分C：4种情况，每次选1个作为全量源，其余3个按10/90划分
for source in BATTERIES:
    others = [b for b in BATTERIES if b != source]
    print(f'划分C: {source}(全量) + {others}(前10%) -> {others}(后90%)')
    base = f'{PREPARED_DATA_DIR}/{source}'
    save_split(f'{base}/split_C_transfer', split_transfer(source, others))

print(f'\n数据划分完成！所有文件已保存至 {PREPARED_DATA_DIR}/')
print('此时所有划分都包含完整的20个候选特征')