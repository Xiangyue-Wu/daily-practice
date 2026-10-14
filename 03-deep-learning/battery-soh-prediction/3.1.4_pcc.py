"""
步骤4：PCC特征选择（任务三）+ 特征裁剪
- 在每个划分的训练集上计算PCC
- 筛选 |PCC| > 阈值的特征
- 用选出的特征裁剪训练/验证/测试集
- 生成PCC分析图
输出：覆盖原来的 .npy 文件（变成筛选后的特征数）
"""
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib
import json
from pathlib import Path

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ====== 参数 ======
BATTERIES = ['B0005', 'B0006', 'B0007', 'B0018']
INTERMEDIATE_DIR = r'outputs/intermediate'
PREPARED_DATA_DIR = r'outputs/prepared_data'
PCC_THRESHOLD = 0.4
# ==================

Path(f'{PREPARED_DATA_DIR}/pcc_analysis').mkdir(parents=True, exist_ok=True)

# 加载特征名称
FEATURE_NAMES = list(np.load(f'{INTERMEDIATE_DIR}/feature_names.npy', allow_pickle=True))

# 收集所有训练集上的PCC结果
all_selected = {}  # 记录每个划分选了什么特征

# 处理每个电池的每种划分
for battery in BATTERIES:
    for split_name in ['split_A_random', 'split_B_sequential', 'split_C_transfer']:
        split_path = Path(f'{PREPARED_DATA_DIR}/{battery}/{split_name}')
        if not split_path.exists():
            continue

        print(f'{battery} - {split_name}')

        # 加载训练集
        train_X = np.load(split_path / 'train_X.npy')
        train_y = np.load(split_path / 'train_y.npy')

        # 在训练集上计算PCC
        pcc_values = []
        for i in range(train_X.shape[1]):
            try:
                corr, _ = pearsonr(train_X[:, i], train_y)
                pcc_values.append(corr)
            except:
                pcc_values.append(0)

        pcc_values = np.array(pcc_values)
        selected_idx = np.where(np.abs(pcc_values) > PCC_THRESHOLD)[0]

        # 如果选出的太少，至少保留3个
        if len(selected_idx) < 3:
            selected_idx = np.argsort(-np.abs(pcc_values))[:3]

        selected_idx = selected_idx[np.argsort(-np.abs(pcc_values[selected_idx]))]

        key = f'{battery}/{split_name}'
        all_selected[key] = {
            'selected_idx': selected_idx.tolist(),
            'pcc_values': np.abs(pcc_values).tolist(),
            'selected_names': [FEATURE_NAMES[i] for i in selected_idx],
            'n_features': len(selected_idx)
        }

        print(f'  选出 {len(selected_idx)} 个特征: {all_selected[key]["selected_names"]}')

        # 裁剪所有数据集
        for data_file in split_path.glob('*.npy'):
            data = np.load(data_file)
            if data.ndim == 2 and data.shape[1] == 20:
                np.save(data_file, data[:, selected_idx].astype(np.float32))

        # 更新 split_info.json
        info_path = split_path / 'split_info.json'
        if info_path.exists():
            with open(info_path, 'r') as f:
                info = json.load(f)
            info['feature_dim'] = len(selected_idx)
            info['selected_features'] = selected_idx.tolist()
            info['pcc_threshold'] = PCC_THRESHOLD
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2, default=int)

# ---- 画PCC热力图 ----
# 收集各电池在全量数据上的PCC
all_pcc = []
short_names = [
    'F1:I_var', 'F2:V_var', 'F3:V_med', 'F4:V_skew', 'F5:V_load_skew',
    'F6:slope_50-500', 'F7:slope_50-1000', 'F8:slope_50-1500',
    'F9:T_max', 'F10:T_mean', 'F11:T_var', 'F12:T_skew', 'F13:T_min',
    'F14:chg_T_mean', 'F15:chg_T_var', 'F16:chg_T_skew', 'F17:chg_T_min',
    'F18:CC_time', 'F19:CV_time', 'F20:disch_time'
]
for battery in BATTERIES:
    X = np.load(f'{INTERMEDIATE_DIR}/{battery}_X.npy')
    y = np.load(f'{INTERMEDIATE_DIR}/{battery}_y.npy')
    pcc = []
    for i in range(X.shape[1]):
        try:
            corr, _ = pearsonr(X[:, i], y)
            pcc.append(corr if not np.isnan(corr) else 0)
        except:
            pcc.append(0)
    all_pcc.append(pcc)
all_pcc = np.array(all_pcc)

fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(np.abs(all_pcc), cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xticks(range(20))
ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(4))
ax.set_yticklabels(BATTERIES, fontsize=10)
ax.set_xlabel('Feature', fontsize=12)
ax.set_title('|PCC| Heatmap: Feature-SOH Correlation Across 4 Batteries', fontsize=13)
for i in range(4):
    for j in range(20):
        val = np.abs(all_pcc[i, j])
        if val > 0.4:
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=6,
                    color='white' if val > 0.7 else 'black')
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('|PCC|', fontsize=11)
plt.tight_layout()
plt.savefig(f'{PREPARED_DATA_DIR}/pcc_analysis/pcc_heatmap.png', dpi=200)
plt.close()

# 保存选特征汇总
with open(f'{PREPARED_DATA_DIR}/pcc_analysis/selection_summary.json', 'w') as f:
    json.dump(all_selected, f, indent=2, default=int)

# 保存特征信息
feature_info = {
    'total_candidate_features': 20,
    'pcc_threshold': PCC_THRESHOLD,
    'note': 'PCC在训练集上计算，不同划分可能选出不同特征子集',
    'splits': {
        k: {'n_features': v['n_features'], 'features': v['selected_names']}
        for k, v in all_selected.items()
    }
}
with open(f'{PREPARED_DATA_DIR}/features_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2, default=int)

print(f'\n特征选择完成！PCC热力图已保存至 {PREPARED_DATA_DIR}/pcc_analysis/')
print(f'特征裁剪完成！所有划分现在使用训练集上选出的特征')

