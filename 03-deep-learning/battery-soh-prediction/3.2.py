import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d  # 用于平滑约束


# -------------------------
# 1. 定义 LSTM 模型
# -------------------------
class SimpleLSTMModel(nn.Module):
    def __init__(self, input_features, hidden_size=32, num_layers=1):  # 减小容量
        super(SimpleLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=input_features, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=0.0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


# -------------------------
# 2. 创建序列（预测差分版本）
# -------------------------
def create_sequences_with_diff(data_X, data_y, window_size):
    """
    创建序列，目标是预测差分（变化量）
    预测目标：SOH_{t+1} - SOH_t
    """
    X_seq, y_seq = [], []
    for i in range(len(data_X) - window_size):
        # 输入序列
        X_seq.append(data_X[i:(i + window_size)])
        # 预测差分：下一个时刻的SOH减去当前时刻的SOH
        diff = data_y[i + window_size] - data_y[i + window_size - 1]
        y_seq.append(diff)
    return np.array(X_seq), np.array(y_seq)


def create_sequences_absolute(data_X, data_y, window_size):
    """
    创建序列，目标是预测绝对值（原始方法）
    """
    X_seq, y_seq = [], []
    for i in range(len(data_X) - window_size):
        X_seq.append(data_X[i:(i + window_size)])
        y_seq.append(data_y[i + window_size])
    return np.array(X_seq), np.array(y_seq)


# -------------------------
# 3. 物理约束函数
# -------------------------
def apply_physical_constraints(predictions_soh, original_predictions=None):
    """
    对预测的 SOH 绝对值应用物理约束

    参数:
        predictions_soh: 预测的 SOH 值（已反归一化）
        original_predictions: 原始预测值（用于参考，可选）

    返回:
        约束后的 SOH 值
    """
    predictions = predictions_soh.copy()

    # 约束1：范围约束 - SOH 应在 0-100 之间（或根据实际范围调整）
    predictions = np.clip(predictions, 50, 100)

    # 约束2：单调递减约束 - SOH 不能上升
    for i in range(1, len(predictions)):
        if predictions[i] > predictions[i - 1]:
            predictions[i] = predictions[i - 1]

    # 约束3：变化率约束 - 每循环下降不超过 1.5%（根据电池特性调整）
    max_drop_per_cycle = 1.5
    for i in range(1, len(predictions)):
        drop = predictions[i - 1] - predictions[i]
        if drop > max_drop_per_cycle:
            predictions[i] = predictions[i - 1] - max_drop_per_cycle
        # 也不能出现上升（上面已处理，这里再确保一下）
        if drop < -0.5:  # 允许微小上升（可能是测量误差），但不能超过0.5%
            predictions[i] = predictions[i - 1] + 0.5

    # 约束4：平滑约束 - 3点滑动平均（可选，减少波动）
    predictions = uniform_filter1d(predictions, size=3, mode='nearest')

    return predictions


# -------------------------
# 4. 差分预测的逆变换
# -------------------------
def inverse_diff_predictions(pred_diff, start_soh_values):
    """
    将预测的差分还原为 SOH 绝对值

    参数:
        pred_diff: 预测的差分值数组
        start_soh_values: 每个序列起始点的真实 SOH 值

    返回:
        还原后的 SOH 绝对值
    """
    soh_predictions = []
    for i, diff_seq in enumerate(pred_diff):
        start_soh = start_soh_values[i]
        soh_sequence = [start_soh]
        for diff in diff_seq:
            soh_sequence.append(soh_sequence[-1] + diff)
        # 只保留最后一个值（即预测的目标）
        soh_predictions.append(soh_sequence[-1])
    return np.array(soh_predictions)


# -------------------------
# 5. 主函数
# -------------------------
base_path = r'outputs/prepared_data'
batteries = ['B0005']
splits = ['split_A_random', 'split_B_sequential', 'split_C_transfer']

window_size = 5
num_epochs = 150  # 增加训练轮数
learning_rate = 0.001
hidden_size = 32  # 减小隐藏层
num_layers = 1  # 单层 LSTM
use_diff = True  # 【开关】是否使用差分预测

for battery in batteries:
    print(f"\n{'=' * 20} 正在处理电池: {battery} {'=' * 20}")
    for split in splits:
        split_path = os.path.join(base_path, battery, split)
        train_X_path = os.path.join(split_path, 'train_X.npy')

        if not os.path.exists(train_X_path):
            print(f"警告: {split_path} 不存在，跳过...")
            continue

        try:
            # ========== 加载数据 ==========
            train_X_2d = np.load(train_X_path)
            train_y_1d = np.load(os.path.join(split_path, 'train_y.npy')).flatten()
            test_X_2d = np.load(os.path.join(split_path, 'test_X.npy'))
            test_y_1d = np.load(os.path.join(split_path, 'test_y.npy')).flatten()

            print(f"\n{split}:")
            print(f"原始训练 y 范围: [{train_y_1d.min():.2f}, {train_y_1d.max():.2f}]")
            print(f"原始测试 y 范围: [{test_y_1d.min():.2f}, {test_y_1d.max():.2f}]")

            # ========== 归一化 ==========
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()

            train_X_scaled = scaler_X.fit_transform(train_X_2d)
            train_y_scaled = scaler_y.fit_transform(train_y_1d.reshape(-1, 1)).flatten()
            test_X_scaled = scaler_X.transform(test_X_2d)
            test_y_scaled = scaler_y.transform(test_y_1d.reshape(-1, 1)).flatten()

            # ========== 创建序列（根据 use_diff 选择方法）==========
            if use_diff:
                # 差分预测
                train_X_3d, train_target = create_sequences_with_diff(
                    train_X_scaled, train_y_scaled, window_size
                )
                test_X_3d, test_target = create_sequences_with_diff(
                    test_X_scaled, test_y_scaled, window_size
                )
                # 记录每个测试序列的起始 SOH（用于逆变换）
                test_start_soh_scaled = test_y_scaled[window_size - 1:-1]
                test_start_soh = scaler_y.inverse_transform(
                    test_start_soh_scaled.reshape(-1, 1)
                ).flatten()
                print(f"使用差分预测模式")
            else:
                # 绝对值预测
                train_X_3d, train_target = create_sequences_absolute(
                    train_X_scaled, train_y_scaled, window_size
                )
                test_X_3d, test_target = create_sequences_absolute(
                    test_X_scaled, test_y_scaled, window_size
                )
                test_start_soh = None
                print(f"使用绝对值预测模式")

            if len(train_X_3d) < 10 or len(test_X_3d) < 5:
                print(f"跳过: 序列样本不足")
                continue

            print(f"训练序列: {train_X_3d.shape}, 测试序列: {test_X_3d.shape}")

            # ========== 转换为 Tensor ==========
            X_train_tensor = torch.tensor(train_X_3d, dtype=torch.float32)
            y_train_tensor = torch.tensor(train_target, dtype=torch.float32).view(-1, 1)
            X_test_tensor = torch.tensor(test_X_3d, dtype=torch.float32)
            y_test_tensor = torch.tensor(test_target, dtype=torch.float32).view(-1, 1)

            # ========== 模型初始化 ==========
            input_features = train_X_3d.shape[2]
            model = SimpleLSTMModel(input_features=input_features,
                                    hidden_size=hidden_size,
                                    num_layers=num_layers)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

            # ========== 训练 ==========
            print(f"开始训练...")
            model.train()
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if (epoch + 1) % 50 == 0:
                    print(f"  Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.6f}")

            # ========== 测试 ==========
            model.eval()
            with torch.no_grad():
                predictions_scaled = model(X_test_tensor).numpy().flatten()
                actuals_scaled = y_test_tensor.numpy().flatten()

                if use_diff:
                    # 差分预测：需要逆变换
                    # 先将预测的差分反归一化
                    predictions_diff = scaler_y.inverse_transform(
                        predictions_scaled.reshape(-1, 1)
                    ).flatten()
                    # 还原为 SOH 绝对值
                    predictions = inverse_diff_predictions(
                        predictions_diff.reshape(-1, 1), test_start_soh
                    )
                    # 真实值也需要转换（测试集的 target 是差分，需要还原）
                    actuals_diff = scaler_y.inverse_transform(
                        actuals_scaled.reshape(-1, 1)
                    ).flatten()
                    actuals = inverse_diff_predictions(
                        actuals_diff.reshape(-1, 1), test_start_soh
                    )
                else:
                    # 绝对值预测：直接反归一化
                    predictions = scaler_y.inverse_transform(
                        predictions_scaled.reshape(-1, 1)
                    ).flatten()
                    actuals = scaler_y.inverse_transform(
                        actuals_scaled.reshape(-1, 1)
                    ).flatten()

                # ========== 应用物理约束 ==========（效果不好，删除）
                predictions_constrained = apply_physical_constraints(predictions)

                # 计算指标（原始预测）
                mae = mean_absolute_error(actuals, predictions)
                rmse = np.sqrt(mean_squared_error(actuals, predictions))
                r2 = r2_score(actuals, predictions)

                # 计算指标（约束后）
                mae_constrained = mean_absolute_error(actuals, predictions_constrained)
                rmse_constrained = np.sqrt(mean_squared_error(actuals, predictions_constrained))
                r2_constrained = r2_score(actuals, predictions_constrained)

                print(f"\n{'=' * 50}")
                print(f"划分方式: {split}")
                print(f"原始预测: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}")
                print(f"物理约束后: MAE={mae_constrained:.4f}, RMSE={rmse_constrained:.4f}, R²={r2_constrained:.4f}")
                print(f"{'=' * 50}")

                # ========== 绘图 ==========
                if split == 'split_B_sequential':
                    plt.figure(figsize=(14, 10))

                    # 图1：原始预测 vs 真实值
                    plt.subplot(2, 2, 1)
                    plt.plot(actuals, label='Real SOH', color='blue', linewidth=2)
                    plt.plot(predictions, label='Predicted (original)', color='red', linestyle='--', linewidth=2)
                    plt.title(f'{battery} - Split B (Original)\nR²={r2:.4f}')
                    plt.xlabel('Test Sample Index')
                    plt.ylabel('SOH')
                    plt.legend()
                    plt.grid(True)

                    # 图2：物理约束后 vs 真实值
                    plt.subplot(2, 2, 2)
                    plt.plot(actuals, label='Real SOH', color='blue', linewidth=2)
                    plt.plot(predictions_constrained, label='Predicted (constrained)', color='green', linestyle='--',
                             linewidth=2)
                    plt.title(f'{battery} - Split B (with Constraints)\nR²={r2_constrained:.4f}')
                    plt.xlabel('Test Sample Index')
                    plt.ylabel('SOH')
                    plt.legend()
                    plt.grid(True)

                    # 图3：残差图（原始）
                    plt.subplot(2, 2, 3)
                    residuals = actuals - predictions
                    plt.scatter(predictions, residuals, alpha=0.6)
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.xlabel('Predicted SOH')
                    plt.ylabel('Residuals')
                    plt.title('Residual Plot (Original)')
                    plt.grid(True)

                    # 图4：残差图（约束后）
                    plt.subplot(2, 2, 4)
                    residuals_constrained = actuals - predictions_constrained
                    plt.scatter(predictions_constrained, residuals_constrained, alpha=0.6, color='green')
                    plt.axhline(y=0, color='r', linestyle='--')
                    plt.xlabel('Predicted SOH (constrained)')
                    plt.ylabel('Residuals')
                    plt.title('Residual Plot (with Constraints)')
                    plt.grid(True)

                    plt.tight_layout()
                    plt.show()

        except Exception as e:
            print(f"处理 {battery} - {split} 时出错: {e}")
            import traceback

            traceback.print_exc()
            continue
