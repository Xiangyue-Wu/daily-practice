import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PREDATA_DIR = os.path.join(BASE_DIR, 'outputs', 'prepared_data')
OUTPUT_DIR = os.path.join(BASE_DIR, 're-3-try7')
os.makedirs(OUTPUT_DIR, exist_ok=True)

BATTERIES = ['B0005', 'B0006', 'B0007', 'B0018']
SPLIT_TYPES = {
    'A': 'split_A_random',
    'B': 'split_B_sequential',
    'C': 'split_C_transfer'
}
GROUP_NAMES = {
    'A': 'Group1_Random',
    'B': 'Group2_Sequential',
    'C': 'Group3_Transfer'
}

LOOK_BACK_A = 1
LOOK_BACK_B = 5
LOOK_BACK_C = 1

BATCH_SIZE = 32
EPOCHS_A = 600
EPOCHS_B = 500
EPOCHS_C_PRETRAIN = 800
EPOCHS_C_TRANSFER = 600
LR = 0.001
LR_PRETRAIN = 0.0005
LR_TRANSFER = 0.0005

LSTM_C_SMALL = 64
DENSE_C_SMALL = 64
CONV_C_SMALL = 64

LSTM_STD = 96
DENSE_STD = 96
CONV_STD = 96
CONV_KERNEL = 3
DROPOUT_STD = 0.2
DROPOUT_TRANSFER = 0.3

SOH_MAX = 100.0


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()


def create_sequences(X, y, look_back):
    if look_back <= 1:
        return X.reshape(X.shape[0], 1, X.shape[1]), y.reshape(-1, 1)
    if len(X) <= look_back:
        return np.array([]), np.array([])
    dataX, dataY = [], []
    for i in range(len(X) - look_back):
        dataX.append(X[i:(i + look_back), :])
        dataY.append(y[i + look_back])
    return np.array(dataX), np.array(dataY).reshape(-1, 1)


def create_sequences_delta(X, y, look_back):
    if len(X) <= look_back:
        return np.array([]), np.array([]), None
    dataX, dataY = [], []
    for i in range(len(X) - look_back):
        dataX.append(X[i:(i + look_back), :])
        dataY.append(y[i + look_back] - y[i + look_back - 1])
    prev_y = y[look_back - 1:-1].reshape(-1, 1)
    return np.array(dataX), np.array(dataY).reshape(-1, 1), prev_y


def load_split_data(battery, split_type):
    data_dir = os.path.join(PREDATA_DIR, battery, split_type)
    train_X = np.load(os.path.join(data_dir, 'train_X.npy'))
    train_y = np.load(os.path.join(data_dir, 'train_y.npy')).astype(np.float32) / SOH_MAX
    test_X = np.load(os.path.join(data_dir, 'test_X.npy'))
    test_y = np.load(os.path.join(data_dir, 'test_y.npy')).astype(np.float32) / SOH_MAX
    val_X, val_y = None, None
    val_path = os.path.join(data_dir, 'val_X.npy')
    if os.path.exists(val_path):
        val_X = np.load(val_path)
        val_y = np.load(os.path.join(data_dir, 'val_y.npy')).astype(np.float32) / SOH_MAX
    return train_X, train_y, val_X, val_y, test_X, test_y


def normalize_x(train_X, val_X, test_X):
    x_scaler = StandardScaler()
    train_X_norm = x_scaler.fit_transform(train_X)
    test_X_norm = x_scaler.transform(test_X) if test_X is not None else None
    val_X_norm = x_scaler.transform(val_X) if val_X is not None else None
    return train_X_norm, val_X_norm, test_X_norm, x_scaler


class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        a = self.dense(x)
        a = F.softmax(a, dim=-1)
        return x * a


class CNNBiLSTMAttention(nn.Module):
    def __init__(self, input_dim, lstm_units=96, dense1=96, conv_filters=96,
                 conv_kernel=3, dropout=0.2):
        super().__init__()
        conv_padding = conv_kernel // 2
        self.conv1d = nn.Conv1d(in_channels=input_dim, out_channels=conv_filters,
                                kernel_size=conv_kernel, padding=conv_padding)
        self.bn1 = nn.BatchNorm1d(conv_filters)
        self.dropout1 = nn.Dropout(dropout)
        lstm_dropout = dropout if lstm_units >= 64 else 0.0
        self.bilstm = nn.LSTM(input_size=conv_filters, hidden_size=lstm_units,
                              num_layers=2, bidirectional=True,
                              batch_first=True, dropout=lstm_dropout)
        self.attention = AttentionBlock(lstm_units * 2)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units * 2, dense1)
        self.bn2 = nn.BatchNorm1d(dense1)
        self.fc2 = nn.Linear(dense1, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.dropout1(F.relu(self.bn1(self.conv1d(x))))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.bilstm(x)
        attn_out = self.attention(lstm_out)
        attn_out = attn_out[:, -1, :]
        attn_out = self.dropout2(attn_out)
        out = F.relu(self.bn2(self.fc1(attn_out)))
        out = self.fc2(out)
        return out

    def freeze_cnn(self, freeze=True):
        for param in self.conv1d.parameters():
            param.requires_grad = not freeze
        for param in self.bn1.parameters():
            param.requires_grad = not freeze

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


def train_model(model, train_loader, val_loader, epochs, lr, model_name,
                weight_decay=0.0, step_size=150, use_cosine=False):
    model = model.to(device)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    if use_cosine:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=epochs//4, T_mult=1, eta_min=lr*0.01)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.5)
    criterion = nn.MSELoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * batch_X.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    loss = criterion(outputs.squeeze(), batch_y.squeeze())
                    val_loss += loss.item() * batch_X.size(0)
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

        log_interval = max(1, epochs // 10)
        if (epoch + 1) % log_interval == 0 or epoch == 0:
            v_str = f', Val Loss: {val_losses[-1]:.6f}' if val_losses else ''
            print(f'  [{model_name}] Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}{v_str}')

    if best_state is not None:
        model.load_state_dict(best_state)
        print(f'  [{model_name}] Best loss: {best_val_loss:.6f}')
    return model, train_losses, val_losses


def evaluate_model(model, data_loader):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())
    preds = np.concatenate(all_preds).ravel() * SOH_MAX
    targets = np.concatenate(all_targets).ravel() * SOH_MAX
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    return mae, rmse, r2, preds, targets


def evaluate_model_delta(model, data_loader, prev_y_np):
    model.eval()
    all_deltas, all_targets = [], []
    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            batch_X = batch_X.to(device)
            outputs = model(batch_X)
            all_deltas.append(outputs.cpu().numpy())
            all_targets.append(batch_y.numpy())
    pred_deltas = np.concatenate(all_deltas).ravel()
    targets_abs = np.concatenate(all_targets).ravel() + prev_y_np.ravel()
    preds_abs = pred_deltas + prev_y_np.ravel()
    preds = preds_abs * SOH_MAX
    targets = targets_abs * SOH_MAX
    mae = mean_absolute_error(targets, preds)
    rmse = np.sqrt(mean_squared_error(targets, preds))
    r2 = r2_score(targets, preds)
    return mae, rmse, r2, preds, targets


def prepare_loaders(train_X, train_y, val_X, val_y, test_X, test_y,
                    look_back, batch_size=32):
    train_seq_X, train_seq_y = create_sequences(train_X, train_y, look_back)
    test_seq_X, test_seq_y = create_sequences(test_X, test_y, look_back)
    val_seq_X, val_seq_y = None, None
    if val_X is not None and val_y is not None:
        val_seq_X, val_seq_y = create_sequences(val_X, val_y, look_back)
    if len(train_seq_X) == 0:
        return None, None, None, None
    train_dataset = TensorDataset(torch.FloatTensor(train_seq_X),
                                   torch.FloatTensor(train_seq_y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = None
    if val_seq_X is not None and len(val_seq_X) > 0:
        val_dataset = TensorDataset(torch.FloatTensor(val_seq_X),
                                     torch.FloatTensor(val_seq_y))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataset = TensorDataset(torch.FloatTensor(test_seq_X),
                                  torch.FloatTensor(test_seq_y))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_X.shape[1]


def prepare_loaders_delta(train_X, train_y, val_X, val_y, test_X, test_y,
                          look_back, batch_size=32):
    train_seq_X, train_seq_y, train_prev_y = create_sequences_delta(
        train_X, train_y, look_back)
    test_seq_X, test_seq_y, test_prev_y = create_sequences_delta(
        test_X, test_y, look_back)
    val_loader = None
    if val_X is not None and val_y is not None:
        val_seq_X, val_seq_y, _ = create_sequences_delta(val_X, val_y, look_back)
        val_dataset = TensorDataset(torch.FloatTensor(val_seq_X),
                                     torch.FloatTensor(val_seq_y))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    if len(train_seq_X) == 0:
        return None, None, None, None, None
    train_dataset = TensorDataset(torch.FloatTensor(train_seq_X),
                                   torch.FloatTensor(train_seq_y))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = TensorDataset(torch.FloatTensor(test_seq_X),
                                  torch.FloatTensor(test_seq_y))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, train_X.shape[1], test_prev_y


def run_group_a():
    print('\n' + '=' * 60)
    print('  Group 1: Split A - Random Split')
    print(f'  LB={LOOK_BACK_A}, Ep={EPOCHS_A}')
    print('=' * 60)
    results, all_preds_targets = {}, {}
    for battery in BATTERIES:
        print(f'\n--- Battery: {battery} ---')
        train_X, train_y, val_X, val_y, test_X, test_y = load_split_data(battery, SPLIT_TYPES['A'])
        train_X_n, val_X_n, test_X_n, _ = normalize_x(train_X, val_X, test_X)
        train_loader, val_loader, test_loader, input_dim = prepare_loaders(
            train_X_n, train_y, val_X_n, val_y, test_X_n, test_y, LOOK_BACK_A, BATCH_SIZE)
        if train_loader is None:
            continue
        print(f'  dim={input_dim}, Train={len(train_loader.dataset)}, Test={len(test_loader.dataset)}')
        model = CNNBiLSTMAttention(input_dim=input_dim, lstm_units=LSTM_STD, dense1=DENSE_STD,
                                    conv_filters=CONV_STD, conv_kernel=CONV_KERNEL, dropout=DROPOUT_STD)
        model, tl, vl = train_model(model, train_loader, val_loader, EPOCHS_A, LR, f'{battery}_A')
        mae, rmse, r2, preds, targets = evaluate_model(model, test_loader)
        print(f'  MAE={mae:.4f}%, RMSE={rmse:.4f}%, R2={r2:.4f}')
        results[battery] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        all_preds_targets[battery] = (preds, targets, tl, vl)
    return results, all_preds_targets


def run_group_b():
    print('\n' + '=' * 60)
    print('  Group 2: Split B - Sequential (Delta-SOH)')
    print(f'  LB={LOOK_BACK_B}, Ep={EPOCHS_B}')
    print('=' * 60)
    results, all_preds_targets = {}, {}
    for battery in BATTERIES:
        print(f'\n--- Battery: {battery} ---')
        train_X, train_y, _, _, test_X, test_y = load_split_data(battery, SPLIT_TYPES['B'])
        train_X_n, _, test_X_n, _ = normalize_x(train_X, None, test_X)
        split_idx = int(len(train_X_n) * 0.85)
        tr_X, tr_y = train_X_n[:split_idx], train_y[:split_idx]
        v_X, v_y = train_X_n[split_idx:], train_y[split_idx:]
        ret = prepare_loaders_delta(tr_X, tr_y, v_X, v_y, test_X_n, test_y, LOOK_BACK_B, BATCH_SIZE)
        if ret[0] is None:
            continue
        train_loader, val_loader, test_loader, input_dim, test_prev_y = ret
        print(f'  dim={input_dim}, Train={len(train_loader.dataset)}, Test={len(test_loader.dataset)}')
        model = CNNBiLSTMAttention(input_dim=input_dim, lstm_units=LSTM_STD, dense1=DENSE_STD,
                                    conv_filters=CONV_STD, conv_kernel=CONV_KERNEL, dropout=DROPOUT_STD)
        model, tl, vl = train_model(model, train_loader, val_loader, EPOCHS_B, LR*0.5, f'{battery}_B')
        mae, rmse, r2, preds, targets = evaluate_model_delta(model, test_loader, test_prev_y)
        print(f'  MAE={mae:.4f}%, RMSE={rmse:.4f}%, R2={r2:.4f}')
        results[battery] = {'mae': mae, 'rmse': rmse, 'r2': r2}
        all_preds_targets[battery] = (preds, targets, tl, vl)
    return results, all_preds_targets


def run_group_c():
    print('\n' + '=' * 60)
    print('  Group 3: Split C - Transfer (try7: smaller model B0005, B0007 transfer)')
    print(f'  B0005: model 64-64-64 + CosineAnnealingWarmRestarts, {EPOCHS_C_PRETRAIN}ep')
    print(f'  B0006: B0005->transfer, {EPOCHS_C_TRANSFER}ep')
    print(f'  B0007: B0005->transfer, {EPOCHS_C_TRANSFER}ep')
    print(f'  B0018: independent, {EPOCHS_C_TRANSFER}ep')
    print('=' * 60)

    results, all_preds_targets = {}, {}

    print('\n--- Step 1: Pretrain B0005 (small model + CosineAnnealing) ---')
    train_X_s, train_y_s, _, _, test_X_s, test_y_s = load_split_data('B0005', SPLIT_TYPES['C'])
    train_X_s_n, _, test_X_s_n, _ = normalize_x(train_X_s, None, test_X_s)
    split_s = int(len(train_X_s_n) * 0.85)
    tr_X_s, tr_y_s = train_X_s_n[:split_s], train_y_s[:split_s]
    v_X_s, v_y_s = train_X_s_n[split_s:], train_y_s[split_s:]
    train_ld_s, val_ld_s, test_ld_s, dim_s = prepare_loaders(
        tr_X_s, tr_y_s, v_X_s, v_y_s, test_X_s_n, test_y_s, LOOK_BACK_C, BATCH_SIZE)

    base_model = CNNBiLSTMAttention(input_dim=dim_s, lstm_units=LSTM_C_SMALL,
                                     dense1=DENSE_C_SMALL, conv_filters=CONV_C_SMALL,
                                     conv_kernel=CONV_KERNEL, dropout=DROPOUT_TRANSFER)
    base_model, src_tl, src_vl = train_model(
        base_model, train_ld_s, val_ld_s, EPOCHS_C_PRETRAIN, LR_PRETRAIN,
        'B0005_pretrain', weight_decay=1e-4, use_cosine=True)

    base_state = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    print('\n--- Step 2: Eval B0005 ---')
    mae, rmse, r2, preds_s, targets_s = evaluate_model(base_model, test_ld_s)
    print(f'  B0005: MAE={mae:.4f}%, RMSE={rmse:.4f}%, R2={r2:.4f}')
    results['B0005'] = {'mae': mae, 'rmse': rmse, 'r2': r2}
    all_preds_targets['B0005'] = (preds_s, targets_s, src_tl, src_vl)

    for tgt, use_transfer in [('B0006', True), ('B0007', True), ('B0018', False)]:
        print(f'\n--- {"Transfer" if use_transfer else "Independent"} {tgt} ---')
        train_X_t, train_y_t, _, _, test_X_t, test_y_t = load_split_data(tgt, SPLIT_TYPES['C'])
        train_X_t_n, _, test_X_t_n, _ = normalize_x(train_X_t, None, test_X_t)
        n_val = max(len(train_X_t_n)//5, 1)
        tr_X_t, tr_y_t = train_X_t_n[:-n_val], train_y_t[:-n_val]
        v_X_t, v_y_t = train_X_t_n[-n_val:], train_y_t[-n_val:]
        train_ld_t, val_ld_t, test_ld_t, dim_t = prepare_loaders(
            tr_X_t, tr_y_t, v_X_t, v_y_t, test_X_t_n, test_y_t, LOOK_BACK_C, BATCH_SIZE)

        if use_transfer and dim_t == dim_s:
            model_t = CNNBiLSTMAttention(input_dim=dim_t, lstm_units=LSTM_C_SMALL,
                                          dense1=DENSE_C_SMALL, conv_filters=CONV_C_SMALL,
                                          conv_kernel=CONV_KERNEL, dropout=DROPOUT_TRANSFER)
            model_t.load_state_dict(base_state)
            print(f'  {tgt}: dim={dim_t}, Finetune={len(train_ld_t.dataset)}, Test={len(test_ld_t.dataset)}')
            half = EPOCHS_C_TRANSFER // 2
            print(f'  Phase 2a: Freeze CNN ({half}ep)')
            model_t.freeze_cnn(True)
            model_t, tl_a, vl_a = train_model(model_t, train_ld_t, val_ld_t, half,
                                               LR_TRANSFER, f'{tgt}_C_a', weight_decay=0.0, step_size=100)
            print(f'  Phase 2b: Unfreeze all ({half}ep)')
            model_t.unfreeze_all()
            model_t, tl_b, vl_b = train_model(model_t, train_ld_t, val_ld_t, half,
                                               LR_TRANSFER*0.2, f'{tgt}_C_b', weight_decay=1e-5, step_size=100)
            mae, rmse, r2, preds_t, targets_t = evaluate_model(model_t, test_ld_t)
            print(f'  {tgt}: MAE={mae:.4f}%, RMSE={rmse:.4f}%, R2={r2:.4f}')
            results[tgt] = {'mae': mae, 'rmse': rmse, 'r2': r2}
            all_preds_targets[tgt] = (preds_t, targets_t, tl_a+tl_b, vl_a+vl_b)
        else:
            model_t = CNNBiLSTMAttention(input_dim=dim_t, lstm_units=LSTM_STD,
                                          dense1=DENSE_STD, conv_filters=CONV_STD,
                                          conv_kernel=CONV_KERNEL, dropout=DROPOUT_TRANSFER)
            print(f'  {tgt}: dim={dim_t}, Train={len(train_ld_t.dataset)}, Test={len(test_ld_t.dataset)}')
            model_t, tl, vl = train_model(model_t, train_ld_t, val_ld_t, EPOCHS_C_TRANSFER,
                                           LR_PRETRAIN, f'{tgt}_C_indep', weight_decay=1e-4, step_size=150)
            mae, rmse, r2, preds_t, targets_t = evaluate_model(model_t, test_ld_t)
            print(f'  {tgt}: MAE={mae:.4f}%, RMSE={rmse:.4f}%, R2={r2:.4f}')
            results[tgt] = {'mae': mae, 'rmse': rmse, 'r2': r2}
            all_preds_targets[tgt] = (preds_t, targets_t, tl, vl)

    return results, all_preds_targets


def plot_prediction_scatter(all_preds_targets, group_name, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, battery in enumerate(BATTERIES):
        if battery not in all_preds_targets:
            continue
        preds, targets, _, _ = all_preds_targets[battery]
        ax = axes[idx]
        ax.scatter(targets, preds, alpha=0.6, s=20, c='steelblue', edgecolors='white')
        min_v, max_v = min(targets.min(), preds.min()), max(targets.max(), preds.max())
        ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5, label='y=x')
        mae = mean_absolute_error(targets, preds)
        rmse = np.sqrt(mean_squared_error(targets, preds))
        r2 = r2_score(targets, preds)
        ax.set_xlabel('True SOH (%)'); ax.set_ylabel('Predicted SOH (%)')
        ax.set_title(f'{battery}', fontweight='bold')
        ax.text(0.05, 0.95, f'MAE={mae:.2f}%\nRMSE={rmse:.2f}%\nR2={r2:.4f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.legend(loc='lower right'); ax.grid(True, alpha=0.3)
    fig.suptitle(f'{group_name} - Scatter', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{group_name}_scatter.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_curve(all_preds_targets, group_name, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, battery in enumerate(BATTERIES):
        if battery not in all_preds_targets:
            continue
        preds, targets, _, _ = all_preds_targets[battery]
        ax = axes[idx]
        n = min(len(targets), 500)
        srt = np.argsort(targets[:n])
        ax.plot(np.arange(n), targets[:n][srt], 'b-', linewidth=1.5, label='True', alpha=0.8)
        ax.plot(np.arange(n), preds[:n][srt], 'r--', linewidth=1.5, label='Pred', alpha=0.8)
        ax.set_title(f'{battery} (MAE={mean_absolute_error(targets, preds):.2f}%)', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle(f'{group_name} - Curve', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{group_name}_curve.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_loss_curves(all_preds_targets, group_name, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    for idx, battery in enumerate(BATTERIES):
        if battery not in all_preds_targets:
            continue
        _, _, tl, vl = all_preds_targets[battery]
        ax = axes[idx]
        ax.plot(tl, 'b-', linewidth=1.5, label='Train', alpha=0.8)
        if vl:
            ax.plot(vl, 'r-', linewidth=1.5, label='Val', alpha=0.8)
        ax.set_title(f'{battery}', fontweight='bold')
        ax.legend(); ax.grid(True, alpha=0.3)
        if tl and tl[-1] > 1e-10:
            ax.set_yscale('log')
    fig.suptitle(f'{group_name} - Loss', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{group_name}_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_outputs(all_group_results, all_group_preds):
    lines = []
    lines.append('=' * 90)
    lines.append('    CNN-BiLSTM-Attention SOH Prediction - Comparison Table (3-try7)')
    lines.append('=' * 90)
    for group_name, results in all_group_results.items():
        lines.append(f'  {group_name}')
        lines.append('-' * 90)
        lines.append(f'  {"Battery":<10} {"MAE(%)":>10} {"RMSE(%)":>10} {"R2":>10}  Rating')
        lines.append('  ' + '-' * 52)
        for battery in BATTERIES:
            if battery in results:
                r = results[battery]
                if r['mae'] < 1.0 and r['rmse'] < 1.5 and r['r2'] > 0.95:
                    rating = 'Excellent'
                elif r['mae'] < 3.0 and r['rmse'] < 4.0 and r['r2'] > 0.85:
                    rating = 'Good'
                else:
                    rating = 'Poor'
                lines.append(f'  {battery:<10} {r["mae"]:>10.4f} {r["rmse"]:>10.4f} {r["r2"]:>10.4f}  {rating}')
        vals = list(results.values())
        lines.append('  ' + '-' * 52)
        lines.append(f'  {"Average":<10} {np.mean([r["mae"] for r in vals]):>10.4f} '
                     f'{np.mean([r["rmse"] for r in vals]):>10.4f} '
                     f'{np.mean([r["r2"] for r in vals]):>10.4f}')
        lines.append('')
    lines.append('=' * 90)
    lines.append('  Criteria: Excellent(MAE<1%,RMSE<1.5%,R2>0.95) Good(MAE<3%,RMSE<4%,R2>0.85)')
    lines.append('  try7: B0005 small(64-64-64)+Cosine, B0007 transfer, B0018 independent')
    lines.append('=' * 90)
    content = '\n'.join(lines)
    with open(os.path.join(OUTPUT_DIR, 'comparison_table.txt'), 'w', encoding='utf-8') as f:
        f.write(content)
    print('\n' + content)

    assess = []
    assess.append('=' * 70)
    assess.append('    Assess Report (3-try7)')
    assess.append('=' * 70)
    assess.append('')
    assess.append('Changes from try4:')
    assess.append('  B0005: smaller model (64-64-64) + CosineAnnealingWarmRestarts')
    assess.append('  B0007: B0005 transfer (was independent training)')
    assess.append('  B0006: B0005 transfer (same as try4)')
    assess.append('  B0018: independent (same as try4)')
    assess.append('  Split A/B: same as try4')
    assess.append('')
    for group_name, results in all_group_results.items():
        assess.append(f'[{group_name}]')
        for battery in BATTERIES:
            if battery in results:
                r = results[battery]
                assess.append(f'  {battery}: MAE={r["mae"]:.4f}% RMSE={r["rmse"]:.4f}% R2={r["r2"]:.4f}')
        assess.append('')
    assess.append('=' * 70)
    with open(os.path.join(OUTPUT_DIR, 'assess.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(assess))

    for group_name, preds_targets in all_group_preds.items():
        plot_prediction_scatter(preds_targets, group_name, OUTPUT_DIR)
        plot_prediction_curve(preds_targets, group_name, OUTPUT_DIR)
        plot_loss_curves(preds_targets, group_name, OUTPUT_DIR)


def main():
    print('=' * 60)
    print('  CNN-BiLSTM-Attention SOH Prediction - 3-try7')
    print(f'  Key: B0005 small model+CosAnneal, B0007 B0005->transfer')
    print(f'  Device: {device}')
    print('=' * 60)

    all_group_results, all_group_preds = {}, {}

    ra, pa = run_group_a()
    all_group_results[GROUP_NAMES['A']] = ra
    all_group_preds[GROUP_NAMES['A']] = pa

    rb, pb = run_group_b()
    all_group_results[GROUP_NAMES['B']] = rb
    all_group_preds[GROUP_NAMES['B']] = pb

    rc, pc = run_group_c()
    all_group_results[GROUP_NAMES['C']] = rc
    all_group_preds[GROUP_NAMES['C']] = pc

    print('\n' + '=' * 60)
    print('  Generating Outputs...')
    generate_outputs(all_group_results, all_group_preds)
    print(f'\n  Results: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()
