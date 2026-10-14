import os
import numpy as np
import scipy.io as sio
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import random
from torch.nn.utils import weight_norm
import math

def load_mat(file_path):
    """
    读取指定路径的 .mat 文件并返回 Python 字典形式。
    参数:
        file_path: 字符串，.mat 文件的绝对/相对路径。
    返回:
        data: dict，scipy.io.loadmat 解析得到的键值对字典。
    说明:
        不做异常处理，读取失败由解释器抛出异常（便于显式发现数据问题）。
    """
    # 读取 .mat 文件，返回字典结构；不做异常处理，若失败将由解释器直接报错
    data = sio.loadmat(file_path)
    return data

def find_variable(d, preferred_names):
    """
    在 .mat 字典中查找可能的序列或标签变量。
    逻辑:
      1) 优先按候选名称列表 preferred_names 顺序匹配；
      2) 若无匹配，则在所有非系统键中选尺寸最大的数组作为兜底变量。
    参数:
      d: dict，.mat 读取结果；
      preferred_names: List[str]，候选键名列表。
    返回:
      选出的变量（可能是 ndarray 或 object 数组），找不到则返回 None。
    """
    # 在 .mat 字典中寻找可能的序列/标签变量
    # 1) 先按候选名称匹配；2) 若无匹配，则选择尺寸最大的变量作为兜底
    if d is None:
        return None
    keys = [k for k in d.keys() if not k.startswith('__')]
    for name in preferred_names:
        if name in d:
            return d[name]
    if not keys:
        return None
    arrs = [(k, np.array(d[k], dtype=object)) for k in keys]
    arrs.sort(key=lambda x: x[1].size if hasattr(x[1], 'size') else 0, reverse=True)
    return d[arrs[0][0]]

def to_labels(var):
    """
    将标签变量统一转换为一维整型数组（dtype=int64）。
    参数:
      var: 任意类型的标签变量（可能是标量、列向量、行向量或嵌套数组）。
    返回:
      一维 numpy 数组，长度为样本数。
    """
    # 将标签统一为一维整型数组
    if var is None:
        return np.array([], dtype=np.int64)
    arr = np.array(var)
    arr = arr.squeeze()
    arr = arr.flatten().astype(np.int64)
    return arr

def to_sequence_list(var, n_samples=None):
    """
    将序列变量统一为 List[np.ndarray]，每个元素形状为 T×D。
    兼容两类输入:
      - MATLAB 的 cell（在 Python 中体现为 dtype=object 的 ndarray），逐个解包；
      - 3D 数组（N×T×D 或其它轴顺序），根据 n_samples 推断样本维并拆分；
    参数:
      var: ndarray 或 object 数组；
      n_samples: 可选，样本数，用于从 3D 数组中确定哪一轴是样本维。
    返回:
      list[np.ndarray]，若无法识别则返回空列表。
    """
    # 将序列统一为 list，每个元素形状为 T×D
    # 兼容 MATLAB 的 cell（object 数组）与 3D 数组
    if var is None:
        return []
    if isinstance(var, np.ndarray) and var.dtype == object:
        flat = var.ravel()
        seqs = []
        for i in range(flat.size):
            x = np.array(flat[i])
            seqs.append(x)
        return seqs
    arr = np.array(var)
    if arr.ndim == 3:
        if n_samples is None:
            n_samples = arr.shape[0]
        axis_candidates = [i for i in range(3) if arr.shape[i] == n_samples]
        idx_n = axis_candidates[0] if axis_candidates else 0
        seqs = []
        if idx_n == 0:
            for i in range(n_samples):
                seqs.append(arr[i, :, :])
        elif idx_n == 1:
            for i in range(n_samples):
                seqs.append(arr[:, i, :])
        else:
            for i in range(n_samples):
                seqs.append(arr[:, :, i])
        return seqs
    if arr.ndim == 2:
        return [arr]
    return []

def prepare_data(base_dir):
    """
    从 base_dir 读取训练/验证的序列与标签 .mat 文件，统一结构形式。
    返回结构:
      {
        "train": {"seqs": List[np.ndarray(T×D)], "labels": np.ndarray(N,)},
        "val":   {"seqs": List[np.ndarray(T×D)], "labels": np.ndarray(M,)}
      }
    """
    # 仅执行“载入字典并统一结构”
    # 输出结构：
    # {
    #   "train": {"seqs": list[np.ndarray(T×D)], "labels": np.ndarray(N,)},
    #   "val":   {"seqs": list[np.ndarray(T×D)], "labels": np.ndarray(M,)}
    # }
    train_seq_mat = load_mat(os.path.join(base_dir, "train_seq.mat"))
    train_lbl_mat = load_mat(os.path.join(base_dir, "train_lbl.mat"))
    val_seq_mat = load_mat(os.path.join(base_dir, "val_seq.mat"))
    val_lbl_mat = load_mat(os.path.join(base_dir, "val_lbl.mat"))
    train_seq_var = find_variable(train_seq_mat, ["train_seq", "seq", "train"])
    train_lbl_var = find_variable(train_lbl_mat, ["train_lbl", "label", "lbl", "labels", "y"])
    val_seq_var = find_variable(val_seq_mat, ["val_seq", "seq", "val"])
    val_lbl_var = find_variable(val_lbl_mat, ["val_lbl", "label", "lbl", "labels", "y"])
    train_labels = to_labels(train_lbl_var)
    val_labels = to_labels(val_lbl_var)
    train_seqs = to_sequence_list(train_seq_var, n_samples=len(train_labels) if train_labels.size > 0 else None)
    val_seqs = to_sequence_list(val_seq_var, n_samples=len(val_labels) if val_labels.size > 0 else None)
    data = {
        "train": {"seqs": train_seqs, "labels": train_labels},
        "val": {"seqs": val_seqs, "labels": val_labels},
    }
    return data

def normalize_by_train_stats(data):
    """
    使用训练集统计参数对所有序列进行标准化: x' = (x - mu) / sigma。
    步骤:
      1) 纵向拼接训练集所有样本，按特征维计算均值 mu 与标准差 sigma；
      2) 用同一组 mu、sigma 标准化训练集与验证集；
      3) 返回标准化后的数据字典以及 (mu, sigma)。
    """
    # 使用训练集统计参数对数据进行归一化处理（零均值 + 单位方差）
    # 1) 计算训练集在特征维度上的均值mu与标准差sigma
    # 2) 用 (x - mu) / sigma 归一化训练与验证序列
    # 3) 返回归一化后的数据字典与参数（mu, sigma）
    train_seqs = data["train"]["seqs"]
    val_seqs = data["val"]["seqs"]
    def stack(seqs):
        arrs = []
        for s in seqs:
            a = np.array(s)
            if a.ndim == 2:
                arrs.append(a)
        if len(arrs) == 0:
            return None
        return np.vstack(arrs)
    Xtr = stack(train_seqs)
    mu = Xtr.mean(axis=0)
    sigma = Xtr.std(axis=0)
    sigma[sigma == 0] = 1.0
    def apply_norm(seqs):
        out = []
        for s in seqs:
            x = (np.array(s) - mu) / sigma
            out.append(x)
        return out
    train_norm = apply_norm(train_seqs)
    val_norm = apply_norm(val_seqs)
    norm_data = {
        "train": {"seqs": train_norm, "labels": data["train"]["labels"]},
        "val": {"seqs": val_norm, "labels": data["val"]["labels"]},
    }
    return norm_data, mu, sigma

class ResidualUnitTorch(nn.Module):
    """
    残差单元（Res-U），预激活结构: BN → ReLU → Conv。
    特性:
      - kernel 长度 f=8；
      - stride=1 时保持时间维长度不变（通过不对称常数填充实现 same）；
      - stride>1 时做下采样（如 stride=2），主分支与跳连分支长度严格一致；
    """
    def __init__(self, Cin, Cout, f=8, stride=1, dropout=0.2):
        super().__init__()
        self.bn = nn.BatchNorm1d(Cin)  # 按通道维做批归一化
        self.act = nn.ReLU()            # 非线性激活
        self.drop = nn.Dropout(dropout)
        self.stride = stride
        if stride == 1:
            pl = (f - 1) // 2
            pr = (f - 1) - pl
            self.pad = nn.ConstantPad1d((pl, pr), 0.0)  # 不对称常数填充，等效 same
            self.conv = weight_norm(nn.Conv1d(Cin, Cout, kernel_size=f, stride=1, padding=0))  # 保持长度
        else:
            pl = (f - 1) // 2
            pr = (f - 1) - pl
            self.pad = nn.ConstantPad1d((pl, pr), 0.0)  # 同样等效 same，但配合 stride>1
            self.conv = weight_norm(nn.Conv1d(Cin, Cout, kernel_size=f, stride=stride, padding=0))  # 下采样
        if stride > 1 or Cin != Cout:
            self.id = weight_norm(nn.Conv1d(Cin, Cout, kernel_size=1, stride=stride, padding=0))  # 跳连匹配通道/步长
        else:
            self.id = nn.Identity()  # 通道相同且不下采样，直接恒等映射
    def forward(self, x):
        y = self.bn(x)
        y = self.act(y)
        y = self.drop(y)
        if self.pad is not None:
            y = self.conv(self.pad(y))
        else:
            y = self.conv(y)
        s = self.id(x)   # 跳连分支（可能带 1×1 卷积与 stride）
        return y + s     # 残差相加

class ResidualBlockTorch(nn.Module):
    """
    残差块（Block），由 N 个同通道的 Res-U 顺序堆叠。
    第一个单元可选择 stride=s_first，其余默认 stride=1。
    """
    def __init__(self, C, N, s_first=1, f=8, dropout=0.2):
        super().__init__()
        units = [ResidualUnitTorch(Cin=C, Cout=C, f=f, stride=s_first, dropout=dropout)]
        for _ in range(max(0, N-1)):
            units.append(ResidualUnitTorch(Cin=C, Cout=C, f=f, stride=1, dropout=dropout))
        self.block = nn.Sequential(*units)
    def forward(self, x):
        return self.block(x)

class ResTCNNet(nn.Module):
    """
    Res-TCN 主网络：
      - 输入维 D；
      - 初始卷积至 64 通道；
      - Block-A(64, NA)、转换 64→128；Block-B(128, NB)、转换 128→256；Block-C(256, NC)；
      - Global Average Pooling → 全连接 Ccls。
    """
    def __init__(self, D, Ccls, f=8, NA=3, NB=2, NC=3, dropout=0.2):
        super().__init__()
        self.conv_in = weight_norm(nn.Conv1d(D, 64, kernel_size=f, stride=1, padding=f//2))
        self.blockA = ResidualBlockTorch(C=64, N=NA, s_first=1, f=f, dropout=dropout)
        self.transAB = ResidualUnitTorch(Cin=64, Cout=128, f=f, stride=2, dropout=dropout)
        self.blockB = ResidualBlockTorch(C=128, N=NB, s_first=1, f=f, dropout=dropout)
        self.transBC = ResidualUnitTorch(Cin=128, Cout=256, f=f, stride=2, dropout=dropout)
        self.blockC = ResidualBlockTorch(C=256, N=NC, s_first=1, f=f, dropout=dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)  # 全局平均池化到每通道 1 值
        self.fc = nn.Linear(256, Ccls)      # 线性分类器
    def forward(self, X):
        X = X.transpose(1, 2)  # 输入形状 (N, T, D) → Conv1d 需要 (N, C_in=D, T)
        X = self.conv_in(X)
        X = self.blockA(X)
        X = self.transAB(X)
        X = self.blockB(X)
        X = self.transBC(X)
        X = self.blockC(X)
        V = self.gap(X).squeeze(-1)  # (N, 256, 1) → (N, 256)
        logits = self.fc(V)          # (N, Ccls)
        return logits

def train_restcn(base_dir, epochs, batch_size, lr, log_dir):
    """
    训练 Res-TCN 并记录指标到 TensorBoard。
    包含: 数据读取/标准化、DataLoader 构造、前向/反向、验证集评估与日志写入。
    返回: 一个字典，含训练/验证损失与准确率曲线、模型与设备、数据加载器等。
    """
    data = prepare_data(base_dir)
    norm_data, mu, sigma = normalize_by_train_stats(data)
    def stack(seqs):
        arrs = []
        for s in seqs:
            a = np.array(s)
            if a.ndim == 2:
                arrs.append(a)
        return np.stack(arrs).astype(np.float32)
    Xtr = stack(norm_data["train"]["seqs"])
    ytr = norm_data["train"]["labels"].astype(np.int64)
    Xva = stack(norm_data["val"]["seqs"])
    yva = norm_data["val"]["labels"].astype(np.int64)
    T0 = Xtr.shape[1]
    D = Xtr.shape[2]
    Ccls = int(max(ytr.max(), yva.max())) + 1
    model = ResTCNNet(D=D, Ccls=Ccls, f=8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, nesterov=True)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    tr_ds = TensorDataset(torch.tensor(Xtr), torch.tensor(ytr))
    va_ds = TensorDataset(torch.tensor(Xva), torch.tensor(yva))
    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    def augment_batch(x):
        if torch.rand(()) < 0.4:
            N, T, D = x.size()
            x = x + torch.randn_like(x) * 0.005
            mask_dims = (torch.rand((N, 1, D), device=x.device) > 0.02).float()
            x = x * mask_dims
            L = max(1, int(T * 0.05))
            starts = torch.randint(0, T - L + 1, (N,), device=x.device)
            for i in range(N):
                x[i, starts[i]:starts[i] + L, :] = 0.0
        return x
    def mixup_batch(x, y, alpha=0.15, p=0.4):
        if torch.rand(()) < p:
            lam = torch.distributions.Beta(alpha, alpha).sample().item()
            idx = torch.randperm(x.size(0), device=x.device)
            x = lam * x + (1 - lam) * x[idx]
            y_b = y[idx]
            return x, y, y_b, lam
        return x, y, y, 1.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    best_va = -1.0
    best_ep = 0
    best_state = None
    patience = max(10, epochs // 5)
    for ep in range(epochs):
        model.train()
        tl_sum = 0.0
        ta_sum = 0.0
        tn = 0
        for xb, yb in tr_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            xb = augment_batch(xb)
            xb, ya, yb_mix, lam = mixup_batch(xb, yb, alpha=0.2, p=0.5)
            logits = model(xb)
            loss = lam * loss_fn(logits, ya) + (1 - lam) * loss_fn(logits, yb_mix)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            tl_sum += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            ta_sum += float((pred == yb).sum().item())
            tn += xb.size(0)
        tl = tl_sum / tn
        ta = ta_sum / tn
        model.eval()
        vl_sum = 0.0
        va_sum = 0.0
        vn = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                vl_sum += float(loss.item()) * xb.size(0)
                pred = logits.argmax(dim=1)
                va_sum += float((pred == yb).sum().item())
                vn += xb.size(0)
        vl = vl_sum / vn
        va = va_sum / vn
        train_losses.append(tl)
        val_losses.append(vl)
        train_accs.append(ta)
        val_accs.append(va)
        writer.add_scalar("loss/train", tl, ep+1)
        writer.add_scalar("loss/val", vl, ep+1)
        writer.add_scalar("acc/train", ta, ep+1)
        writer.add_scalar("acc/val", va, ep+1)
        scheduler.step()
        if va > best_va:
            best_va = va
            best_ep = ep + 1
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if (ep + 1) - best_ep >= patience:
            break
    writer.flush()
    writer.close()
    if best_state is not None:
        model.load_state_dict(best_state)
    return {
        "train_loss": train_losses,
        "val_loss": val_losses,
        "train_acc": train_accs,
        "val_acc": val_accs,
        "classes": Ccls,
        "T": T0,
        "D": D,
        "model": model,
        "device": device,
        "train_loader": tr_loader,
        "val_loader": va_loader,
    }

def log_tensorboard_curves(metrics, log_dir, result_dir):
    """
    将训练/验证损失与准确率曲线保存为图片，字体使用中文兼容设置。
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    os.makedirs(result_dir, exist_ok=True)
    ep = np.arange(1, len(metrics["train_loss"]) + 1)
    plt.figure(figsize=(9, 6))
    plt.subplot(2,1,1)
    plt.plot(ep, metrics["train_loss"], label="训练损失", color="tab:blue")
    plt.plot(ep, metrics["val_loss"], label="验证损失", color="tab:orange")
    plt.xlabel("Epoch")
    plt.ylabel("损失")
    plt.title("Res-TCN 训练/验证损失")
    plt.legend()
    plt.subplot(2,1,2)
    plt.plot(ep, metrics["train_acc"], label="训练准确率", color="tab:green")
    plt.plot(ep, metrics["val_acc"], label="验证准确率", color="tab:red")
    plt.xlabel("Epoch")
    plt.ylabel("准确率")
    plt.title("Res-TCN 训练/验证准确率")
    plt.legend()
    plt.tight_layout()
    out_png = os.path.join(result_dir, "restcn_curves.png")
    plt.savefig(out_png)
    plt.close()
    return None

def compute_confusion(model, loader, device, num_classes):
    """
    计算混淆矩阵（行是真实类别，列是预测类别）。
    """
    # 计算混淆矩阵：行是真实类别，列是预测类别
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            t = yb.view(-1).cpu().numpy()
            p = pred.view(-1).cpu().numpy()
            for ti, pi in zip(t, p):
                cm[int(ti), int(pi)] += 1
    return cm

def plot_confusion(cm, title, out_path):
    """
    可视化并保存按行归一化后的混淆矩阵。
    """
    # 可视化混淆矩阵（按行归一化），中文标签并保存
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    sums = cm.sum(axis=1, keepdims=True)
    sums[sums == 0] = 1
    cm_norm = cm / sums
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm_norm, cmap='Blues', vmin=0.0, vmax=1.0)
    plt.title(title)
    plt.xlabel('预测类别')
    plt.ylabel('真实类别')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    classes = cm.shape[0]
    if classes <= 20:
        ticks = np.arange(classes)
        plt.xticks(ticks)
        plt.yticks(ticks)
    else:
        step = max(1, classes // 10)
        ticks = np.arange(0, classes, step)
        plt.xticks(ticks)
        plt.yticks(ticks)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    return None

def set_seed(s):
    """
    设定 Python、NumPy、PyTorch 的随机种子，保证可复现性。
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)

def plot_val_curve_band(runs, out_path):
    """
    绘制多次运行的验证集准确率均值曲线及 ±1 标准差带，保存为图片。
    """
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    lens = [len(r["val_acc"]) for r in runs if "val_acc" in r]
    if len(lens) == 0:
        return None
    K = min(lens)
    if K == 0:
        return None
    M = np.stack([np.array(r["val_acc"][:K], dtype=np.float64) for r in runs], axis=0)
    mu = M.mean(axis=0)
    sd = M.std(axis=0)
    x = np.arange(1, K+1)
    plt.figure(figsize=(6,4))
    plt.plot(x, mu, label="验证集准确率均值", color="tab:red")
    plt.fill_between(x, mu-sd, mu+sd, alpha=0.2, color="tab:red", label="±1 标准差")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.legend(); plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def run_stability_with_curves(base_dir, result_dir, repeats=5, epochs=5, batch_size=64, lr=1e-3):
    """
    多次重复训练以评估稳定性，绘制验证集准确率的均值与带宽。
    返回 runs 列表与末尾验证准确率数组。
    """
    runs = []
    for i in range(repeats):
        set_seed(1000 + i)
        log_dir = os.path.join(result_dir, f"tb_logs/run_{i:03d}")
        m = train_restcn(base_dir, epochs=epochs, batch_size=batch_size, lr=lr, log_dir=log_dir)
        runs.append(m)
    os.makedirs(result_dir, exist_ok=True)
    plot_val_curve_band(runs, out_path=os.path.join(result_dir, "stability_val_acc_band.png"))
    vals = np.array([m["val_acc"][-1] for m in runs], dtype=np.float64)
    print("val_acc mean=", float(vals.mean()), "std=", float(vals.std()))
    return runs, vals


def main():
    """
    主入口：训练模型、保存曲线、计算并可视化混淆矩阵、进行稳定性测试。
    """
    # 1.训练 Res-TCN 并写入 TensorBoard
    metrics = train_restcn(BASE_DIR, epochs=200, batch_size=64, lr=5e-2, log_dir=os.path.join(RESULT_DIR, "tb_logs"))
    # 2.可视化并保存训练/验证曲线
    log_tensorboard_curves(metrics, log_dir=os.path.join(RESULT_DIR, "tb_logs"), result_dir=RESULT_DIR)
    # 3.计算并可视化训练集与验证集的混淆矩阵
    model = metrics["model"]
    device = metrics["device"]
    tr_loader = metrics["train_loader"]
    va_loader = metrics["val_loader"]
    num_classes = metrics["classes"]
    os.makedirs(RESULT_DIR, exist_ok=True)
    cm_tr = compute_confusion(model, tr_loader, device, num_classes)
    cm_va = compute_confusion(model, va_loader, device, num_classes)
    plot_confusion(cm_tr, title="训练集混淆矩阵（行归一化）", out_path=os.path.join(RESULT_DIR, "confusion_train.png"))
    plot_confusion(cm_va, title="验证集混淆矩阵（行归一化）", out_path=os.path.join(RESULT_DIR, "confusion_val.png"))

    # 4.可视化验证集准确率曲线
    runs, vals = run_stability_with_curves(base_dir=BASE_DIR, result_dir=RESULT_DIR, repeats=10, epochs=200, batch_size=64, lr=5e-2)

if __name__=="__main__":
    BASE_DIR = r"."
    RESULT_DIR = r"outputs/results"
    main()
