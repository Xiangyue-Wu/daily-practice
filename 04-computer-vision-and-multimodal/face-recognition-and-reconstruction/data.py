import os
import re
import cv2
import numpy as np
import random
import shutil
import matplotlib.pyplot as plt

# 支持中文标签与正常显示负号
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def _get_paths():
    """返回项目中的数据、训练、验证、过程目录的绝对路径。
    说明：本项目的图像目录位于上一级目录 img 下，分别为：
    - img/data：原始采集的图像
    - img/training：按“人名_1/人名_2”重命名后的训练图像
    - img/validation：按“人名”重命名的验证图像
    - img/process：用于保存处理后的可视化结果
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_dir = os.path.join(base_dir, 'img', 'data')
    train_dir = os.path.join(base_dir, 'img', 'training')
    val_dir = os.path.join(base_dir, 'img', 'validation')
    proc_dir = os.path.join(base_dir, 'img', 'process')
    return data_dir, train_dir, val_dir, proc_dir

def _ensure_clean_dir(path):
    """确保目录存在且为空。
    若目录不存在则创建；若目录存在则清空其中的文件（保留子目录）。"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    else:
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                try:
                    os.remove(fp)
                except Exception:
                    pass
    return None

def _get_person_key(fname):
    """根据文件名提取人名的 key。
    规则：取文件名中靠前的连续字母作为人名（转小写），例如 sampleuser_1.jpg → sampleuser。"""
    base = os.path.splitext(os.path.basename(fname))[0]
    m = re.match(r'[a-zA-Z]+', base)
    key = m.group().lower() if m else base.lower()
    return key

def split_train_val():
    """将 img/data 中的原始图像按人分组，随机选两张入训练集、剩余一张入验证集。
    - 训练集重命名为 person_1.ext、person_2.ext
    - 验证集重命名为 person.ext
    - 便于后续从文件名直接获取标签名（人名）"""
    data_dir, train_dir, val_dir, _ = _get_paths()
    _ensure_clean_dir(train_dir)
    _ensure_clean_dir(val_dir)
    files = [f for f in sorted(os.listdir(data_dir)) if f.lower().endswith(('jpg', 'png', 'jpeg', 'bmp'))]
    groups = {}
    for fname in files:
        key = _get_person_key(fname)
        groups.setdefault(key, []).append(fname)
    rng = random.Random()
    for person, flist in groups.items():
        if len(flist) < 3:
            continue
        picks = flist.copy()
        rng.shuffle(picks)
        train_sel = picks[:2]
        val_sel = picks[2]
        # 保存训练两张：重命名为 person_1.ext, person_2.ext
        for i, src in enumerate(train_sel, start=1):
            ext = os.path.splitext(src)[1]
            dst = f'{person}_{i}{ext}'.lower()
            shutil.copy2(os.path.join(data_dir, src), os.path.join(train_dir, dst))
        # 保存验证一张：重命名为 person.ext
        ext_v = os.path.splitext(val_sel)[1]
        dst_v = f'{person}{ext_v}'.lower()
        shutil.copy2(os.path.join(data_dir, val_sel), os.path.join(val_dir, dst_v))
    return None

def load_training_data():
    """读取训练集为灰度图并统一大小为 480×640。
    返回一个形状为 (n, 480, 640) 的 numpy 数组，n 为训练样本数。"""
    _, train_dir, _, _ = _get_paths()
    files = [f for f in sorted(os.listdir(train_dir)) if f.lower().endswith(('jpg', 'png', 'jpeg', 'bmp'))]
    # 固定顺序：按人名与 _1/_2 排序保证矩阵列稳定
    files.sort(key=lambda x: (_get_person_key(x), x))
    imgs = []
    for fname in files:
        img = cv2.imread(os.path.join(train_dir, fname), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (640, 480))
        imgs.append(img.astype(np.float32))
    if len(imgs) == 0:
        return None
    X = np.stack(imgs, axis=0)
    return X

def visualize_processed_data():
    """可视化处理后的训练/验证图像各 3 张（每人两张训练 + 一张验证）。
    - 生成弹窗进行浏览，同时将拼接图保存到 img/process/person.png
    - 有助于快速检查数据处理是否正确（尺寸、灰度、命名）。"""
    data_dir, train_dir, val_dir, proc_dir = _get_paths()
    _ensure_clean_dir(proc_dir)
    # 组织每人的训练与验证文件
    train_files = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    val_files = [f for f in os.listdir(val_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    groups = {}
    for f in train_files:
        key = _get_person_key(f)
        groups.setdefault(key, {'train': [], 'val': None})['train'].append(f)
    for f in val_files:
        key = _get_person_key(f)
        if key in groups:
            groups[key]['val'] = f
        else:
            groups.setdefault(key, {'train': [], 'val': f})
    # 每人生成一个弹窗并保存
    for person, info in sorted(groups.items()):
        imgs = []
        for tf in sorted(info.get('train', [])):
            img = cv2.imread(os.path.join(train_dir, tf), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (640, 480))
            imgs.append(img)
        vf = info.get('val')
        if vf:
            imgv = cv2.imread(os.path.join(val_dir, vf), cv2.IMREAD_GRAYSCALE)
            if imgv is not None:
                imgv = cv2.resize(imgv, (640, 480))
                imgs.append(imgv)
        if len(imgs) != 3:
            continue
        fig = plt.figure(f'处理结果 - {person}', figsize=(9, 4))
        fig.set_facecolor('white')
        for i, im in enumerate(imgs, start=1):
            plt.subplot(1, 3, i)
            plt.imshow(im, cmap='gray')
            plt.axis('off')
            plt.title(f'{person} 第{i}张')
        out_path = os.path.join(proc_dir, f'{person}.png')
        plt.savefig(out_path)
        plt.show()
        plt.close(fig)
    return None

def save_matrices():
    """将训练/验证数据保存为列向量矩阵到 X_training.npy。
    - X_training.npy 保存一个字典：
      {'X_training': 形状 (307200, n_train), 'X_validation': 形状 (307200, n_val)}
    - 每张 480×640 灰度图拉直为 307200 维列向量并按列拼接
    - 后续 exercise_1.load_data 会识别此结构并正确读取训练矩阵"""
    _, train_dir, val_dir, _ = _get_paths()
    # 训练
    t_files = [f for f in sorted(os.listdir(train_dir)) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    t_files.sort(key=lambda x: (_get_person_key(x), x))
    t_imgs = []
    for f in t_files:
        im = cv2.imread(os.path.join(train_dir, f), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, (640, 480))
        t_imgs.append(im.astype(np.float32))
    # 验证
    v_files = [f for f in sorted(os.listdir(val_dir)) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    v_files.sort(key=lambda x: _get_person_key(x))
    v_imgs = []
    for f in v_files:
        im = cv2.imread(os.path.join(val_dir, f), cv2.IMREAD_GRAYSCALE)
        if im is None:
            continue
        im = cv2.resize(im, (640, 480))
        v_imgs.append(im.astype(np.float32))
    if len(t_imgs) == 0 or len(v_imgs) == 0:
        return None
    # 生成矩阵（按列拼接）
    H, W = 640, 480 
    D = H * W
    T = np.stack(t_imgs, axis=0).reshape(len(t_imgs), D)
    V = np.stack(v_imgs, axis=0).reshape(len(v_imgs), D)
    X_training = T.T
    X_validation = V.T
    out_npy = os.path.join(os.path.dirname(__file__), 'X_training.npy')
    np.save(out_npy, {'X_training': X_training.astype(np.float32), 'X_validation': X_validation.astype(np.float32)})
    return None

def main():
    """数据预处理主流程：划分训练/验证、读取与可视化、保存矩阵。"""
    # 1.划分训练/验证集并重命名
    _ = split_train_val()
    # 2.读取并处理训练图像为灰度 480×640
    X_train_imgs = load_training_data()
    # 3.可视化每人三张处理后的图像并保存
    _ = visualize_processed_data()
    # 4.保存训练/验证矩阵到 X_training.npy
    _ = save_matrices()
    return None

if __name__ == '__main__':
    # 数据导入、初始变量定义等
    _ = None
    main()

