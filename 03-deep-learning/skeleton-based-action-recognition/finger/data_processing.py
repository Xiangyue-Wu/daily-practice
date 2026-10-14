import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 加载数据
data = sio.loadmat('../FPHA/train_seq.mat')
train_seq = data['train_seq']  # shape: (1, 600), dtype=object

# 取第一个样本
first_sample = train_seq[0, 0]  # shape: (63, T) ← 注意！是 (63, T)，不是 (T, 63)
print("✅ 第一个样本原始形状:", first_sample.shape)  # e.g., (63, 50)

# 检查是否 63 行（即 21*3）
assert first_sample.shape[0] == 63, f"第一维应为63（21关节×3坐标），但得到 {first_sample.shape[0]}"

# 转置 + reshape
T = first_sample.shape[1]
skeleton = first_sample.T.reshape(T, 21, 3)  # 关键：先 .T 再 reshape
print("✅ 转换后形状:", skeleton.shape)  # 应为 (50, 21, 3)

# 骨架连接定义
BONES = [
    (0,1),(1,2),(2,3),(3,4),   # 拇指
    (0,5),(5,6),(6,7),(7,8),   # 食指
    (0,9),(9,10),(10,11),(11,12),   # 中指
    (0,13),(13,14),(14,15),(15,16),   # 无名指
    (0,17),(17,18),(18,19),(19,20)   # 小指
]

# 可视化第一帧
first_frame = skeleton[0]  # (21, 3)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

for start, end in BONES:
    xs = [first_frame[start, 0], first_frame[end, 0]]
    ys = [first_frame[start, 1], first_frame[end, 1]]
    zs = [first_frame[start, 2], first_frame[end, 2]]
    ax.plot(xs, ys, zs, 'b-', linewidth=2)

ax.scatter(first_frame[:, 0], first_frame[:, 1], first_frame[:, 2], c='red', s=40)
ax.set_title('First Sample – First Frame (Transposed & Reshaped)')
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.view_init(elev=30, azim=-90)

plt.tight_layout()
plt.show()