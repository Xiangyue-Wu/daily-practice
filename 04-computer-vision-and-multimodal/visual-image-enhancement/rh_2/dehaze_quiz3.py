# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


"""
I(x) = J(x) t(x) + A(1-t(x))
 
I(x): 含雾图像
J(x): 去雾图像
t(x): 透射率
A: 大气光成分

估计是需先估计A，之后根据公式计算得到t(x)，最后得到J(x)

需要实现：
计算A的函数：get_atmo(img)
计算t(x)的函数：get_trans(img, atom, w = 0.95)
给出J的表达公式

"""


def dark_channel(img, window_size=15):
    r, g, b = cv2.split(img)
    # 计算每个像素R、G、B分量中的最小值，得到单通道暗原图
    min_rgb = np.minimum(np.minimum(r, g), b)
    # 定义一个window_size x window_size的矩形结构元素，用于最小值滤波（形态学腐蚀）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    # 使用边界复制以避免边缘处由于零填充导致的伪影（白边/亮边）
    dc_img = cv2.erode(min_rgb, kernel, borderType=cv2.BORDER_REFLECT)
    return dc_img


def get_atmo(img, dc_img):
    # 估计大气光成分A
    # dark = dark_channel(img)  # 移除原有计算，使用传入的dc_img
    height, width = dc_img.shape
    total_pixels = height * width
    percent = 0.001
    num_pixels = int(total_pixels * percent)

    # 将暗通道展平并获取最亮像素的索引
    dark_flat = dc_img.reshape(-1)
    indices = np.argsort(dark_flat)[-num_pixels:]

    # 获取原图中对应位置像素
    img_flat = img.reshape(-1, 3)
    candidate_pixels = img_flat[indices]

    # 计算每个候选像素的亮度（最大通道值）
    brightness = np.max(candidate_pixels, axis=1)

    # 添加阈值过滤：排除异常高值像素（亮度 > 0.95）
    threshold = 0.95
    valid_mask = brightness <= threshold
    valid_indices = indices[valid_mask]
    if len(valid_indices) == 0:
        valid_indices = indices  # 如果全部排除，使用原候选

    # 结合位置信息：排除图像边缘像素（距离边界 < 5%）
    edge_margin = 0.05
    valid_positions = []
    for idx in valid_indices:
        y = idx // width
        x = idx % width
        if (edge_margin * height < y < (1 - edge_margin) * height) and \
           (edge_margin * width < x < (1 - edge_margin) * width):
            valid_positions.append(idx)

    if not valid_positions:
        valid_positions = valid_indices  # 如果全部排除，使用阈值后候选

    # 从有效候选中选择亮度最高的像素
    valid_pixels = img_flat[valid_positions]
    valid_brightness = np.max(valid_pixels, axis=1)
    max_idx = np.argmax(valid_brightness)
    A = valid_pixels[max_idx]
    return A

def get_trans(img, atom, omega=0.9, window_size=15):
    # 计算透射率t(x)
    x = img / atom  # 归一化
    t = 1 - omega * dark_channel(x, window_size)  # 使用调整后的omega和quiz2风格的dark_channel
    # 限制透射率范围，避免过小值导致噪声放大和过大值导致过曝
    t = np.clip(t, 0.1, 0.98)
    return t


def guided_filter(I, p, r=60, eps=1e-3):
    # 引导滤波函数，用于细化透射率t
    # I: 指导图像 (灰度图), p: 初始透射率图
    mean_I = cv2.boxFilter(I, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    mean_Ip = cv2.boxFilter(I * p, -1, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(I * I, -1, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, -1, (r, r))
    mean_b = cv2.boxFilter(b, -1, (r, r))

    q = mean_a * I + mean_b
    return q


def dehaze(path, output = None):
    im = cv2.imread(path)
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    # 计算暗通道图像
    dc_img = dark_channel(img)

    # quiz
    atom = get_atmo(img, dc_img)  # 更新调用，传入dc_img
    print(f"图像 {os.path.basename(path)} 的大气光A: {atom}")

    # quiz
    trans = get_trans(img, atom)

    # 使用引导滤波细化透射率
    t_refined = guided_filter(img_gray, trans)
    print(f"图像 {os.path.basename(path)} 的透射率t (均值): {np.mean(t_refined)}")

    result = np.empty_like(img)
    for i in range(3):
        # quiz
        result[:, :, i] = (img[:, :, i] - atom[i]) / t_refined + atom[i]

    # 裁剪结果防止越界
    result = np.clip(result, 0.0, 1.0)

    # cv2.imshow("source", img)
    # cv2.imshow("result", result)
    # cv2.waitKey()
    if output is not None:
        cv2.imwrite(output, (result * 255).astype(np.uint8))


def display_comparison(original_path, dehazed_path, filename, result_folder):
    original = cv2.imread(original_path)
    dehazed = cv2.imread(dehazed_path)
    if original is None or dehazed is None:
        print("无法读取图像")
        return
    # 转换为 RGB 以便 Matplotlib 显示
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    dehazed_rgb = cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6), dpi=150)
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title('处理前图像')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(dehazed_rgb)
    plt.title('处理后图像')
    plt.axis('off')
    # plt.suptitle('去雾效果对比：处理前 vs 处理后', y=0.995)
    # 计算上边距为10像素：高度=6英寸*dpi=150=900像素，top=1 - 10/900 ≈ 0.9889
    plt.subplots_adjust(left=0, right=1, top=0.9889, bottom=0)
    save_path = os.path.join(result_folder, f'对比_{os.path.splitext(filename)[0]}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    plt.show()
    plt.close()
    return None

if __name__ == '__main__':
    input_folder = r'data/exp_photos'
    output_folder = r'outputs/result2'
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png', '.bmp', '.tif')):
            img_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, 'dehazed_' + filename)
            dehaze(img_path, output_path)
            display_comparison(img_path, output_path, filename, output_folder)