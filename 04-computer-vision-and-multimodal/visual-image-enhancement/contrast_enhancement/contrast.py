import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def enhance_contrast(img_path, output_path):
    # 读取
    img = cv2.imread(img_path)

    # 将图像从BGR色彩空间转换为LAB色彩空间，将亮度 L与颜色信息A和B分离，
    # 这样可以独立地对亮度进行操作，而不会影响颜色。
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    # 将多通道图像分离成独立的单通道图像
    l, a, b = cv2.split(lab)

    # 运用CLAHE（自适应直方图均衡化方法），通过限制对比度增强的幅度来避免过度放大噪声
    # clipLimit: 对比度限制阈值，越大增强越明显，但噪声可能更多
    # tileGridSize: 直方图均衡化的网格大小（(8,8) 表示将图像分成8x8的网格）。图像会被分成多个小块，并在每个小块上独立进行均衡化
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))

    # 对单通道图像l应用 CLAHE 算法
    cl = clahe.apply(l)

    # 将增强后的L通道与原始的A和B通道合并，重新构建 LAB 图像，并转换回 BGR 色彩空间
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    cv2.imwrite(output_path, enhanced)
    return None

def display_comparison(original_path, enhanced_path, filename, result_folder):
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path)
    if original is None or enhanced is None:
        print("无法读取图像")
        return
    # 转换为 RGB 以便 Matplotlib 显示
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(original_rgb)
    plt.title('原始图像')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(enhanced_rgb)
    plt.title('增强后图像')
    plt.axis('off')
    plt.suptitle('图像效果对比：原始 vs 增强')
    save_path = os.path.join(result_folder, f'图像效果对比_{os.path.splitext(filename)[0]}.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()
    return None

def display_histogram_comparison(original_path, enhanced_path, filename, result_folder):
    original = cv2.imread(original_path)
    enhanced = cv2.imread(enhanced_path)
    if original is None or enhanced is None:
        print("无法读取图像")
        return
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    enhanced_lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
    original_l = cv2.split(original_lab)[0]
    enhanced_l = cv2.split(enhanced_lab)[0]
    
    plt.figure(figsize=(10, 5))
    # 原始图像的亮度直方图
    plt.subplot(1, 2, 1)
    plt.hist(original_l.ravel(), bins=256, range=[0, 256], color='steelblue', alpha=0.9)
    plt.title('原始 L 通道直方图')
    plt.xlabel('亮度值（L 通道 0–255）')
    plt.ylabel('像素数量')
    plt.xlim(0, 255)
    plt.grid(True, linestyle='--', alpha=0.3)
    # 增强后图像的亮度直方图
    plt.subplot(1, 2, 2)
    plt.hist(enhanced_l.ravel(), bins=256, range=[0, 256], color='tomato', alpha=0.9)
    plt.title('增强后 L 通道直方图')
    plt.xlabel('亮度值（L 通道 0–255）')
    plt.ylabel('像素数量')
    plt.xlim(0, 255)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(result_folder, f'L通道直方图对比_{os.path.splitext(filename)[0]}.png')
    plt.savefig(save_path)
    plt.show()
    plt.close()
    return None

def main():
    input_folder = r'data/exp_photos'
    output_folder = r'outputs/contrast_enhancement'
    result_folder = os.path.join(output_folder, 'result')
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.png', '.bmp', '.tif')):
            img_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, 'enhanced_' + filename)
            enhance_contrast(img_path, output_path)
            # 调用可视化函数
            display_comparison(img_path, output_path, filename, result_folder)
            display_histogram_comparison(img_path, output_path, filename, result_folder)

    return None

if __name__ == "__main__":
    main()

