import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文标签
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

def estimate_depth(left_path, right_path, show=True, save_results=True):
    # 读取左右两张图像
    img_left = cv2.imread(left_path, 0)
    img_right = cv2.imread(right_path, 0)
    
    # StereoBM 参数
    num_disparities = 64  # 视差搜索范围，必须是16的整数倍
    block_size = 21       # 匹配块大小，必须是奇数
    
    stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)
    
    # 计算视差图
    disparity = stereo.compute(img_left, img_right)
    
    # 将视差图归一化用于显示
    disparity_norm = cv2.normalize(disparity, None, alpha=0, beta=255, 
                                 norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # 应用色彩映射以便更好地可视化
    disparity_color = cv2.applyColorMap(disparity_norm, cv2.COLORMAP_JET)
    
    # 计算深度图（需要相机参数，这里使用近似值）
    # 注意：实际应用中需要准确的相机标定参数
    focal_length = 700  # 焦距（像素单位），需要根据实际相机标定设置
    baseline = 0.1     # 基线长度（米），需要根据实际相机设置
    
    # 避免除零错误，将视差为0的位置设置为最大深度
    disparity_for_depth = disparity.astype(np.float32)
    disparity_for_depth[disparity_for_depth <= 0] = 0.1
    
    # 计算深度图：depth = (f * b) / disparity
    depth_map = (focal_length * baseline) / disparity_for_depth
    
    if show:
        # 显示结果
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 2, 1)
        plt.imshow(img_left, cmap='gray')
        plt.title('左图像')
        plt.axis('off')
        
        plt.subplot(2, 2, 2)
        plt.imshow(img_right, cmap='gray')
        plt.title('右图像')
        plt.axis('off')
        
        plt.subplot(2, 2, 3)
        plt.imshow(disparity_norm, cmap='gray')
        plt.title('归一化视差图')
        plt.axis('off')
        plt.colorbar()
        
        plt.subplot(2, 2, 4)
        # 将BGR图像转换为RGB以供Matplotlib显示
        disparity_color_rgb = cv2.cvtColor(disparity_color, cv2.COLOR_BGR2RGB)
        plt.imshow(disparity_color_rgb)
        plt.title('彩色视差图 (修正后)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 同时显示OpenCV窗口（设置窗口可调整大小）
        cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Right Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Disparity', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Disparity Color', cv2.WINDOW_NORMAL)
        
        cv2.imshow('Left Image', img_left)
        cv2.imshow('Right Image', img_right)
        cv2.imshow('Disparity', disparity_norm)
        cv2.imshow('Disparity Color', disparity_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if save_results:
        # 保存结果图像
        cv2.imwrite('disparity.jpg', disparity_norm)
        cv2.imwrite('disparity_color.jpg', disparity_color)
        cv2.imwrite('depth_map.jpg', cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U))
    
    return disparity, depth_map

def main():
    # 1. 定义图像路径
    left_img = 'src1/2.jpg'
    right_img = 'src1/1.jpg'
    
    # 2. 调用深度估计函数
    disparity, depth_map = estimate_depth(left_img, right_img)
    
    return None

if __name__ == "__main__":
    main()