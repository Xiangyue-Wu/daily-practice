import cv2
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class DigitRecognizer:
    def __init__(self, template_dir, plates_dir):
        """初始化数字识别器（集成模型版本）"""
        self.template_dir = template_dir
        self.plates_dir = plates_dir
        self.templates = {}
        self.template_features = {}
        self.feature_extractor = None
        self.scaler = StandardScaler()
        
        # 集成模型组件
        self.hog_classifier = SVC(probability=True)       # HOG特征分类器
        self.topology_classifier = RandomForestClassifier(n_estimators=100)  # 拓扑特征分类器
        self.final_classifier = SVC(probability=True)     # 最终集成分类器
        
        self.is_trained = False
        self.training_history = {'accuracy': []}
        self.hog_dim = 0  # HOG特征维度
    
    def load_templates(self):
        """加载模板图像"""
        print("正在加载模板图像...")
        for i in range(10):
            template_path = os.path.join(self.template_dir, f"{i}.jpg")
            if not os.path.exists(template_path):
                template_path = os.path.join(self.template_dir, f"{i}.png")
            if not os.path.exists(template_path):
                print(f"警告: 未找到数字{i}的模板图像")
                continue
                
            img = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                self.templates[i] = img
                print(f"成功加载数字{i}的模板图像")
            else:
                print(f"警告: 无法加载数字{i}的模板图像")
    
    def preprocess_image(self, img, target_size=(28, 28)):
        """增强的图像预处理"""
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # 添加对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        img = clahe.apply(img)
        
        # 高斯模糊
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        # 自适应阈值（多种参数尝试）
        binary_methods = [
            cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2),
            cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 3),
            cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 7, 2)
        ]
        
        # 选择效果最好的二值化结果（基于前景像素数量）
        best_binary = max(binary_methods, key=lambda x: np.sum(x > 0))
        
        # 形态学操作
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(best_binary, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(best_binary, cv2.MORPH_OPEN, kernel)
        
        return img

    
    def extract_features(self, img, use_kmeans=True):
        """特征提取 + 封闭区域数量计算（集成模型版）"""
        if img.shape != (28, 28):
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        # 1. 计算HOG特征
        hog = cv2.HOGDescriptor((28, 28), (14, 14), (7, 7), (7, 7), 9)
        
        hog_result = hog.compute(img)
    
    # 处理返回结果（兼容元组和数组）
        if isinstance(hog_result, tuple):
          hog_features = hog_result[0].flatten()  # 从元组中提取数组[1](@ref)
        else:
            hog_features = hog_result.flatten()
        
       
        self.hog_dim = len(hog_features)  # 记录HOG维度
        
        # 2. 计算封闭区域数量（拓扑特征）
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        hole_count = 0
        if hierarchy is not None and len(hierarchy) > 0:
            for h in hierarchy[0]:
                if h[3] != -1:  # 存在父轮廓，即内部空洞
                    hole_count += 1
        
        # 3. 合并特征（HOG + 拓扑特征）
        combined_features = np.hstack([hog_features, np.array([hole_count])])
        
        # 4. KMeans聚类（可选）
        if use_kmeans and self.feature_extractor is not None:
            kmeans_features = self.feature_extractor.predict([combined_features])[0]
            return np.hstack([combined_features, kmeans_features])
        else:
            return combined_features
    
    def augment_data(self, img, num_augmentations=5):
        """数据增强"""
        augmented_images = [img]
        
        for _ in range(num_augmentations):
            angle = random.uniform(-15, 15)
            rows, cols = img.shape
            M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
            rotated = cv2.warpAffine(img, M, (cols, rows), borderValue=255)
            
            tx = random.randint(-3, 3)
            ty = random.randint(-3, 3)
            M = np.float32([[1, 0, tx], [0, 1, ty]])
            translated = cv2.warpAffine(rotated, M, (cols, rows), borderValue=255)
            
            scale = random.uniform(0.9, 1.1)
            scaled = cv2.resize(translated, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            scaled = cv2.resize(scaled, (28, 28), interpolation=cv2.INTER_AREA)
            
            augmented_images.append(scaled)
        
        return augmented_images
    
    def train(self):
        """训练集成识别模型"""
        print("开始训练集成模型...")
        if not self.templates:
            print("错误: 没有加载模板图像")
            return False
        
        # 准备训练数据
        X = []  # 所有特征
        y = []  # 标签
        all_features = []  # 用于KMeans
        
        for digit, img in self.templates.items():
            preprocessed = self.preprocess_image(img)
            augmented_imgs = self.augment_data(preprocessed, num_augmentations=10)
            
            for aug_img in augmented_imgs:
                features = self.extract_features(aug_img, use_kmeans=False)
                X.append(features)
                y.append(digit)
                all_features.append(features)
        
        if not all_features:
            print("错误: 无法提取特征")
            return False
        
        # 1. KMeans特征聚类
        all_features = np.array(all_features, dtype=np.float64)
        self.feature_extractor = KMeans(n_clusters=10, random_state=42)
        self.feature_extractor.fit(all_features)
        
        # 2. 重新提取特征（带KMeans标签）
        X_kmeans = []
        for features in X:
            kmeans_label = self.feature_extractor.predict([features])[0]
            X_kmeans.append(np.hstack([features, kmeans_label]))
        
        # 3. 特征标准化
        X_scaled = self.scaler.fit_transform(np.array(X_kmeans, dtype=np.float64))
        
        # 4. 分割HOG特征和拓扑特征
        hog_features = [x[:self.hog_dim+1] for x in X_scaled]  # HOG + 拓扑特征
        kmeans_features = [x[self.hog_dim+1:] for x in X_scaled]  # KMeans标签
        
        # 5. 训练集成模型的各个组件
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # 训练HOG+拓扑特征分类器
        self.hog_classifier.fit(hog_features, y)
        
        # 训练KMeans特征分类器（示例中使用随机森林）
        self.topology_classifier.fit(kmeans_features, y)
        
        # 6. 集成特征（各分类器的概率输出）
        hog_probs = self.hog_classifier.predict_proba(hog_features)
        topology_probs = self.topology_classifier.predict_proba(kmeans_features)
        combined_probs = np.hstack([hog_probs, topology_probs])
        
        # 7. 训练最终集成分类器
        self.final_classifier.fit(combined_probs, y)
        
        # 评估模型
        y_pred = self.final_classifier.predict(self._get_combined_probs(X_test, hog_features, kmeans_features))
        accuracy = accuracy_score(y_test, y_pred)
        print(f"集成模型训练完成，测试准确率: {accuracy:.2f}")
        
        self.training_history['accuracy'].append(accuracy)
        self.is_trained = True
        return True
    
    def _get_combined_probs(self, X, hog_features, kmeans_features):
        """获取集成模型的输入特征（概率组合）"""
        hog_test = [x[:self.hog_dim+1] for x in X]
        kmeans_test = [x[self.hog_dim+1:] for x in X]
        
        hog_probs = self.hog_classifier.predict_proba(hog_test)
        topology_probs = self.topology_classifier.predict_proba(kmeans_test)
        return np.hstack([hog_probs, topology_probs])
    
    def segment_digits(self, plate_img):
        """使用优化的连通域算法分割车牌字符"""
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
        
        # 增强预处理：提高分割效果
        preprocessed = self.preprocess_image(gray, target_size=None)
        _, binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        h, w = binary.shape
        digit_regions = self._segment_characters_clean_optimized(binary)
        
        digits = []
        for x, y, w_c, h_c in digit_regions:
            # 提取字符区域并预处理
            digit_img = gray[y:y+h_c, x:x+w_c]
            preprocessed_digit = self.preprocess_image(digit_img)
            digits.append(preprocessed_digit)
        
        print(f"优化分割后找到 {len(digits)} 个字符")
        return digits, digit_regions
    
    def _segment_characters_clean_optimized(self, binary_img):
        """优化的字符分割算法（处理粘连和倾斜）"""
        h, w = binary_img.shape
        
        # 1. 形态学操作：使用多种核
        kernels = [
            np.ones((2, 1), np.uint8),  # 垂直连接
            np.ones((3, 1), np.uint8),  # 更强垂直连接
            np.ones((1, 2), np.uint8)   # 水平连接
        ]
        
        processed_images = []
        for kernel in kernels:
            processed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            processed_images.append(processed)
        
        all_candidates = []
        
        # 2. 对每个处理后的图像进行连通域分析
        for i, processed in enumerate(processed_images):
            # 反转图像进行连通域分析
            inverted = 255 - processed
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
            
            char_candidates = []
            for j in range(1, num_labels):
                x = stats[j][cv2.CC_STAT_LEFT]
                y = stats[j][cv2.CC_STAT_TOP]
                w_c = stats[j][cv2.CC_STAT_WIDTH]
                h_c = stats[j][cv2.CC_STAT_HEIGHT]
                area = stats[j][cv2.CC_STAT_AREA]
                
                aspect = w_c / h_c if h_c > 0 else 0
                
                # 根据不同的核调整阈值
                min_area_ratio = 0.002 if i == 0 else 0.003
                max_aspect = 1.3 if i < 2 else 1.1
                
                if (area > h * w * min_area_ratio and
                    area < h * w * 0.3 and
                    0.1 < aspect < max_aspect and
                    h_c > h * 0.25 and
                    w_c > w * 0.01):
                    char_candidates.append((x, y, w_c, h_c, area, i))
            
            all_candidates.extend(char_candidates)
        
        if not all_candidates:
            return []
        
        # 3. 去重和合并相近的候选区域
        all_candidates.sort(key=lambda x: x[0])  # 按x坐标排序
        
        merged_candidates = []
        used = set()
        
        for i, cand1 in enumerate(all_candidates):
            if i in used:
                continue
                
            x1, y1, w1, h1, area1, src1 = cand1
            merged_group = [cand1]
            used.add(i)
            
            # 查找重叠的候选区域
            for j, cand2 in enumerate(all_candidates[i+1:], i+1):
                if j in used:
                    continue
                    
                x2, y2, w2, h2, area2, src2 = cand2
                
                # 计算重叠度
                overlap_x = max(0, min(x1+w1, x2+w2) - max(x1, x2))
                overlap_y = max(0, min(y1+h1, y2+h2) - max(y1, y2))
                overlap_area = overlap_x * overlap_y
                
                union_area = w1*h1 + w2*h2 - overlap_area
                iou = overlap_area / union_area if union_area > 0 else 0
                
                if iou > 0.3:  # 重叠度阈值
                    merged_group.append(cand2)
                    used.add(j)
            
            # 合并组内所有候选区域
            if merged_group:
                xs = [c[0] for c in merged_group]
                ys = [c[1] for c in merged_group]
                x_ends = [c[0] + c[2] for c in merged_group]
                y_ends = [c[1] + c[3] for c in merged_group]
                
                merged_x = min(xs)
                merged_y = min(ys)
                merged_w = max(x_ends) - merged_x
                merged_h = max(y_ends) - merged_y
                
                merged_candidates.append((merged_x, merged_y, merged_w, merged_h))
        
        # 4. 按x坐标排序
        merged_candidates.sort(key=lambda x: x[0])
        
        # 5. 限制字符数量（7个字符：省份简称+字母+5个数字）
        if len(merged_candidates) > 7:
            # 保留最大的7个区域
            merged_candidates = sorted(merged_candidates, key=lambda x: x[2]*x[3], reverse=True)[:7]
            merged_candidates.sort(key=lambda x: x[0])  # 重新按x排序
        
        # 6. 分割过宽的区域
        final_candidates = []
        for x, y, w_c, h_c in merged_candidates:
            aspect = w_c / h_c if h_c > 0 else 0
            
            # 如果区域过宽，可能包含多个字符
            if aspect > 1.8 and w_c > w * 0.12:
                # 分割为两个区域
                w1 = w_c // 2
                w2 = w_c - w1
                final_candidates.append((x, y, w1, h_c))
                final_candidates.append((x + w1, y, w2, h_c))
            else:
                final_candidates.append((x, y, w_c, h_c))
        
        # 7. 最终排序和数量限制
        final_candidates.sort(key=lambda x: x[0])
        return final_candidates[:7]  # 最多7个字符
    
    def recognize_digit(self, digit_img):
        """使用集成模型识别单个数字"""
        if not self.is_trained:
            print("错误: 模型未训练")
            return -1, 0.0
        
        # 提取特征
        features = self.extract_features(digit_img)
        features = np.array(features, dtype=np.float64)
        features_scaled = self.scaler.transform([features])
        
        # 分割特征
        hog_feature = features_scaled[0, :self.hog_dim+1].reshape(1, -1)
        kmeans_feature = features_scaled[0, self.hog_dim+1:].reshape(1, -1)
        
        # 获取各分类器的概率输出
        hog_probs = self.hog_classifier.predict_proba(hog_feature)
        topology_probs = self.topology_classifier.predict_proba(kmeans_feature)
        
        # 集成概率
        combined_probs = np.hstack([hog_probs, topology_probs])
        
        # 最终分类
        digit = self.final_classifier.predict(combined_probs)[0]
        confidence = np.max(self.final_classifier.predict_proba(combined_probs))
        
        return digit, confidence
    
    def visualize_all_digits_recognition(self):
        """可视化所有数字的识别过程"""
        if not self.is_trained or not self.templates:
            print("错误: 模型未训练或无模板")
            return
        
        # 创建大图显示所有数字的识别过程
        fig, axes = plt.subplots(4, 10, figsize=(25, 10))
        fig.suptitle('所有数字识别过程可视化', fontsize=20, fontweight='bold')
        
        for digit in range(10):
            if digit not in self.templates:
                # 如果某个数字没有模板，显示空白
                for row in range(4):
                    axes[row, digit].text(0.5, 0.5, f'数字 {digit}\n(无模板)', 
                                        ha='center', va='center', transform=axes[row, digit].transAxes,
                                        fontsize=12, color='red')
                    axes[row, digit].axis('off')
                continue
            
            # 获取原始模板
            original_img = self.templates[digit]
            
            # 第一行：原始模板
            axes[0, digit].imshow(original_img, cmap='gray')
            axes[0, digit].set_title(f'数字 {digit}\n(原始模板)', fontsize=10)
            axes[0, digit].axis('off')
            
            # 第二行：预处理后
            preprocessed = self.preprocess_image(original_img)
            axes[1, digit].imshow(preprocessed, cmap='gray')
            axes[1, digit].set_title('预处理后', fontsize=10)
            axes[1, digit].axis('off')
            
            # 第三行：数据增强示例
            augmented = self.augment_data(preprocessed, num_augmentations=1)[1]
            axes[2, digit].imshow(augmented, cmap='gray')
            axes[2, digit].set_title('数据增强示例', fontsize=10)
            axes[2, digit].axis('off')
            
            # 第四行：识别结果
            digit_pred, confidence = self.recognize_digit(preprocessed)
            axes[3, digit].imshow(preprocessed, cmap='gray')
            color = 'green' if digit_pred == digit else 'red'
            axes[3, digit].set_title(f'识别: {digit_pred}\n置信度: {confidence:.2f}', 
                                   fontsize=10, color=color)
            axes[3, digit].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_feature_comparison(self):
        """可视化不同数字的特征对比"""
        if not self.is_trained or not self.templates:
            print("错误: 模型未训练或无模板")
            return
        
        # 提取所有数字的特征
        features_data = {}
        for digit in range(10):
            if digit in self.templates:
                preprocessed = self.preprocess_image(self.templates[digit])
                features = self.extract_features(preprocessed)
                features_scaled = self.scaler.transform([features])[0]
                features_data[digit] = features_scaled
        
        if not features_data:
            print("错误: 无法提取特征")
            return
        
        # 选择前10个特征进行可视化
        feature_count = min(10, len(list(features_data.values())[0]))
        
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle('不同数字的特征对比', fontsize=16, fontweight='bold')
        
        for i in range(10):
            row = i // 5
            col = i % 5
            
            if i in features_data:
                features = features_data[i][:feature_count]
                axes[row, col].bar(range(feature_count), features, color='skyblue', alpha=0.7)
                axes[row, col].set_title(f'数字 {i} 的特征', fontsize=12)
                axes[row, col].set_xlabel('特征索引')
                axes[row, col].set_ylabel('特征值')
                axes[row, col].grid(True, alpha=0.3)
            else:
                axes[row, col].text(0.5, 0.5, f'数字 {i}\n(无数据)', 
                                  ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_specific_plates(self, plate_range=(0, 9)):
        """可视化特定范围的车牌识别结果"""
        if not self.is_trained:
            print("错误: 模型未训练")
            return
        
        start_plate, end_plate = plate_range
        results_data = {}
        
        # 处理指定范围的车牌
        for i in range(start_plate, end_plate + 1):
            plate_file = f"plate_{i}.jpeg"
            plate_path = os.path.join(self.plates_dir, plate_file)
            
            if not os.path.exists(plate_path):
                print(f"警告: 未找到文件 {plate_file}")
                continue
            
            print(f"处理车牌: {plate_file}")
            plate_img = cv2.imread(plate_path)
            if plate_img is None:
                print(f"错误: 无法加载图像 {plate_file}")
                continue
            
            # 分割和识别数字
            digits, digit_regions = self.segment_digits(plate_img)
            recognition_results = []
            
            for digit_img in digits:
                digit, confidence = self.recognize_digit(digit_img)
                recognition_results.append((digit, confidence))
            
            results_data[i] = {
                'plate_img': plate_img,
                'digits': digits,
                'digit_regions': digit_regions,
                'results': recognition_results,
                'plate_file': plate_file
            }
        
        if not results_data:
            print("错误: 没有找到有效的车牌图像")
            return
        
        # 创建可视化
        self._create_plate_recognition_visualization(results_data)
    
    def _create_plate_recognition_visualization(self, results_data):
        """创建车牌识别结果的可视化"""
        num_plates = len(results_data)
        max_digits = max(len(data['digits']) for data in results_data.values())
        
        # 创建大图显示所有车牌识别结果
        fig, axes = plt.subplots(3, num_plates, figsize=(4*num_plates, 12))
        fig.suptitle('车牌识别结果可视化', fontsize=20, fontweight='bold')
        
        for idx, (plate_num, data) in enumerate(results_data.items()):
            plate_img = data['plate_img']
            digits = data['digits']
            results = data['results']
            
            # 第一行：原始车牌图像
            axes[0, idx].imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
            axes[0, idx].set_title(f'plate_{plate_num}.jpeg\n(原始图像)', fontsize=12)
            axes[0, idx].axis('off')
            
            # 第二行：分割结果
            vis_img = plate_img.copy()
            for i, (x, y, w, h) in enumerate(data['digit_regions']):
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis_img, str(i + 1), (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            axes[1, idx].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            axes[1, idx].set_title(f'分割结果\n(检测到{len(digits)}个数字)', fontsize=12)
            axes[1, idx].axis('off')
            
            # 第三行：识别结果
            if digits:
                # 创建识别结果图像
                result_img = np.ones((100, 300, 3), dtype=np.uint8) * 255
                result_text = ""
                for i, (digit, conf) in enumerate(results):
                    result_text += f"{digit}"
                   
                    result_text += " "
                
                # 在图像上绘制文本
                cv2.putText(result_img, result_text, (10, 50),  # X从10→30, Y从50→90
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)  # 字体缩放从1→2.0, 线宽从2→3
                
                axes[2, idx].imshow(result_img)
                axes[2, idx].set_title(f'识别结果\n{result_text.strip()}', fontsize=12)
                axes[2, idx].axis('off')
            else:
                axes[2, idx].text(0.5, 0.5, '未检测到数字', 
                                ha='center', va='center', transform=axes[2, idx].transAxes,
                                fontsize=12, color='red')
                axes[2, idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细结果
        self._print_detailed_results(results_data)
    
    def _print_detailed_results(self, results_data):
        """打印详细的识别结果"""
        print("\n" + "="*60)
        print("车牌识别详细结果")
        print("="*60)
        
        for plate_num, data in results_data.items():
            print(f"\n车牌 {plate_num} ({data['plate_file']}):")
            print("-" * 40)
            
            if data['results']:
                # 显示识别到的数字
                digits_str = ""
                for i, (digit, conf) in enumerate(data['results']):
                    digits_str += str(digit)
                    if conf < 0.5:
                        digits_str += "?"
                    digits_str += " "
                
                print(f"识别结果: {digits_str.strip()}")
                print("详细分析:")
                
                for i, (digit, conf) in enumerate(data['results']):
                    confidence_level = "高" if conf > 0.8 else "中" if conf > 0.6 else "低"
                    print(f"  数字 {i+1}: {digit} (置信度: {conf:.3f}, {confidence_level})")
            else:
                print("未检测到任何数字")
        
        # 统计信息
        total_digits = sum(len(data['results']) for data in results_data.values())
        high_conf_digits = sum(1 for data in results_data.values() 
                              for _, conf in data['results'] if conf > 0.8)
        medium_conf_digits = sum(1 for data in results_data.values() 
                                for _, conf in data['results'] if 0.6 < conf <= 0.8)
        low_conf_digits = sum(1 for data in results_data.values() 
                             for _, conf in data['results'] if conf <= 0.6)
        
        print(f"\n统计信息:")
        print(f"总车牌数: {len(results_data)}")
        print(f"检测到的总数字数: {total_digits}")
        print(f"高置信度数字: {high_conf_digits} ({high_conf_digits/total_digits*100:.1f}%)" if total_digits > 0 else "高置信度数字: 0")
        print(f"中等置信度数字: {medium_conf_digits} ({medium_conf_digits/total_digits*100:.1f}%)" if total_digits > 0 else "中等置信度数字: 0")
        print(f"低置信度数字: {low_conf_digits} ({low_conf_digits/total_digits*100:.1f}%)" if total_digits > 0 else "低置信度数字: 0")

    def visualize_plate_preprocessing(self, plate_range=(0, 9)):
        """可视化车牌预处理效果"""
        print("\n" + "="*80)
        print("开始可视化车牌预处理效果")
        print("="*80)
    
        num_plates = plate_range[1] - plate_range[0] + 1
        fig, axes = plt.subplots(num_plates, 2, figsize=(15, 3 * num_plates))
        fig.suptitle('车牌图像预处理效果展示', fontsize=20, fontweight='bold')
    
        for i, plate_num in enumerate(range(plate_range[0], plate_range[1] + 1)):
            plate_file = f"plate_{plate_num}.jpeg"
            plate_path = os.path.join(self.plates_dir, plate_file)
        
            if not os.path.exists(plate_path):
                print(f"警告: 未找到文件 {plate_file}")
                continue
                
            plate_img = cv2.imread(plate_path)
            if plate_img is None:
                print(f"错误: 无法加载图像 {plate_file}")
                continue
        
        # 预处理流程保持不变
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY_INV, 11, 2)
            kernel = np.ones((2, 2), np.uint8)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 处理单行布局情况
            if num_plates == 1:
                ax_row = axes
            else:
                ax_row = axes[i]
        
        # 显示原始图像（第一列）
            ax_row[0].imshow(cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB))
            ax_row[0].set_title(f'原始图像: {plate_file}', fontsize=12)
            ax_row[0].axis('off')
        
        # 显示预处理图像（第二列）
            ax_row[1].imshow(closed, cmap='gray')
            ax_row[1].set_title('预处理后效果', fontsize=12)
            ax_row[1].axis('off')
    
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)  # 为主标题留出空间
        plt.show()

    def _detect_plate_enhanced_contour(self, img):
        """增强的基于轮廓的车牌检测（处理倾斜和透视）"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 多尺度边缘检测
        edges_list = []
        for sigma in [0.5, 1.0, 1.5]:
            blurred = cv2.GaussianBlur(gray, (5, 5), sigma)
            edges = cv2.Canny(blurred, 50, 150)
            edges_list.append(edges)
        
        # 合并不同尺度的边缘
        combined_edges = np.zeros_like(edges_list[0])
        for edges in edges_list:
            combined_edges = cv2.bitwise_or(combined_edges, edges)
        
        # 使用不同形状的核进行形态学操作
        kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)),   # 标准矩形
            cv2.getStructuringElement(cv2.MORPH_RECT, (30, 6)),   # 更大矩形
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 8)) # 椭圆形
        ]
        
        best_candidates = []
        
        for kernel in kernels:
            closed = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
            opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
            
            # 查找轮廓
            contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 使用最小外接矩形处理倾斜
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = box.astype(np.int32)
                
                # 计算矩形的宽高和角度
                width = int(rect[1][0])
                height = int(rect[1][1])
                angle = rect[2]
                
                # 确保width > height（车牌是横向的）
                if width < height:
                    width, height = height, width
                    angle += 90
                
                aspect_ratio = width / height if height > 0 else 0
                area = cv2.contourArea(contour)
                
                # 车牌特征判断（更宽松的条件）
                if (1.8 < aspect_ratio < 6.5 and  # 放宽长宽比范围
                    area > 800 and area < img.shape[0] * img.shape[1] * 0.4 and
                    width > 60 and height > 15 and
                    abs(angle) < 45):  # 允许45度以内的倾斜
                    
                    best_candidates.append((contour, rect, area, aspect_ratio))
        
        if not best_candidates:
            return None
        
        # 选择最佳候选区域（综合面积和长宽比）
        best_candidate = max(best_candidates, key=lambda x: x[2] * (1 + 1/max(abs(x[3] - 3.5), 0.1)))
        contour, rect, _, _ = best_candidate
        
        # 透视变换矫正倾斜的车牌
        return self._correct_perspective(img, rect)
    
    def _correct_perspective(self, img, rect):
        """透视变换矫正倾斜车牌"""
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        
        # 计算矩形的宽高
        width = int(rect[1][0])
        height = int(rect[1][1])
        
        # 确保width > height
        if width < height:
            width, height = height, width
        
        # 排序四个角点：左上、右上、右下、左下
        box = box[np.argsort(box[:, 0])]  # 按x坐标排序
        left_points = box[:2]
        right_points = box[2:]
        
        # 左侧点按y坐标排序
        left_points = left_points[np.argsort(left_points[:, 1])]
        # 右侧点按y坐标排序
        right_points = right_points[np.argsort(right_points[:, 1])]
        
        src_points = np.array([
            left_points[0],   # 左上
            right_points[0],  # 右上
            right_points[1],  # 右下
            left_points[1]    # 左下
        ], dtype=np.float32)
        
        # 目标矩形（标准车牌比例）
        dst_points = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # 应用透视变换
        corrected = cv2.warpPerspective(img, matrix, (width, height))
        
        return corrected

    def _detect_plate_color_enhanced(self, img):
        """增强的颜色检测（处理光照变化）"""
        # 直方图均衡化改善光照
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(lab[:,:,0])
        img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        hsv = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2HSV)
        
        # 扩展颜色范围以适应不同光照条件
        color_ranges = [
            # 蓝色车牌（多个范围）
            ([100, 30, 30], [140, 255, 255]),
            ([90, 40, 40], [130, 255, 200]),
            # 白色车牌
            ([0, 0, 180], [180, 40, 255]),
            ([0, 0, 160], [180, 50, 255]),
            # 黄色车牌
            ([10, 40, 40], [40, 255, 255]),
            ([15, 50, 50], [35, 255, 200]),
            # 绿色车牌（新能源）
            ([40, 40, 40], [80, 255, 255])
        ]
        
        combined_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        
        for lower, upper in color_ranges:
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            combined_mask = cv2.bitwise_or(combined_mask, mask)
        
        # 形态学操作优化
        kernel_rect = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))
        kernel_ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 8))
        
        # 多次形态学操作
        processed = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_rect)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel_ellipse)
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel_rect)
        
        # 查找轮廓并选择最佳候选
        contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plate_candidates = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            width = max(rect[1])
            height = min(rect[1])
            aspect_ratio = width / height if height > 0 else 0
            area = cv2.contourArea(contour)
            
            if (1.8 < aspect_ratio < 6.0 and 
                area > 600 and area < img.shape[0] * img.shape[1] * 0.3 and
                width > 50 and height > 12):
                plate_candidates.append((contour, rect, area))
        
        if plate_candidates:
            # 选择面积最大的候选
            best_contour, best_rect, _ = max(plate_candidates, key=lambda x: x[2])
            return self._correct_perspective(img, best_rect)
        
        return None
    
    def detect_and_extract_plate(self, img_path):
        """优化的车牌检测和提取方法"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"错误: 无法加载图像 {img_path}")
            return None, None
        
        # 尝试多种检测方法，按优先级排序
        plate_region = self._detect_plate_enhanced_contour(img)
        if plate_region is None:
            plate_region = self._detect_plate_color_enhanced(img)
        if plate_region is None:
            plate_region = self._detect_plate_morphology(img)
        
        return plate_region, img


    def _detect_plate_cascade(self, img):
        """方法1: 使用Haar级联分类器检测车牌"""
        try:
            # 尝试加载预训练的车牌检测器
            plate_cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            plates = plate_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(plates) > 0:
                # 选择最大的检测区域
                x, y, w, h = max(plates, key=lambda p: p[2] * p[3])
                return img[y:y+h, x:x+w]
        except:
            pass
        return None

    def _detect_plate_contour(self, img):
        """方法2: 基于轮廓的车牌检测"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 边缘检测
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 200)
        
        # 形态学操作连接断开的边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 筛选车牌候选区域
        plate_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            # 车牌特征：长宽比2-6，面积适中
            if (2.0 < aspect_ratio < 6.0 and 
                area > 1000 and area < img.shape[0] * img.shape[1] * 0.3 and
                w > 80 and h > 20):
                plate_candidates.append((x, y, w, h, area))
        
        if plate_candidates:
            # 选择面积最大的候选区域
            x, y, w, h, _ = max(plate_candidates, key=lambda p: p[4])
            return img[y:y+h, x:x+w]
        
        return None

    def _detect_plate_color_edge(self, img):
        """方法3: 基于颜色和边缘特征的车牌检测"""
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 定义蓝色车牌的HSV范围
        lower_blue = np.array([100, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        # 定义白色/黄色车牌的HSV范围
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        lower_yellow = np.array([15, 50, 50])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # 合并颜色掩码
        color_mask = cv2.bitwise_or(blue_mask, cv2.bitwise_or(white_mask, yellow_mask))
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        plate_candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h
            area = cv2.contourArea(contour)
            
            if (2.0 < aspect_ratio < 6.0 and 
                area > 500 and area < img.shape[0] * img.shape[1] * 0.2 and
                w > 60 and h > 15):
                plate_candidates.append((x, y, w, h, area))
        
        if plate_candidates:
            x, y, w, h, _ = max(plate_candidates, key=lambda p: p[4])
            return img[y:y+h, x:x+w]
        
        return None

    def process_real_plates(self, plate_range=(1, 2)):
        """处理真实车牌图像"""
        print("\n" + "="*80)
        print("开始处理真实车牌图像")
        print("="*80)
        
        results_data = {}
        
        for i in range(plate_range[0], plate_range[1] + 1):
            plate_file = f"real_{i}.jpg"
            plate_path = os.path.join(self.plates_dir, plate_file)
            
            if not os.path.exists(plate_path):
                print(f"警告: 未找到文件 {plate_file}")
                continue
            
            print(f"处理真实车牌: {plate_file}")
            
            # 1. 检测并提取车牌区域
            plate_region, original_img = self.detect_and_extract_plate(plate_path)
            
            if plate_region is None:
                print(f"警告: 未能检测到车牌区域 - {plate_file}")
                continue
            
            # 2. 分割和识别数字
            digits, digit_regions = self.segment_digits(plate_region)
            recognition_results = []
            
            for digit_img in digits:
                digit, confidence = self.recognize_digit(digit_img)
                recognition_results.append((digit, confidence))
            
            results_data[i] = {
                'original_img': original_img,
                'plate_region': plate_region,
                'digits': digits,
                'digit_regions': digit_regions,
                'results': recognition_results,
                'plate_file': plate_file
            }
        
        if results_data:
            self._visualize_real_plate_results(results_data)
        else:
            print("错误: 没有成功处理任何真实车牌图像")

    def _visualize_real_plate_results(self, results_data):
        """可视化真实车牌识别结果"""
        import matplotlib.pyplot as plt
        
        num_plates = len(results_data)
        fig, axes = plt.subplots(4, num_plates, figsize=(6*num_plates, 16))
        fig.suptitle('真实车牌识别结果可视化', fontsize=20, fontweight='bold')
        
        for idx, (plate_num, data) in enumerate(results_data.items()):
            original_img = data['original_img']
            plate_region = data['plate_region']
            digits = data['digits']
            results = data['results']
            
            # 处理单个车牌的情况
            if num_plates == 1:
                ax_col = axes
            else:
                ax_col = axes[:, idx]
            
            # 第一行：原始图像
            ax_col[0].imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
            ax_col[0].set_title(f'{data["plate_file"]}\n(原始图像)', fontsize=12)
            ax_col[0].axis('off')
            
            # 第二行：提取的车牌区域
            ax_col[1].imshow(cv2.cvtColor(plate_region, cv2.COLOR_BGR2RGB))
            ax_col[1].set_title('提取的车牌区域', fontsize=12)
            ax_col[1].axis('off')
            
            # 第三行：字符分割结果
            vis_img = plate_region.copy()
            for i, (x, y, w, h) in enumerate(data['digit_regions']):
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(vis_img, str(i + 1), (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            ax_col[2].imshow(cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB))
            ax_col[2].set_title(f'字符分割\n(检测到{len(digits)}个字符)', fontsize=12)
            ax_col[2].axis('off')
            
            # 第四行：识别结果
            if results:
                result_img = np.ones((60, 200, 3), dtype=np.uint8) * 255
                result_text = "".join([str(digit) for digit, _ in results])
                
                cv2.putText(result_img, result_text, (10, 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
                
                ax_col[3].imshow(result_img)
                ax_col[3].set_title(f'识别结果: {result_text}', fontsize=12)
                ax_col[3].axis('off')
            else:
                ax_col[3].text(0.5, 0.5, '未识别到字符', 
                              ha='center', va='center', transform=ax_col[3].transAxes,
                              fontsize=12, color='red')
                ax_col[3].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # 打印详细结果
        self._print_real_plate_results(results_data)

    def _print_real_plate_results(self, results_data):
        """打印真实车牌识别的详细结果"""
        print("\n" + "="*60)
        print("真实车牌识别详细结果")
        print("="*60)
        
        for plate_num, data in results_data.items():
            print(f"\n真实车牌 {plate_num} ({data['plate_file']}):")
            print("-" * 40)
            
            if data['results']:
                digits_str = "".join([str(digit) for digit, _ in data['results']])
                print(f"识别结果: {digits_str}")
                print("详细分析:")
                
                for i, (digit, conf) in enumerate(data['results']):
                    confidence_level = "高" if conf > 0.8 else "中" if conf > 0.6 else "低"
                    print(f"  字符 {i+1}: {digit} (置信度: {conf:.3f}, {confidence_level})")
            else:
                print("未检测到任何字符")
        
        # 统计信息
        total_chars = sum(len(data['results']) for data in results_data.values())
        high_conf = sum(1 for data in results_data.values() 
                       for _, conf in data['results'] if conf > 0.8)
        
        print(f"\n统计信息:")
        print(f"处理的真实车牌数: {len(results_data)}")
        print(f"检测到的总字符数: {total_chars}")
        if total_chars > 0:
            print(f"高置信度字符: {high_conf} ({high_conf/total_chars*100:.1f}%)")
    
    
# 使用示例
if __name__ == "__main__":
    # 定义路径
    template_dir = r"data/char_tmpl"
    plates_dir = r"data/plates"
    
    # 创建并运行识别器
    recognizer = DigitRecognizer(template_dir, plates_dir)
    
    # 加载模板
    recognizer.load_templates()
    
    # 训练模型
    if recognizer.train():
        # 处理真实车牌
        recognizer.process_real_plates(plate_range=(1, 2))
        # 可视化所有数字的识别过程
        recognizer.visualize_all_digits_recognition()
        
        # 可视化不同数字的特征对比
        recognizer.visualize_feature_comparison()
        
        # 可视化车牌预处理效果
        recognizer.visualize_plate_preprocessing(plate_range=(0, 9))

        recognizer.visualize_segmentation_debug(plate_range=(0, 5))
        
        # 专门可视化plate_0.jpeg到plate_9.jpeg的识别结果
        print("\n" + "="*80)
        print("开始处理plate_0.jpeg到plate_9.jpeg的识别结果可视化")
        print("="*80)
        
        recognizer.visualize_specific_plates(plate_range=(0, 9))
        
        # 测试3和8的识别
        print("\n特别测试数字3和8:")
        if 3 in recognizer.templates and 8 in recognizer.templates:
            digit3 = recognizer.templates[3]
            digit8 = recognizer.templates[8]
            
            pred3, conf3 = recognizer.recognize_digit(digit3)
            pred8, conf8 = recognizer.recognize_digit(digit8)
            
            print(f"数字3识别: {pred3} (置信度: {conf3:.3f})")
            print(f"数字8识别: {pred8} (置信度: {conf8:.3f})")
        else:
            print("警告: 缺少3或8的模板图像，无法进行特别测试")
    else:
        print("模型训练失败")


def segment_digits(self, plate_img):
    """车牌字符分割主函数：预处理 + 连通域分析"""
    # 1. 灰度化与二值化预处理
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
    preprocessed = self.preprocess_image(gray, target_size=None)  # 包含高斯模糊与自适应阈值
    _, binary = cv2.threshold(preprocessed, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 2. 调用优化版连通域分割算法
    digit_regions = self._segment_characters_clean_optimized(binary)
    
    # 3. 提取并预处理字符区域
    digits = []
    for x, y, w_c, h_c in digit_regions:
        digit_img = gray[y:y+h_c, x:x+w_c]
        preprocessed_digit = self.preprocess_image(digit_img)  # 标准化为28x28
        digits.append(preprocessed_digit)
    return digits, digit_regions

