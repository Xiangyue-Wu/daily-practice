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
        """图像预处理"""
        if target_size is not None:
            img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        img = cv2.GaussianBlur(img, (3, 3), 0)
        
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
        
        kernel = np.ones((2, 2), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
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
        """基于连通域的优化字符分割算法"""
        h, w = binary_img.shape

    # 1. 形态学操作：改用垂直核增强字符连接（适配横向数字）
        kernel = np.ones((3, 1), np.uint8)  # 垂直核更适合连接横向字符
        processed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)

    # 2. 连通域分析：放宽过滤条件
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        255 - processed, connectivity=8
    )
        char_candidates = []
        for i in range(1, num_labels):
            x, y, w_c, h_c, area = stats[i][cv2.CC_STAT_LEFT], stats[i][cv2.CC_STAT_TOP], \
                               stats[i][cv2.CC_STAT_WIDTH], stats[i][cv2.CC_STAT_HEIGHT], stats[i][cv2.CC_STAT_AREA]
            aspect = w_c / h_c if h_c > 0 else 0

        # 放宽阈值：面积、高宽比、高度更宽松
            if (area > h * w * 0.003  # 原0.005 → 更小，保留小面积字符
                and area < h * w * 0.25
                and 0.15 < aspect < 1.2  # 原0.2-1.0 → 范围更大
                and h_c > h * 0.3  # 原0.4 → 降低高度要求
                and w_c > w * 0.015):
                char_candidates.append((x, y, w_c, h_c))
        # 3. 按x排序
        char_candidates.sort(key=lambda x: x[0])

        # 4. 统计主流高度，过滤高度偏小的框
        if len(char_candidates) > 0:
            heights = [h_c for (_, _, _, h_c) in char_candidates]
            median_height = np.median(heights)
            char_candidates = [c for c in char_candidates if c[3] > median_height * 0.6]

        # 5. 合并过近的区域（最多7个字符）
        while len(char_candidates) > 7:
            min_dist = float('inf')
            min_idx = 0
            for i in range(len(char_candidates) - 1):
                dist = char_candidates[i+1][0] - (char_candidates[i][0] + char_candidates[i][2])
                if dist < min_dist and dist < w * 0.05:
                    min_dist = dist
                    min_idx = i
            if min_dist == float('inf'):
                break
                
            x1, y1, w1, h1 = char_candidates[min_idx]
            x2, y2, w2, h2 = char_candidates[min_idx+1]
            x_new = min(x1, x2)
            y_new = min(y1, y2)
            w_new = max(x1 + w1, x2 + w2) - x_new
            h_new = max(y1 + h1, y2 + h2) - y_new
            char_candidates[min_idx] = (x_new, y_new, w_new, h_new)
            del char_candidates[min_idx+1]

        # 6. 分割过宽的区域（至少7个字符）
        while len(char_candidates) < 7 and len(char_candidates) > 0:
            max_w = 0
            max_idx = -1
            for i, (x, y, w_c, h_c) in enumerate(char_candidates):
                if w_c > max_w and w_c > w * 0.1:
                    max_w = w_c
                    max_idx = i
            if max_w < w * 0.15 or max_idx == -1:
                break
                
            x, y, w_c, h_c = char_candidates[max_idx]
            w1 = w_c // 2
            w2 = w_c - w1
            char_candidates[max_idx] = (x, y, w1, h_c)
            char_candidates.insert(max_idx+1, (x + w1, y, w2, h_c))

        # 7. 最终按x排序
        char_candidates.sort(key=lambda x: x[0])
        return char_candidates
    
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
        # 可视化所有数字的识别过程
        recognizer.visualize_all_digits_recognition()
        
        # 可视化不同数字的特征对比
        recognizer.visualize_feature_comparison()
        
        # 可视化车牌预处理效果
        recognizer.visualize_plate_preprocessing(plate_range=(0, 9))
        
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

