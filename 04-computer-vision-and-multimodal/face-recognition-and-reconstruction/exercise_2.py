import os
import re
import cv2
import numpy as np
import exercise_1
# 本文件实现一个基于 PyQt 的人脸识别与重构 GUI：
# - 训练阶段：读取训练矩阵，计算特征脸与主成分系数
# - 交互功能：拍摄/识别/随机遮挡/重构，支持鼠标框选裁剪
# - 验证展示：启动时弹窗显示验证失败样本并保存截图，主界面左下角展示验证准确率
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout, QDialog, QGridLayout
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt5.QtCore import QTimer, QPoint, QRect

def np_gray_to_qpix(arr, width, height):
    """将灰度 numpy 数组转为 QPixmap，用于标签显示。
    使用 QImage.Format_Grayscale8，确保灰度图像在 Qt 中正确显示。"""
    # 输入：arr 为灰度二维数组，width/height 为像素尺寸（列数/行数）
    # 过程：借助 QImage 构造 8 位灰度图，再转为 QPixmap 以便 QLabel 显示
    img_u8 = np.asarray(arr, dtype=np.uint8)
    bytes_per_line = width
    qimg = QImage(img_u8.data, width, height, bytes_per_line, QImage.Format_Grayscale8).copy()
    qp = QPixmap.fromImage(qimg)
    return qp

class CroppableLabel(QLabel):
    # 可框选的图像标签：支持鼠标拖拽绘制绿色矩形选择框，用于后续裁剪
    def __init__(self, parent=None):
        super().__init__(parent)
        self._selecting = False
        self._start = None
        self._end = None
        return None

    def mousePressEvent(self, event):
        # 鼠标按下：开始选择，记录起点坐标
        self._selecting = True
        self._start = event.pos()
        self._end = event.pos()
        self.update()
        return None

    def mouseMoveEvent(self, event):
        # 鼠标移动：更新终点坐标，刷新绘制
        if self._selecting:
            self._end = event.pos()
            self.update()
        return None

    def mouseReleaseEvent(self, event):
        # 鼠标释放：结束选择，固定终点坐标
        self._selecting = False
        self._end = event.pos()
        self.update()
        return None

    def paintEvent(self, event):
        # 在已有图像之上叠加绘制绿色选择框（不修改底图）
        super().paintEvent(event)
        if self._start is not None and self._end is not None:
            painter = QPainter(self)
            pen = QPen(QColor(0, 255, 0))
            pen.setWidth(2)
            painter.setPen(pen)
            rect = QRect(self._start, self._end).normalized()
            painter.drawRect(rect)
            painter.end()
        return None

    def clear_selection(self):
        # 清空选择框
        self._start = None
        self._end = None
        self.update()
        return None

    def has_selection(self):
        # 判断当前是否存在有效选择框
        return (self._start is not None and self._end is not None)

    def get_selection_rect(self):
        # 返回当前选择框的几何数据 (x, y, w, h)，若无选择则返回 None
        if not self.has_selection():
            return None
        rect = QRect(self._start, self._end).normalized()
        return (rect.x(), rect.y(), rect.width(), rect.height())

class FaceGUI(QWidget):
    def __init__(self, start_preview=True):
        """主界面初始化：
        - 加载训练数据并计算特征脸空间（mean_face、W、S_train）
        - 构建姓名列表 names 与摄像头实时预览
        - 创建两个 QLabel（图像与姓名）与四个按钮（拍摄/识别/随机遮挡/重构）
        - 预览通过 QTimer 定时更新；点击“拍摄”后冻结当前帧"""
        super().__init__()
        self.setWindowTitle('人脸识别与重构')
        # 训练阶段：加载与PCA
        # 返回：X_flat(样本×像素)、n(样本数)、n_rows(高=480)、n_cols(宽=640)
        X_flat, n, self.n_rows, self.n_cols = exercise_1.load_data()
        self.mean_face, self.W, self.S_train, self.evals, self.ratio = exercise_1.compute_eigenfaces(X_flat)
        self.names = self._load_names(n)
        # 状态变量
        self.current_image = None
        self.masked_image = None
        self.reconstructed_image = None
        self.cropped_image = None
        self._frozen = False
        # 摄像头与实时预览（可延迟启动）
        # 说明：为满足“启动前先展示验证失败弹窗”的流程，预览可延迟到主界面显示后再启动
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_frame)
        if start_preview:
            self._start_preview_internal()
        # 界面控件
        self.img_label = CroppableLabel()
        # 固定显示尺寸为训练数据尺寸（宽 n_cols，高 n_rows），所有图像显示均按此缩放
        self.img_label.setFixedSize(self.n_cols, self.n_rows)
        self.name_label = QLabel('')
        self.acc_label = QLabel('')
        # 初始显示实时预览（若失败则显示黑屏）
        black = np.zeros((self.n_rows, self.n_cols), dtype=np.uint8)
        self.img_label.setPixmap(np_gray_to_qpix(black, self.n_cols, self.n_rows))
        # 按钮
        self.btn_take = QPushButton('拍摄')
        self.btn_rec = QPushButton('识别')
        self.btn_mask = QPushButton('随机遮挡')
        self.btn_crop = QPushButton('裁剪')
        self.btn_recon = QPushButton('重构')
        # 事件绑定
        self.btn_take.clicked.connect(self.on_take_photo)
        self.btn_rec.clicked.connect(self.on_recognize)
        self.btn_mask.clicked.connect(self.on_random_mask)
        self.btn_recon.clicked.connect(self.on_reconstruct)
        self.btn_crop.clicked.connect(self.on_crop)
        # 布局
        v = QVBoxLayout()
        v.addWidget(self.img_label)
        v.addWidget(self.name_label)
        h = QHBoxLayout()
        # 左下角显示验证准确率（文本标签），右侧依次为各功能按钮
        h.addWidget(self.acc_label)
        h.addStretch()
        h.addWidget(self.btn_take)
        h.addWidget(self.btn_rec)
        h.addWidget(self.btn_mask)
        h.addWidget(self.btn_crop)
        h.addWidget(self.btn_recon)
        v.addLayout(h)
        self.setLayout(v)
        return None

    def _update_frame(self):
        """从摄像头抓取一帧进行实时预览；若被冻结则跳过更新。"""
        # 说明：将 BGR 帧转灰度并缩放到固定显示尺寸，更新标签显示与当前图像缓存
        if self._frozen:
            return None
        if not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (self.n_cols, self.n_rows))
        self.img_label.setPixmap(np_gray_to_qpix(gray, self.n_cols, self.n_rows))
        self.current_image = gray.astype(np.float64)
        return None

    def on_take_photo(self):
        """点击“拍摄”后冻结当前帧为待识别图像。"""
        # 行为：停止定时器，锁定 current_image；便于后续裁剪/遮挡/识别
        self._frozen = True
        self.timer.stop()
        self.name_label.setText('')
        return None

    def on_recognize(self):
        """特征脸识别按钮：
        - 去中心化：x_c = x - mean
        - 投影：s = x_c @ W
        - 最近邻：与 S_train 逐样本欧氏距离；阈值法判断 Unknown
        - 显示人名或 Unknown"""
        # 支持裁剪后识别：
        # - 若已裁剪，则先显示为 480×640（竖幅）以满足展示需求
        # - 模型实际识别仍回到训练维度（n_rows×n_cols），保证特征空间一致
        if self.cropped_image is not None:
            img_src = self.cropped_image.astype(np.float64)
            img_480x640 = cv2.resize(img_src, (480, 640))
            disp = cv2.resize(np.clip(img_480x640, 0, 255).astype(np.uint8), (self.n_cols, self.n_rows))
            self.img_label.setPixmap(np_gray_to_qpix(disp, self.n_cols, self.n_rows))
            x_model = cv2.resize(img_480x640, (self.n_cols, self.n_rows)).flatten()
        else:
            if self.current_image is None:
                return None
            x_model = self.current_image.flatten()
        # 投影到特征脸空间并做最近邻识别（欧氏距离），均值±标准差为阈值
        x_c = x_model - self.mean_face
        s = x_c @ self.W
        dists = np.linalg.norm(self.S_train - s.reshape(1, -1), axis=1)
        idx = int(np.argmin(dists))
        min_dist = float(dists[idx])
        tau = float(dists.mean() + dists.std())
        name = self.names[idx] if idx < len(self.names) else f'person_{idx+1}'
        text = name if min_dist < tau else 'Unknown'
        self.name_label.setText(text)
        return None

    def on_random_mask(self):
        """随机遮挡当前图像：优先在人脸区域绘制一个纯黑矩形并更新显示。"""
        base = self.cropped_image if self.cropped_image is not None else self.current_image
        if base is None:
            return None
        img = base.copy()
        h, w = img.shape
        img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        try:
            cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(img_u8, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        except Exception:
            faces = ()
        if isinstance(faces, (list, tuple)) and len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            cv2.rectangle(img, (int(x), int(y)), (int(x + fw), int(y + fh)), 0, thickness=-1)
        else:
            rng = np.random.default_rng()
            rw = int(rng.uniform(w * 0.15, w * 0.35))
            rh = int(rng.uniform(h * 0.15, h * 0.35))
            x0 = int(rng.uniform(0, w - rw))
            y0 = int(rng.uniform(0, h - rh))
            cv2.rectangle(img, (x0, y0), (x0 + rw, y0 + rh), 0, thickness=-1)
        self.masked_image = img
        disp = cv2.resize(np.clip(img, 0, 255).astype(np.uint8), (self.n_cols, self.n_rows))
        self.img_label.setPixmap(np_gray_to_qpix(disp, self.n_cols, self.n_rows))
        self.name_label.setText('')
        return None

    def on_reconstruct(self):
        """重构受损图像：
        - 去中心化与投影
        - 调用 exercise_1.reconstruct_with_q 使用前 q 主成分重构
        - 弹窗并列展示原始/遮挡/重构三幅图"""
        # 输入：masked_image（可能来自裁剪图的遮挡结果）
        # 为适配训练维度，重构前将遮挡图缩放到 (n_cols, n_rows)
        if self.masked_image is None:
            return None
        src_mask = self.masked_image
        if src_mask.shape != (self.n_rows, self.n_cols):
            src_mask = cv2.resize(src_mask, (self.n_cols, self.n_rows))
        x = src_mask.flatten()
        x_c = x - self.mean_face
        s = x_c @ self.W
        q = min(15, self.W.shape[1])
        X_hat = exercise_1.reconstruct_with_q(self.mean_face, self.W, s.reshape(1, -1), q)
        x_hat = X_hat[0, :].reshape(self.n_rows, self.n_cols)
        self.reconstructed_image = x_hat
        # 弹窗展示三图对比
        win = QWidget()
        win.setWindowTitle('原始/遮挡/重构 对比')
        lbl_o = QLabel()
        lbl_m = QLabel()
        lbl_r = QLabel()
        # 原始优先展示裁剪图，其次为拍摄帧；遮挡显示最近一次遮挡结果；重构显示 x_hat
        orig = self.cropped_image if self.cropped_image is not None else self.current_image
        if orig is None:
            orig = np.zeros((self.n_rows, self.n_cols), dtype=np.uint8)
        disp_o = cv2.resize(np.clip(orig, 0, 255).astype(np.uint8), (self.n_cols, self.n_rows))
        disp_m = cv2.resize(np.clip(self.masked_image, 0, 255).astype(np.uint8), (self.n_cols, self.n_rows))
        lbl_o.setPixmap(np_gray_to_qpix(disp_o, self.n_cols, self.n_rows))
        lbl_m.setPixmap(np_gray_to_qpix(disp_m, self.n_cols, self.n_rows))
        lbl_r.setPixmap(np_gray_to_qpix(np.clip(x_hat, 0, 255), self.n_cols, self.n_rows))
        layout = QHBoxLayout()
        layout.addWidget(lbl_o)
        layout.addWidget(lbl_m)
        layout.addWidget(lbl_r)
        win.setLayout(layout)
        win.show()
        # 为了保持子窗体引用，防止被回收
        self._compare_window = win
        return None

    def on_crop(self):
        # 根据当前选择框裁剪 current_image，显示结果并停止预览
        if self.current_image is None:
            return None
        rect = self.img_label.get_selection_rect()
        if rect is None:
            return None
        x, y, w, h = rect
        if w <= 1 or h <= 1:
            return None
        x2 = min(self.n_cols, x + w)
        y2 = min(self.n_rows, y + h)
        x = max(0, x)
        y = max(0, y)
        crop = self.current_image[y:y2, x:x2]
        if crop.size == 0:
            return None
        self.cropped_image = crop
        self._frozen = True
        if self.timer.isActive():
            self.timer.stop()
        disp = cv2.resize(np.clip(crop, 0, 255).astype(np.uint8), (self.n_cols, self.n_rows))
        self.img_label.setPixmap(np_gray_to_qpix(disp, self.n_cols, self.n_rows))
        self.img_label.clear_selection()
        self.name_label.setText('')
        return None

    def _start_preview_internal(self):
        """启动摄像头与预览定时器。"""
        # 说明：在延迟启动场景下由 start_preview 调用；若设备可用则以 ~30fps 刷新
        self.cap = cv2.VideoCapture(0)
        if self.cap is not None and self.cap.isOpened():
            self.timer.start(33)
        return None

    def start_preview(self):
        """暴露的启动预览方法，在延迟启动场景下调用。"""
        # 若尚未创建摄像头句柄，则执行内部启动流程
        if self.cap is None:
            self._start_preview_internal()
        return None

    def _load_names(self, expected_len):
        """从训练目录文件名构建姓名列表。
        文件命名规则：person_1.ext、person_2.ext；提取靠前连续字母作为姓名。"""
        # 返回顺序需与训练矩阵的样本顺序一致；若数量不匹配则回退为 person_i
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        train_dir = os.path.join(base_dir, 'img', 'training')
        if not os.path.isdir(train_dir):
            return [f'person_{i+1}' for i in range(expected_len)]
        files = [f for f in sorted(os.listdir(train_dir)) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        def _key(fname):
            base = os.path.splitext(os.path.basename(fname))[0]
            m = re.match(r'[a-zA-Z]+', base)
            person = (m.group().lower() if m else base.lower())
            return (person, base)
        files.sort(key=_key)
        names = []
        for fname in files:
            base = os.path.splitext(os.path.basename(fname))[0]
            m = re.match(r'[a-zA-Z]+', base)
            person = (m.group().lower() if m else base.lower())
            names.append(person)
        if len(names) != expected_len:
            return [f'person_{i+1}' for i in range(expected_len)]
        return names

    def closeEvent(self, event):
        """关闭窗口时停止定时器并释放摄像头资源。"""
        # 资源管理：确保停止计时器与释放 VideoCapture，避免设备占用
        try:
            if hasattr(self, 'timer') and self.timer.isActive():
                self.timer.stop()
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
        finally:
            event.accept()
        return None

    def _predict_name_from_gray(self, img_gray):
        """对单张灰度图执行特征脸识别并返回预测姓名或 Unknown。"""
        # 与 on_recognize 的核心流程一致，但以传入灰度图为数据源
        x = img_gray.flatten().astype(np.float64)
        x_c = x - self.mean_face
        s = x_c @ self.W
        dists = np.linalg.norm(self.S_train - s.reshape(1, -1), axis=1)
        idx = int(np.argmin(dists))
        tau = float(dists.mean() + dists.std())
        name = self.names[idx] if idx < len(self.names) else f'person_{idx+1}'
        return name if float(dists[idx]) < tau else 'Unknown'

    def _extract_label(self, fname):
        """从验证集文件名提取标签姓名（字母前缀）。"""
        # 例如 sampleuser.jpg → 标签为 'sampleuser'
        base = os.path.splitext(os.path.basename(fname))[0]
        m = re.match(r'[a-zA-Z]+', base)
        label = (m.group().lower() if m else base.lower())
        return label

    def _evaluate_validation_results(self):
        """遍历验证集，返回准确率与失败样本列表。"""
        # 输出：acc（准确率），failures（列表，含文件名/真实标签/预测/图像）
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        val_dir = os.path.join(base_dir, 'img', 'validation')
        if not os.path.isdir(val_dir):
            return (0.0, [])
        files = [f for f in sorted(os.listdir(val_dir)) if f.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        total = 0
        correct = 0
        failures = []
        for fname in files:
            img = cv2.imread(os.path.join(val_dir, fname), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (self.n_cols, self.n_rows))
            pred = self._predict_name_from_gray(img)
            label = self._extract_label(fname)
            total += 1
            if pred == label:
                correct += 1
            else:
                failures.append((fname, label, pred, img))
        acc = (correct / total) if total > 0 else 0.0
        return (acc, failures)

    def evaluate_validation(self):
        """计算验证准确率并显示到左下角标签。"""
        # 若验证集为空则提示；否则以百分比格式展示结果
        acc, _ = self._evaluate_validation_results()
        if acc > 0:
            self.acc_label.setText('验证准确率：{:.2f}%'.format(acc * 100))
        else:
            self.acc_label.setText('验证集为空')
        return None

    def show_validation_failures_popup(self):
        """弹出窗口显示识别失败样本，并保存窗口截图到 src2 目录。"""
        # 展示：以网格布局排列失败样本（灰度图 + 文本标签“标签/预测”），并抓取弹窗截图
        # 保存：validation_failures.png 于 src2 目录，便于回溯与分享
        acc, failures = self._evaluate_validation_results()
        dlg = QDialog(self)
        dlg.setWindowTitle('验证失败样本')
        grid = QGridLayout()
        if len(failures) == 0:
            lbl = QLabel('验证集全部识别正确，准确率：{:.2f}%'.format(acc * 100))
            grid.addWidget(lbl, 0, 0)
        else:
            cols = 3
            r = 0
            c = 0
            for fname, label, pred, img in failures:
                img_label = QLabel()
                img_label.setPixmap(np_gray_to_qpix(img, self.n_cols, self.n_rows))
                txt_label = QLabel('标签：{}，预测：{}'.format(label, pred))
                cell = QVBoxLayout()
                w = QWidget()
                w.setLayout(cell)
                cell.addWidget(img_label)
                cell.addWidget(txt_label)
                grid.addWidget(w, r, c)
                c += 1
                if c >= cols:
                    c = 0
                    r += 1
        dlg.setLayout(grid)
        dlg.show()
        QApplication.processEvents()
        save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'validation_failures.png'))
        try:
            pix = dlg.grab()
            pix.save(save_path)
        except Exception:
            pass
        dlg.exec_()
        return None

def main():
    # 1.创建应用，先显示验证失败弹窗并保存截图
    # 启动顺序：先弹窗（阻塞等待关闭）→ 再展示主界面并启动预览 → 显示验证准确率
    app = QApplication([])
    gui = FaceGUI(start_preview=False)
    gui.show_validation_failures_popup()
    # 2.显示主界面，启动预览并在左下角显示准确率
    gui.show()
    gui.start_preview()
    gui.evaluate_validation()
    app.exec_()
    return None

if __name__ == '__main__':
    # 数据导入、初始变量定义等在 GUI 初始化时完成
    main()
