from PIL import Image
import numpy as np
import glob
import os

def get_32D_and_labels(path):
    data = [] # 创建一个空列表来存储图片数据
    labels = []  # 创建一个空列表来存储图片标签

    # 遍历图片文件夹，获取图片路径和对应的标签
    image_folder = path  #
    for image_path in glob.glob(os.path.join(image_folder, '*.bmp')):  # 假设图片为bmp格式，如有其他格式请相应修改
        with open(image_path, 'rb') as f:
            # 读取图像文件
            image = Image.open(f)
            # 将彩色图像转换为灰度图像
            gray_image = image.convert('L')

            # 将灰度图像转换为NumPy数组，并重塑为一维数组
            pixel_values = np.array(gray_image).flatten()

            # 将一维数组重塑为32x32的二维数组
            reshaped_image = pixel_values.reshape((32, 32))

            reshaped_image = reshaped_image / 255.0  # 归一化到[0, 1]范围

            data.append(reshaped_image)  # 将归一化后的图片数据添加到列表中

            if '标准数据' in path:
                label = int(list(os.path.basename(image_path))[-5])  # 得到数字标签
                labels.append(label)  # 将标签添加到列表中
            else:
                pass
    return labels,data

def get_kind(array1):
    dist_data= []
    # 遍历标准数据，计算其与测试数据的欧式距离，并用列表记录
    for array2 in stan_data:
        dist = np.linalg.norm(array1 - array2)
        dist_data.append(dist)

    dist_min=min(dist_data) # 得到最小距离的值

    kind_index = dist_data.index(dist_min) #类型在标签列表中的索引==最小距离值在距离列表中的索引
    kind = stan_labels[kind_index]

    return kind


stan_labels,stan_data = get_32D_and_labels('data/标准数据')
test_labels,test_data = get_32D_and_labels('data/测试数据')
#得到标准数字的32维数组列表及对应标签列表、测试数字的32维数组列表及空标签列表，空标签列表将用于记录数字类别

for test_32D in test_data:

    test_kind = get_kind(test_32D)
    test_labels.append(test_kind)

print(stan_labels)
print(test_labels) # 根据我的数据库，打印内容应与stan_labels一致
