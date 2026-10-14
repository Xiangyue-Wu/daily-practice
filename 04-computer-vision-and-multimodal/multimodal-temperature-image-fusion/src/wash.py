import os
import pandas as pd

# 该脚本用于清洗数据集，根据温度值和图像类型过滤无效条目。
# 过滤规则：
# - 对于 normal_ 图像，如果最大温度 > 50，则删除。
# - 对于 val_img_ 和 img_ 图像，如果最大温度 < 50，则删除。
# - 同时确保图像文件存在。

def load_excel_data(file_path):
    """
    加载 Excel 文件数据。

    参数:
    file_path (str): Excel 文件的路径。

    返回:
    pd.DataFrame: 加载的 DataFrame。

    异常:
    如果读取失败，则打印错误并抛出异常。
    """
    try:
        dataframe = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading excel: {e}")
        raise
    return dataframe

def filter_invalid_entries(dataframe, base_directory):
    """
    过滤无效数据条目，根据温度和图像存在性。

    参数:
    dataframe (pd.DataFrame): 输入 DataFrame。
    base_directory (str): 图像文件的基本目录。

    返回:
    pd.DataFrame: 过滤后的 DataFrame。
    """
    clean_indices = []  # 存储有效条目的索引
    for idx, row in dataframe.iterrows():
        image_identifier = row['对应图像']  # 获取图像标识
        temperatures = [row['温度1'], row['温度2'], row['温度3']]  # 获取温度值
        max_temperature = max(temperatures)  # 计算最大温度
        
        # 根据图像类型确定路径并应用过滤规则
        if 'normal_' in image_identifier:
            image_location = os.path.join(base_directory, 'normal', f"{image_identifier}.png")
            if max_temperature > 50:
                continue  # 如果正常图像温度过高，跳过
        elif 'val_img_' in image_identifier:
            image_location = os.path.join(base_directory, 'fire_test', 'images', f"{image_identifier}.jpg")
            if max_temperature < 50:
                continue  # 如果火灾图像温度过低，跳过
        elif 'img_' in image_identifier:
            image_location = os.path.join(base_directory, 'fire', 'images', f"{image_identifier}.jpg")
            if max_temperature < 50:
                continue  # 如果火灾图像温度过低，跳过
        else:
            continue  # 未知类型，跳过
        
        # 检查图像文件是否存在
        if os.path.exists(image_location):
            clean_indices.append(idx)
    
    # 创建过滤后的 DataFrame
    cleaned_dataframe = dataframe.iloc[clean_indices].reset_index(drop=True)
    return cleaned_dataframe

def save_filtered_data(cleaned_dataframe, output_file_path):
    """
    保存过滤后的 DataFrame 到 CSV 文件。
    参数:
    cleaned_dataframe (pd.DataFrame): 要保存的 DataFrame。
    output_file_path (str): 输出 CSV 文件的路径。
    """
    # 重命名列名
    cleaned_dataframe = cleaned_dataframe.rename(columns={
        '组序号': 'order', 
        '温度1': 'temp1', 
        '温度2': 'temp2', 
        '温度3': 'temp3', 
        '数据类型': 'type', 
        '对应图像': 'co_im'
    })
    
    # 保存为CSV文件
    cleaned_dataframe.to_csv(output_file_path, index=False)
    print(f"已保存清洗后的数据到: {output_file_path}")
    return None

def main():
    # 1. 加载Excel数据
    df = load_excel_data(excel_path)
    print(f"原始数据条目数: {len(df)}")
    
    # 2. 清洗数据，过滤无效条目
    cleaned_df = filter_invalid_entries(df, root_dir)
    print(f"清洗后数据条目数: {len(cleaned_df)}")
    print(f"过滤掉的数据条目数: {len(df) - len(cleaned_df)}")
    
    # 3. 保存清洗后的数据
    save_filtered_data(cleaned_df, output_excel_path)
    
    return None

if __name__ == "__main__":
    """
    脚本入口：定义路径并调用主函数。
    """
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    excel_path = os.path.join(project_dir, 'data', 'mn.xlsx')  # 输入 Excel 文件路径
    root_dir = os.path.join(project_dir, 'data', 'dataset')  # 图像根目录
    output_excel_path = os.path.join(project_dir, 'data', 'cleaned_dataset.csv')  # 输出 CSV 文件路径
    
    # 注意：由于保存为CSV文件，建议将输出路径改为.csv扩展名
    # 如果确实需要保存为Excel文件，可以使用 cleaned_dataframe.to_excel(output_file_path, index=False)
    main()
