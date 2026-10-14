"""
数据挖掘实验一：数据预处理与探索性分析
包含数据加载、清洗、可视化、特征工程等完整流程
数据集：UCI Online Retail II
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder

sns.set_style('whitegrid')  # 设置seaborn绘图风格为白色网格
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi']
plt.rcParams['axes.unicode_minus'] = False


def load_data(file_path, sample_size):
    """
    加载数据文件，支持多种编码格式
    
    参数:
    file_path: 数据文件路径
    sample_size: 采样大小（读取前n行）
    
    返回:
    df: 加载的DataFrame
    """
    # 尝试多种编码格式读取CSV文件
    encodings = ['utf-8', 'latin1']
    last_error = None
    for encoding in encodings:
        try:
            # 尝试用当前编码读取指定行数的数据
            df = pd.read_csv(file_path, encoding=encoding, nrows=sample_size)
            print(f'成功使用编码 {encoding} 读取数据。')
            break
        except Exception as error:
            # 记录错误，继续尝试下一个编码
            last_error = error
            df = None
    if df is None:
        # 所有编码尝试都失败，抛出异常
        raise ValueError(f'数据读取失败: {last_error}')

    # 重命名列名，统一字段名称
    rename_map = {
        'Customer ID': 'CustomerID',  # 统一客户ID列名
        'Price': 'UnitPrice'           # 统一单价列名
    }
    df = df.rename(columns=rename_map)

    # 打印数据预览信息
    print('\n前5行数据:')
    print(df.head())
    print('\n后5行数据:')
    print(df.tail())
    print(f'\n数据形状: {df.shape}')
    return df


def data_overview(df):
    """
    数据概览分析，显示数据信息和统计特征
    
    参数:
    df: 输入的DataFrame
    
    返回:
    None
    """
    print('\n数据概览信息:')
    df.info()  # 显示数据基本信息（列名、数据类型、非空值数量等）
    print('\n数值型统计特征:')
    print(df.describe())  # 显示数值列的统计摘要（计数、均值、标准差、分位数等）
    return None
    
# 3 ===========================
def handle_missing_values(df):
    """
    处理缺失值，分析并删除关键字段的缺失值
    
    参数:
    df: 输入的DataFrame
    
    返回:
    df: 处理缺失值后的DataFrame
    """
    # 计算各列的缺失值数量和比例
    missing_count = df.isnull().sum()  # 统计每列缺失值数量
    missing_ratio = (missing_count / len(df)).sort_values(ascending=False)  # 计算缺失比例并降序排列
    
    # 创建缺失值报告DataFrame
    missing_report = pd.DataFrame({
        '缺失数量': missing_count,  # 缺失数量列
        '缺失比例': missing_ratio   # 缺失比例列
    }).sort_values(by='缺失比例', ascending=False)  # 按缺失比例降序排列
    
    print('\n缺失值报告:')
    print(missing_report)

    # 删除关键字段的缺失值
    if 'Description' in df.columns:
        df = df.dropna(subset=['Description'])  # 删除描述字段为空的行
    if 'CustomerID' in df.columns:
        df = df.dropna(subset=['CustomerID'])  # 删除客户ID为空的行

    print(f'\n缺失值处理后数据形状: {df.shape}')
    return df

# 4 ================
def detect_and_handle_outliers(df):
    """
    检测和处理异常值，使用箱线图法和IQR方法
    
    参数:
    df: 输入的DataFrame
    
    返回:
    df: 处理异常值后的DataFrame
    """
    # 处理数量列的异常值（负值表示退货）
    if 'Quantity' in df.columns:
        df = df[df['Quantity'] > 0]  # 只保留数量大于0的记录（排除退货记录）

    # 绘制单价箱线图观察异常值分布
    plt.figure(figsize=(8, 4))  # 创建图形窗口，设置大小
    sns.boxplot(x=df['UnitPrice'])  # 绘制单价箱线图
    plt.title('UnitPrice 箱线图（异常值观察）')  # 设置标题
    plt.xlabel('UnitPrice')  # 设置x轴标签
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形

    # 使用IQR（四分位距）方法检测和处理异常值
    q1 = df['UnitPrice'].quantile(0.25)  # 计算第一四分位数
    q3 = df['UnitPrice'].quantile(0.75)  # 计算第三四分位数
    iqr = q3 - q1  # 计算四分位距
    
    # 计算异常值边界
    lower_bound = q1 - 1.5 * iqr  # 下边界
    upper_bound = q3 + 1.5 * iqr  # 上边界

    # 识别异常值
    outlier_mask = (df['UnitPrice'] < lower_bound) | (df['UnitPrice'] > upper_bound)  # 创建异常值掩码
    outlier_count = outlier_mask.sum()  # 统计异常值数量
    print(f'\nUnitPrice IQR下界: {lower_bound:.4f}, 上界: {upper_bound:.4f}, 异常值数量: {outlier_count}')

    # 删除异常值
    df = df[~outlier_mask]  # 使用掩码的反向选择，保留非异常值
    print(f'异常值处理后数据形状: {df.shape}')
    return df

# 5.=================
def fix_types_and_clean_text(df):
    """
    修正数据类型和清洗文本字段
    
    参数:
    df: 输入的DataFrame
    
    返回:
    df: 处理后的DataFrame
    """
    # 修正日期字段类型
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')  # 转换为日期时间类型
    df = df.dropna(subset=['InvoiceDate'])  # 删除转换失败的记录

    # 从日期中提取时间特征
    df['Year'] = df['InvoiceDate'].dt.year  # 提取年份
    df['Month'] = df['InvoiceDate'].dt.month  # 提取月份
    df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek  # 提取星期几（0=周一，6=周日）

    # 清洗文本描述字段
    df['Description'] = df['Description'].astype(str).str.strip().str.upper()  # 转换为字符串，去除首尾空格，转大写

    # 创建新特征：Revenue = Quantity * UnitPrice
    if 'Revenue' not in df.columns:
        df['Revenue'] = df['Quantity'] * df['UnitPrice']

    return df


def univariate_analysis(df):
    """
    单变量分析，探索单个变量的分布
    
    参数:
    df: 输入的DataFrame
    
    返回:
    None
    """
    # 绘制数量和单价的分布直方图
    plt.figure(figsize=(12, 5))  # 创建图形窗口
    
    # 子图1：数量分布
    plt.subplot(1, 2, 1)  # 1行2列的第1个子图
    sns.histplot(df['Quantity'], kde=True, bins=20)  # 绘制直方图+KDE密度曲线
    plt.title('Quantity 分布（直方图+KDE）')  # 设置标题
    
    # 子图2：单价分布
    plt.subplot(1, 2, 2)  # 1行2列的第2个子图
    sns.histplot(df['UnitPrice'], kde=True, bins=20)  # 绘制直方图+KDE密度曲线
    plt.title('UnitPrice 分布（直方图+KDE）')  # 设置标题
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形

    # 绘制Top10国家订单量柱状图
    top10_country = df['Country'].value_counts().head(10)  # 获取订单量前10的国家
    plt.figure(figsize=(10, 5))  # 创建新图形窗口
    sns.barplot(x=top10_country.index, y=top10_country.values)  # 绘制柱状图
    plt.title('Top10 国家订单量')  # 设置标题
    plt.xlabel('Country')  # 设置x轴标签
    plt.ylabel('订单量')  # 设置y轴标签
    plt.xticks(rotation=45)  # 旋转x轴标签45度
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形
    return None


def multivariate_analysis(df):
    """
    多变量分析，探索变量间的关系
    
    参数:
    df: 输入的DataFrame
    
    返回:
    None
    """
    # 数值特征相关性热力图
    numeric_df = df.select_dtypes(include=[np.number])  # 选择数值型列
    corr = numeric_df.corr(method='pearson')  # 计算皮尔逊相关系数矩阵

    plt.figure(figsize=(10, 7))  # 创建图形窗口
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm')  # 绘制相关系数热力图
    plt.title('数值特征相关性热力图')  # 设置标题
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形

    # Top10国家的单价箱线图比较
    top_country = df['Country'].value_counts().head(10).index  # 获取订单量前10的国家
    temp_df = df[df['Country'].isin(top_country)]  # 筛选Top10国家的数据
    plt.figure(figsize=(12, 6))  # 创建图形窗口
    sns.boxplot(data=temp_df, x='Country', y='UnitPrice')  # 绘制箱线图
    plt.title('不同国家 UnitPrice 箱线图（Top10）')  # 设置标题
    plt.xticks(rotation=45)  # 旋转x轴标签45度
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形

    # 月度销售额趋势图
    monthly_revenue = df.groupby(df['InvoiceDate'].dt.to_period('M'))['Revenue'].sum()  # 按月聚合销售额
    monthly_revenue.index = monthly_revenue.index.astype(str)  # 将时间索引转换为字符串格式

    plt.figure(figsize=(10, 5))  # 创建图形窗口
    plt.plot(monthly_revenue.index, monthly_revenue.values, marker='o')  # 绘制折线图
    plt.title('按月聚合销售额趋势')  # 设置标题
    plt.xlabel('Month')  # 设置x轴标签
    plt.ylabel('Revenue')  # 设置y轴标签
    plt.xticks(rotation=45)  # 旋转x轴标签45度
    plt.tight_layout()  # 自动调整子图参数
    plt.show()  # 显示图形
    return None


def build_user_item_matrix(df):
    """
    构建用户-物品矩阵，用于推荐系统分析
    
    参数:
    df: 输入的DataFrame
    
    返回:
    user_item_matrix: 用户-物品矩阵（行=用户，列=物品，值=消费额）
    """
    # 创建透视表：用户为行，商品为列，值为消费额
    user_item_matrix = df.pivot_table(
        index='CustomerID',  # 行索引：用户ID
        columns='StockCode',  # 列索引：商品编码
        values='Revenue',  # 值：消费额
        aggfunc='sum',  # 聚合函数：求和
        fill_value=0  # 缺失值填充为0
    )
    print(f'\n用户-物品矩阵形状: {user_item_matrix.shape}')  # 打印矩阵维度
    return user_item_matrix


def calculate_similarity_metrics(user_item_matrix):
    """
    计算用户相似度指标（欧氏距离、余弦相似度、皮尔逊相关系数）
    
    参数:
    user_item_matrix: 用户-物品矩阵
    
    返回:
    top5_users: 前5个最活跃用户
    compare_df: 用户相似度比较DataFrame
    """
    # 找出最活跃的用户（购买商品种类最多的用户）
    active_user_count = (user_item_matrix > 0).sum(axis=1).sort_values(ascending=False)  # 统计每个用户购买的非零商品数
    top5_users = active_user_count.head(5).index.tolist()  # 获取前5个最活跃用户的ID列表

    # 创建子矩阵，只包含前5个活跃用户的数据
    sample_matrix = user_item_matrix.loc[top5_users]
    sample_users = sample_matrix.index.tolist()  # 用户ID列表

    # 初始化相似度结果列表
    euclidean_results = []  # 欧氏距离结果
    cosine_results = []  # 余弦相似度结果
    pearson_results = []  # 皮尔逊相关系数结果

    # 计算用户两两之间的相似度
    for i in range(len(sample_users)):
        for j in range(i + 1, len(sample_users)):
            user_i = sample_users[i]  # 用户i
            user_j = sample_users[j]  # 用户j
            vec_i = sample_matrix.loc[user_i].values  # 用户i的特征向量
            vec_j = sample_matrix.loc[user_j].values  # 用户j的特征向量

            # 计算欧氏距离
            euc_dist = euclidean(vec_i, vec_j)
            
            # 计算余弦相似度
            cos_sim = cosine_similarity([vec_i], [vec_j])[0][0]
            
            # 计算皮尔逊相关系数（处理标准差为0的情况）
            if np.std(vec_i) == 0 or np.std(vec_j) == 0:
                pearson_corr = np.nan  # 标准差为0时，皮尔逊相关系数无法计算
            else:
                pearson_corr, _ = pearsonr(vec_i, vec_j)

            # 将结果添加到相应列表
            euclidean_results.append((user_i, user_j, euc_dist))
            cosine_results.append((user_i, user_j, cos_sim))
            pearson_results.append((user_i, user_j, pearson_corr))

    # 以第一个用户为目标用户，计算与其他用户的相似度
    target_user = top5_users[0]  # 目标用户
    compare_rows = []  # 比较结果列表

    for other_user in top5_users[1:]:
        vec_target = sample_matrix.loc[target_user].values  # 目标用户特征向量
        vec_other = sample_matrix.loc[other_user].values  # 其他用户特征向量

        # 计算各种相似度指标
        euc_dist = euclidean(vec_target, vec_other)  # 欧氏距离
        cos_sim = cosine_similarity([vec_target], [vec_other])[0][0]  # 余弦相似度
        
        # 皮尔逊相关系数
        if np.std(vec_target) == 0 or np.std(vec_other) == 0:
            pearson_corr = np.nan
        else:
            pearson_corr, _ = pearsonr(vec_target, vec_other)

        # 记录比较结果
        compare_rows.append({
            'TargetUser': target_user,  # 目标用户
            'OtherUser': other_user,  # 其他用户
            'EuclideanDistance': euc_dist,  # 欧氏距离
            'CosineSimilarity': cos_sim,  # 余弦相似度
            'PearsonCorrelation': pearson_corr  # 皮尔逊相关系数
        })

    # 创建比较结果DataFrame
    compare_df = pd.DataFrame(compare_rows)
    # 按余弦相似度降序排列，取前3个最相似用户
    top3_cosine = compare_df.sort_values(by='CosineSimilarity', ascending=False).head(3)

    # 打印结果
    print(f'\n前5名活跃用户: {top5_users}')
    print('\n用户两两欧氏距离示例:')
    print(pd.DataFrame(euclidean_results, columns=['User1', 'User2', 'EuclideanDistance']))
    print('\n用户两两余弦相似度示例:')
    print(pd.DataFrame(cosine_results, columns=['User1', 'User2', 'CosineSimilarity']))
    print('\n用户两两皮尔逊相关系数示例:')
    print(pd.DataFrame(pearson_results, columns=['User1', 'User2', 'PearsonCorrelation']))

    print(f'\n以用户 {target_user} 为目标用户，最相似Top3用户（按余弦相似度）:')
    print(top3_cosine)
    print('\n结论：在高维稀疏购买矩阵中，余弦相似度更关注购买结构比例，弱化购买量级差异，通常比欧氏距离更稳定。')

    return top5_users, compare_df


def feature_scaling(df):
    """
    特征缩放：对数值特征进行归一化和标准化
    
    参数:
    df: 输入的DataFrame
    
    返回:
    df: 添加缩放特征后的DataFrame
    """
    # 选择需要缩放的特征列
    feature_cols = ['Quantity', 'UnitPrice', 'Revenue']

    # 初始化缩放器
    minmax_scaler = MinMaxScaler()  # 最小-最大归一化
    zscore_scaler = StandardScaler()  # Z-Score标准化

    # 执行特征缩放
    minmax_values = minmax_scaler.fit_transform(df[feature_cols])  # 最小-最大缩放
    zscore_values = zscore_scaler.fit_transform(df[feature_cols])  # Z-Score标准化

    # 将缩放后的特征添加到DataFrame
    for idx, col in enumerate(feature_cols):
        df[f'{col}_MinMax'] = minmax_values[:, idx]  # 添加最小-最大缩放列
        df[f'{col}_ZScore'] = zscore_values[:, idx]  # 添加Z-Score标准化列

    plot_df = df
    if len(plot_df) > 200000:
        plot_df = plot_df.sample(n=200000, random_state=42)

    for col in feature_cols:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

        low, high = plot_df[col].quantile([0.01, 0.99])
        original_series = plot_df[col].clip(lower=low, upper=high)
        sns.histplot(original_series, kde=True, bins=50, stat='density', ax=axes[0])
        axes[0].set_title(f'{col} 原始分布')
        if col in {'Quantity', 'Revenue'}:
            axes[0].set_xscale('log')

        sns.histplot(plot_df[f'{col}_MinMax'], kde=True, bins=50, stat='density', ax=axes[1])
        axes[1].set_title(f'{col} Min-Max')

        z_low, z_high = plot_df[f'{col}_ZScore'].quantile([0.01, 0.99])
        z_series = plot_df[f'{col}_ZScore'].clip(lower=z_low, upper=z_high)
        sns.histplot(z_series, kde=True, bins=50, stat='density', ax=axes[2])
        axes[2].set_title(f'{col} Z-Score')

        plt.tight_layout()
        plt.show()

    return df


def encode_categorical_features(df):
    """
    对分类特征进行编码：标签编码和独热编码
    
    参数:
    df: 输入的DataFrame
    
    返回:
    df: 添加编码特征后的DataFrame
    """
    # 标签编码：将国家名称转换为数值标签
    label_encoder = LabelEncoder()  # 初始化标签编码器
    df['Country_Label'] = label_encoder.fit_transform(df['Country'])  # 对Country列进行标签编码

    # 独热编码：创建国家虚拟变量
    country_dummies = pd.get_dummies(df['Country'], prefix='Country')  # 创建独热编码
    # 打印维度变化信息
    print(f'\nOne-Hot前维度: {df.shape}, One-Hot后维度: {(pd.concat([df, country_dummies], axis=1)).shape}')

    # 将独热编码结果合并到原DataFrame
    df = pd.concat([df, country_dummies], axis=1)
    return df


def export_data(df, output_path):
    """
    导出清洗后的数据到CSV文件
    
    参数:
    df: 要导出的DataFrame
    output_path: 输出文件路径
    
    返回:
    None
    """
    # 导出数据到CSV文件
    df.to_csv(output_path, index=False, encoding='utf-8-sig')  # 不保存索引，使用UTF-8-BOM编码支持Excel中文
    print(f'\n已导出清洗后数据到: {output_path}')
    return None


def main(data_path, output_path, sample_size):
    """
    主函数，整合数据预处理流程
    
    参数:
    data_path: 原始数据文件路径
    output_path: 清洗后数据输出路径
    sample_size: 采样大小
    
    返回:
    None
    """
    # 1. 数据加载与基础概览
    df = load_data(data_path, sample_size)  # 加载数据
    data_overview(df)  # 数据概览

    # 2. 缺失值处理与异常值处理
    df = handle_missing_values(df)  # 处理缺失值
    df = detect_and_handle_outliers(df)  # 处理异常值

    # 3. 类型修正与文本清洗
    df = fix_types_and_clean_text(df)  # 修正数据类型和清洗文本

    # 4. 单变量分析与多变量分析
    univariate_analysis(df)  # 单变量分析
    multivariate_analysis(df)  # 多变量分析

    # 5. 构建用户物品矩阵并计算相似性
    user_item_matrix = build_user_item_matrix(df)  # 构建用户-物品矩阵
    calculate_similarity_metrics(user_item_matrix)  # 计算相似度指标

    # 6. 特征缩放与分类编码
    df = feature_scaling(df)  # 特征缩放
    df = encode_categorical_features(df)  # 分类特征编码

    # 7. 导出数据
    export_data(df, output_path)  # 导出清洗后数据
    print(f'\n当前仅抽样前 {sample_size} 条数据用于验证，可在 __main__ 中修改 SAMPLE_SIZE 调整数据量。')
    return None


if __name__ == '__main__':
    """
    程序入口点
    设置文件路径和采样大小，执行主函数
    """
    # 文件路径配置
    DATA_PATH = r'data/online_retail_II.csv'  # 原始数据路径
    OUTPUT_PATH = r'outputs/cleaned_online_retail.csv'  # 输出数据路径
    SAMPLE_SIZE = 1067371  # 采样大小（仅用于快速验证，实际使用可设为None或较大值）
    
    # 执行主函数
    main(DATA_PATH, OUTPUT_PATH, SAMPLE_SIZE)
