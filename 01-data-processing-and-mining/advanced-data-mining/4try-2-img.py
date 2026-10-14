# ============================================================
# 实验4：居民家庭用电时序分析与BP神经网络预测
# 本代码用于Jupyter Notebook中逐任务运行，每个任务函数定义后立即调用
# 所有图表通过 plt.show() 在Jupyter中直接显示，不保存到文件
# ============================================================

# ---------- 全局导入与设置 ----------
import io  # io.StringIO用于捕获DataFrame.info()的文本输出
import os  # os.path用于路径操作
import warnings  # warnings用于过滤不必要的警告信息

import matplotlib.pyplot as plt  # matplotlib.pyplot用于绘制所有图表
import numpy as np  # numpy用于数值计算和数组操作
import pandas as pd  # pandas用于数据读取、处理和分析

# scikit-learn相关导入
from sklearn.ensemble import IsolationForest  # 孤立森林，用于基于模型的离群点检测
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # 回归评估指标
from sklearn.neural_network import MLPRegressor  # 多层感知机回归器（BP神经网络）
from sklearn.preprocessing import StandardScaler  # Z-Score标准化器

# statsmodels相关导入
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # 绘制ACF和PACF相关图
from statsmodels.tsa.seasonal import seasonal_decompose  # 时序分解（趋势+季节+残差）
from statsmodels.tsa.stattools import adfuller  # ADF平稳性检验

# 全局设置：过滤警告，避免大量非关键警告干扰Jupyter输出
warnings.filterwarnings("ignore")
# 全局设置：matplotlib中文字体，使图表标题和标签正常显示中文
plt.rcParams["font.sans-serif"] = ["SimHei"]
# 全局设置：解决matplotlib中负号显示为方块的问题
plt.rcParams["axes.unicode_minus"] = False

# 全局变量：数据集文件路径（分号分隔的文本文件，非标准CSV）
DATASET_PATH = r"data/household_power_consumption.txt"
# 全局变量：滞后小时数，用于构造监督学习特征（用过去6小时预测当前时刻）
LAG_HOURS = 6
# 全局变量：目标列名称（全局有功功率，单位为千瓦）
TARGET_COLUMN = "Global_active_power"

# ===== 快速测试配置 =====
# 设为 True 使用部分数据集快速测试，设为 False 使用完整数据集
USE_SUBSET = True
# 子数据集使用的行数（从原始数据中取前 N 行），可根据需要调整
# 完整数据集约 200 万行，50000 行约 35 天数据，运行只需 1-2 分钟
SUBSET_ROWS = 50000


# ==================== 任务1：数据加载与概览 ====================

def task1_load_and_overview():
    """
    任务1：数据加载与概览
    1. 读取CSV文件，解析日期时间列，设为索引
    2. 使用info()、describe()查看数据基本情况
    3. 检查缺失值（原始数据中表示的缺失值，转换为NaN）
    4. 对缺失值进行插值或删除
    """
    # 使用pandas读取数据文件：
    # sep=";" - 以分号作为分隔符
    # na_values=["?"] - 将"?"字符识别为缺失值NaN
    # low_memory=False - 禁用低内存模式，确保数据类型推断准确
    raw_data = pd.read_csv(
        DATASET_PATH,
        sep=";",
        na_values=["?"],
        low_memory=False,
    )
    # 将Date和Time两列合并为Datetime，并转换为datetime类型
    # dayfirst=True - 以日/月/年格式解析日期（符合欧洲日期格式）
    # errors="coerce" - 无法解析的日期设为NaT（Not a Time）
    raw_data["Datetime"] = pd.to_datetime(
        raw_data["Date"] + " " + raw_data["Time"],
        dayfirst=True,
        errors="coerce",
    )
    # 删除原始的Date和Time列，保留合并后的Datetime列
    raw_data = raw_data.drop(columns=["Date", "Time"])
    # 将Datetime列设为DataFrame的行索引，便于时序操作
    raw_data = raw_data.set_index("Datetime")
    # 按时间索引升序排列，确保时序顺序正确
    raw_data = raw_data.sort_index()
    # 将所有列强制转换为数值类型：errors="coerce"将无法转换的值设为NaN
    for column_name in raw_data.columns:
        raw_data[column_name] = pd.to_numeric(raw_data[column_name], errors="coerce")

    # ===== 如需快速测试，截取子数据集 =====
    if USE_SUBSET:
        raw_data = raw_data.head(SUBSET_ROWS)

    # 打印任务标题，便于在Jupyter中定位
    print("=" * 60)
    print("任务1：数据加载与概览")
    if USE_SUBSET:
        print(f"（快速测试模式：仅使用前 {SUBSET_ROWS} 行数据）")
    print("=" * 60)

    # 输出原始数据形状：行数（时间点数）× 列数（电表读数变量数）
    print(f"\n原始数据形状：{raw_data.shape}（时间点数 × 变量数）")

    # 使用io.StringIO捕获info()的文本输出，info()本身直接打印到控制台不返回字符串
    info_buffer = io.StringIO()
    raw_data.info(buf=info_buffer)  # 将info输出重定向到buffer
    info_text = info_buffer.getvalue()  # 获取buffer中的文本内容
    print("\n>>> df.info() 输出：")
    print(info_text)  # 打印info()输出：列名、非空数量、数据类型、内存占用

    # describe()输出各数值列的统计摘要：均值、标准差、最小值、四分位数、最大值
    describe_text = raw_data.describe().to_string()  # to_string()格式化DataFrame为可打印字符串
    print("\n>>> df.describe() 输出：")
    print(describe_text)

    # 统计每列的缺失值数量：isna()生成布尔矩阵，sum()按列求和
    raw_missing = raw_data.isna().sum()
    print("\n>>> 原始数据缺失值统计（每列）：")
    print(raw_missing)  # 打印各列缺失值数量
    print(f"总缺失值数量: {raw_missing.sum()}")  # 打印总计缺失值数量

    # ===== 缺失值处理 =====
    # 使用时间插值法填充缺失值：interpolate(method="time")根据时间索引线性插值
    # 插值后仍可能有边界NaN，再用ffill（前向填充）和bfill（后向填充）兜底
    cleaned_data = raw_data.interpolate(method="time")  # 按时间序列线性插值
    cleaned_data = cleaned_data.ffill().bfill()  # 前向和后向填充处理边界缺失
    # 确保没有残留的缺失值（转换过程中可能产生）
    cleaned_data = cleaned_data.dropna()

    # 输出清洗后的数据形状和缺失值统计
    cleaned_missing = cleaned_data.isna().sum()
    print(f"\n清洗后数据形状：{cleaned_data.shape}")
    print(f"清洗后缺失值总数: {cleaned_missing.sum()}（应为0）")

    # ===== 构造小时级数据 =====
    # 将原始数据（分钟级采样）重采样为小时均值：resample("h").mean()
    # numeric_only=True - 仅对数值列计算均值（pandas 2.x要求）
    hourly_data = cleaned_data.resample("h").mean(numeric_only=True)
    # 重采样后可能仍有空洞，再次插值填充
    hourly_data = hourly_data.interpolate(method="time").ffill().bfill()
    print(f"\n小时级数据形状：{hourly_data.shape}")

    # 提取目标列（全局有功功率）作为一维Series，.copy()创建独立副本
    hourly_series = hourly_data[TARGET_COLUMN].copy()
    print(f"小时级目标序列长度: {len(hourly_series)}")
    # 显示前5条数据预览
    print(f"\n目标序列前5条预览：\n{hourly_series.head()}")

    # 返回所有结果供后续任务使用
    return raw_data, info_text, describe_text, raw_missing, cleaned_data, hourly_data, hourly_series


# 任务1函数定义完成后立即调用，并将结果赋值给全局变量
(raw_data, info_text, describe_text, raw_missing,
 cleaned_data, hourly_data, hourly_series) = task1_load_and_overview()


# ==================== 任务2：时序基础分析 ====================

def task2_basic_time_series_analysis(hourly_series):
    """
    任务2：时序基础分析
    1. 绘制全局有功功率随时间变化的折线图
    2. 计算日平均功率，绘制箱线图观察每日用电模式
    3. 计算ACF和PACF，ADF检验判断序列平稳性
    """
    print("=" * 60)
    print("任务2：时序基础分析")
    print("=" * 60)

    # ===== 2.1 全局有功功率时序折线图 =====
    fig, ax = plt.subplots(figsize=(15, 5))  # 创建画布，宽15英寸高5英寸
    # 绘制折线图：color="tab:blue"使用matplotlib标准蓝色，linewidth=0.8控制线条细度
    ax.plot(hourly_series.index, hourly_series.values, color="tab:blue", linewidth=0.8)
    ax.set_title("全局有功功率小时序列")  # 设置图表标题
    ax.set_xlabel("时间")  # 设置x轴标签
    ax.set_ylabel("功率（kW）")  # 设置y轴标签
    ax.grid(alpha=0.3)  # 添加网格线，alpha=0.3使网格半透明
    fig.tight_layout()  # 自动调整子图间距，防止标签被裁剪
    plt.show()  # 在Jupyter中直接显示图表

    # ===== 2.2 日平均功率箱线图 =====
    # 将小时级数据按天(D)重采样求均值
    daily_average = hourly_series.resample("D").mean().to_frame(name="日平均功率")
    # 映射星期几（0=周一, 6=周日）为中文标签
    weekday_map = {
        0: "周一", 1: "周二", 2: "周三", 3: "周四",
        4: "周五", 5: "周六", 6: "周日",
    }
    # dayofweek返回0-6，用map映射为中文，添加"星期几"列
    daily_average["星期几"] = daily_average.index.dayofweek.map(weekday_map)

    fig, ax = plt.subplots(figsize=(10, 6))  # 创建画布
    # boxplot：按"星期几"分组绘制"日平均功率"的箱线图
    # by="星期几" - 分组依据列，column="日平均功率" - 绘图列
    daily_average.boxplot(column="日平均功率", by="星期几", ax=ax)
    ax.set_title("不同星期的日平均功率箱线图")  # 设置标题
    ax.set_xlabel("星期")  # x轴标签
    ax.set_ylabel("日平均功率（kW）")  # y轴标签
    fig.suptitle("")  # 清除pandas自动添加的suptitle，避免标题重叠
    fig.tight_layout()
    plt.show()

    # ===== 2.3 自相关（ACF）和偏自相关（PACF）分析 =====
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))  # 2行1列的子图布局
    # 绘制ACF图：lags=72显示72阶延迟（3天，每小时1阶）
    # ACF衡量序列与自身滞后值之间的线性相关性
    plot_acf(hourly_series, lags=72, ax=axes[0])
    axes[0].set_title("全局有功功率 ACF（自相关函数）")
    # 绘制PACF图：method="ywm"使用Yule-Walker方法估计
    # PACF衡量去除中间滞后影响后的直接相关性
    plot_pacf(hourly_series, lags=72, ax=axes[1], method="ywm")
    axes[1].set_title("全局有功功率 PACF（偏自相关函数）")
    fig.tight_layout()
    plt.show()

    # ===== 2.4 ADF平稳性检验 =====
    # ADF（Augmented Dickey-Fuller）检验零假设：序列存在单位根（非平稳）
    # p值 < 0.05 则拒绝零假设，认为序列平稳
    adf_result = adfuller(hourly_series)  # 返回元组：(统计量, p值, 滞后阶数, 样本量, ...)
    adf_summary = {
        "ADF统计量": adf_result[0],  # 检验统计量（越负越拒绝零假设）
        "p值": adf_result[1],  # p值：小于0.05表示序列平稳
        "滞后阶数": adf_result[2],  # 检验使用的滞后阶数
        "样本量": adf_result[3],  # 实际使用的样本量
    }
    print("\n>>> ADF平稳性检验结果：")
    # 逐项打印ADF检验结果
    for key, value in adf_summary.items():
        print(f"  {key}: {value}")
    # 根据p值判断平稳性并输出结论
    if adf_result[1] < 0.05:
        print("  结论：p值 < 0.05，序列是平稳的（拒绝存在单位根的零假设）")
    else:
        print("  结论：p值 >= 0.05，序列是非平稳的（不能拒绝存在单位根的零假设）")

    return daily_average, adf_summary


# 任务2调用，传入小时级序列
daily_average, adf_summary = task2_basic_time_series_analysis(hourly_series)


# ==================== 任务3：时序分解 ====================

def task3_time_series_decomposition(hourly_series):
    """
    任务3：时序分解
    1. 使用seasonal_decompose将时间序列分解为趋势、季节性和残差
    2. 绘制分解图，分析用电量的日/周周期性
    """
    print("=" * 60)
    print("任务3：时序分解")
    print("=" * 60)

    # ===== 3.1 时序分解：趋势 + 季节 + 残差 =====
    # model="additive"使用加法模型：Y = 趋势 + 季节 + 残差
    # period=24表示周期为24小时（即日周期），数据为小时级采样
    decomposition = seasonal_decompose(hourly_series, model="additive", period=24)
    # decomposition.plot()返回Figure对象，可直接设置大小
    fig = decomposition.plot()
    fig.set_size_inches(14, 10)  # 设置画布大小
    fig.tight_layout()
    plt.show()  # 显示分解图（原始序列、趋势、季节、残差共4个子图）

    # ===== 3.2 工作日 vs 周末小时用电模式对比 =====
    # 构建分析用DataFrame：保留功率值，添加"小时"和"是否周末"列
    profile_data = hourly_series.to_frame(name="功率")
    # index.hour提取每个时间戳的小时部分（0-23）
    profile_data["小时"] = profile_data.index.hour
    # index.dayofweek >= 5即周六(5)或周日(6)为周末，其余为工作日
    profile_data["是否周末"] = np.where(profile_data.index.dayofweek >= 5, "周末", "工作日")
    # 按"是否周末"和"小时"分组求平均功率，unstack(0)将"是否周末"展开为列
    profile_table = profile_data.groupby(["是否周末", "小时"])["功率"].mean().unstack(0)

    fig, ax = plt.subplots(figsize=(12, 5))  # 创建画布
    # 遍历工作日和周末两列（或只遍历存在的列），绘制小时平均功率曲线
    for column_name in profile_table.columns:
        # marker="o"在每个数据点添加圆点标记，label用于图例
        ax.plot(profile_table.index, profile_table[column_name],
                marker="o", label=column_name)
    ax.set_title("工作日与周末的小时平均功率对比")
    ax.set_xlabel("小时")
    ax.set_ylabel("平均功率（kW）")
    ax.grid(alpha=0.3)  # 添加半透明网格
    ax.legend()  # 显示图例（工作日/周末）
    fig.tight_layout()
    plt.show()

    # 打印工作日和周末功率对比摘要
    workday_avg = float(profile_data[profile_data["是否周末"] == "工作日"]["功率"].mean())
    weekend_avg = float(profile_data[profile_data["是否周末"] == "周末"]["功率"].mean())
    print(f"\n用电量日/周周期分析：")
    print(f"  工作日平均功率: {workday_avg:.4f} kW")
    print(f"  周末平均功率: {weekend_avg:.4f} kW")

    return decomposition, profile_table


# 任务3调用
decomposition, profile_table = task3_time_series_decomposition(hourly_series)


# ==================== 任务4：离群点检测——基于统计方法 ====================

def task4_outlier_zscore(hourly_series, threshold=3.0):
    """
    任务4：离群点检测——基于统计方法
    1. 使用Z-score方法检测全局有功功率的离群点
    2. 阈值设为3倍标准差
    3. 标记离群点并绘制在原始序列上
    """
    print("=" * 60)
    print("任务4：离群点检测——基于统计方法（Z-score）")
    print("=" * 60)

    # ===== 4.1 计算Z-score =====
    # Z-score = (值 - 均值) / 标准差
    # 当|Z-score| > threshold时判定为离群点
    # ddof=0表示总体标准差（除以n），ddof=1为样本标准差（除以n-1）
    z_score = (hourly_series - hourly_series.mean()) / hourly_series.std(ddof=0)
    # np.abs(z_score) > threshold：布尔数组，标记Z-score超出阈值的点
    # .astype(int)转换为0（正常）/1（离群）标签
    outlier_flag = (np.abs(z_score) > threshold).astype(int)
    # 提取被标记为离群的功率值
    outlier_points = hourly_series[outlier_flag == 1]

    # ===== 4.2 绘制离群点检测图 =====
    fig, ax = plt.subplots(figsize=(15, 5))  # 创建画布
    # 绘制原始功率序列作为背景（蓝色细线）
    ax.plot(hourly_series.index, hourly_series.values,
            color="tab:blue", linewidth=0.8, label="原始序列")
    # 在原始序列上叠加绘制离群点（红色散点，s=20控制点的大小）
    ax.scatter(
        outlier_points.index,
        outlier_points.values,
        color="red",
        s=20,
        label=f"Z-score离群点 (|Z|>{threshold})",
    )
    ax.set_title(f"基于 Z-score 的离群点检测（阈值={threshold}）")
    ax.set_xlabel("时间")
    ax.set_ylabel("功率（kW）")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # 输出离群点统计信息
    outlier_count = outlier_flag.sum()
    total_count = len(hourly_series)
    print(f"\nZ-score离群点检测结果（阈值={threshold}）：")
    print(f"  离群点数量: {outlier_count}")
    print(f"  总样本数: {total_count}")
    print(f"  离群点占比: {outlier_count / total_count * 100:.2f}%")
    # 列出离群值最大的几个时间点
    print(f"\n  离群值最大的5个时间点：")
    top_outliers = np.abs(z_score).sort_values(ascending=False).head(5)
    for time_point in top_outliers.index:
        print(f"    {time_point}: 功率={hourly_series[time_point]:.4f}, Z-score={z_score[time_point]:.4f}")

    return z_score, outlier_flag


# 任务4调用
z_score, zscore_flag = task4_outlier_zscore(hourly_series, threshold=3.0)


# ==================== 任务5：离群点检测——基于模型方法 ====================

def build_iforest_features(hourly_series, window_size):
    """
    辅助函数：构建孤立森林所需的时序特征
    输入：小时级功率序列、滑动窗口大小
    输出：包含多种统计特征的DataFrame
    """
    # 创建一个以原序列索引为行索引的空DataFrame
    feature_data = pd.DataFrame(index=hourly_series.index)
    # 特征1：当前时刻的功率值
    feature_data["current_power"] = hourly_series
    # 特征2：过去window_size小时的均值（反映短期趋势）
    feature_data["mean_6h"] = hourly_series.rolling(window=window_size).mean()
    # 特征3：过去window_size小时的标准差（反映短期波动程度）
    feature_data["std_6h"] = hourly_series.rolling(window=window_size).std()
    # 特征4：过去window_size小时的最小值
    feature_data["min_6h"] = hourly_series.rolling(window=window_size).min()
    # 特征5：过去window_size小时的最大值
    feature_data["max_6h"] = hourly_series.rolling(window=window_size).max()
    # 特征6：1小时差分（当前时刻与上一时刻的差值，反映瞬时变化率）
    feature_data["diff_1h"] = hourly_series.diff(1)
    # 删除因滚动窗口和差分产生的NaN行
    feature_data = feature_data.dropna()
    return feature_data


def task5_outlier_iforest(hourly_series, zscore_flag, window_size=6):
    """
    任务5：离群点检测——基于模型方法
    1. 使用孤立森林（IsolationForest）对时序特征进行离群点检测
    2. 对比Z-score与孤立森林的检测结果
    """
    print("=" * 60)
    print("任务5：离群点检测——基于模型方法（孤立森林）")
    print("=" * 60)

    # ===== 5.1 构建特征并训练孤立森林 =====
    # 调用辅助函数构建滚动窗口时序特征
    feature_data = build_iforest_features(hourly_series, window_size)
    # 创建孤立森林模型：
    # n_estimators=200 - 使用200棵决策树构建森林
    # contamination=0.01 - 预期数据中约1%为异常点（控制异常比例）
    # random_state=42 - 固定随机种子，保证结果可重复
    model = IsolationForest(
        n_estimators=200,
        contamination=0.01,
        random_state=42,
    )
    # fit_predict返回预测标签：1为正常点，-1为异常点
    prediction = model.fit_predict(feature_data)
    # 将-1（异常）转换为1，将1（正常）转换为0，与Z-score的标记方式一致
    iforest_flag_compact = pd.Series(
        np.where(prediction == -1, 1, 0),  # -1→1（离群），1→0（正常）
        index=feature_data.index,  # 使用特征数据的索引（因rolling已去除部分行）
        name="iforest_outlier",
    )
    # 将紧凑版标志对齐到完整的原序列索引上（缺失部分填0，即默认正常）
    aligned_iforest_flag = pd.Series(0, index=hourly_series.index, name="iforest_outlier")
    aligned_iforest_flag.loc[iforest_flag_compact.index] = iforest_flag_compact

    # ===== 5.2 绘制孤立森林离群点图 =====
    outlier_points = hourly_series[aligned_iforest_flag == 1]  # 提取离群点
    fig, ax = plt.subplots(figsize=(15, 5))
    # 绘制原始序列（蓝色细线背景）
    ax.plot(hourly_series.index, hourly_series.values,
            color="tab:blue", linewidth=0.8, label="原始序列")
    # 绘制孤立森林检测的离群点（橙色散点）
    ax.scatter(
        outlier_points.index,
        outlier_points.values,
        color="darkorange",
        s=20,
        label="孤立森林离群点",
    )
    ax.set_title("基于孤立森林的离群点检测")
    ax.set_xlabel("时间")
    ax.set_ylabel("功率（kW）")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # ===== 5.3 对比Z-score与孤立森林的检测结果 =====
    # 构建两种方法检测结果的比较DataFrame
    comparison_data = pd.DataFrame(index=zscore_flag.index)
    comparison_data["zscore"] = zscore_flag.astype(int)
    comparison_data["iforest"] = aligned_iforest_flag.astype(int)
    # 统计两种方法共同识别的离群点数量（两个标记都为1的行）
    overlap_count = int(((comparison_data["zscore"] == 1) & (comparison_data["iforest"] == 1)).sum())
    comparison_summary = {
        "Z-score离群点数量": int(comparison_data["zscore"].sum()),
        "孤立森林离群点数量": int(comparison_data["iforest"].sum()),
        "共同识别数量": overlap_count,
        "仅Z-score识别": int(comparison_data["zscore"].sum()) - overlap_count,
        "仅孤立森林识别": int(comparison_data["iforest"].sum()) - overlap_count,
    }

    print("\n>>> 离群点检测结果对比：")
    for key, value in comparison_summary.items():
        print(f"  {key}: {value}")
    print("\n方法对比分析：")
    print("  Z-score：基于正态分布假设，适合数据近似正态分布的场景，")
    print("           计算简单、可解释性强，但对非对称分布或长尾数据效果差。")
    print("  孤立森林：不依赖分布假设，能捕获多维特征中的复杂异常模式，")
    print("           适合非线性和多特征场景，但需要合理的参数调优。")

    return aligned_iforest_flag, feature_data, comparison_summary


# 任务5调用
iforest_flag, iforest_feature_data, comparison_summary = task5_outlier_iforest(
    hourly_series, zscore_flag, window_size=LAG_HOURS
)

# 将两种离群点检测结果合并：任一方法标记为离群即视为离群
combined_anomaly_flag = ((zscore_flag == 1) | (iforest_flag == 1)).astype(int)


# ==================== 任务6：构造监督学习数据集 ====================

def task6_build_supervised_dataset(hourly_series, lag_hours, anomaly_flag):
    """
    任务6：构造监督学习数据集
    1. 将时序预测转化为监督学习：用过去n个时间步预测下一时刻
    2. 添加时间特征：小时、星期几、是否为周末
    3. 划分训练集（前80%）和测试集（后20%），保持时序顺序
    """
    print("=" * 60)
    print("任务6：构造监督学习数据集")
    print("=" * 60)

    # ===== 6.1 构建监督学习数据集 =====
    # 创建DataFrame，索引与原序列相同，首列为预测目标（当前功率）
    supervised_data = pd.DataFrame(index=hourly_series.index)
    supervised_data["target"] = hourly_series  # 目标列：当前时刻的功率
    # 添加滞后特征：对每个lag_step（1到lag_hours），创建shift(lag_step)列
    # 例如lag_1为上一小时的功率，lag_6为6小时前的功率
    for lag_step in range(1, lag_hours + 1):
        supervised_data[f"lag_{lag_step}"] = hourly_series.shift(lag_step)
    # 添加时间特征：提取小时（0-23），反映日周期模式
    supervised_data["hour"] = supervised_data.index.hour
    # 添加时间特征：星期几（0=周一，6=周日），反映周周期模式
    supervised_data["day_of_week"] = supervised_data.index.dayofweek
    # 添加时间特征：是否周末（周六/周日=1，其他=0）
    supervised_data["is_weekend"] = (supervised_data.index.dayofweek >= 5).astype(int)
    # 添加时间特征：月份（1-12），反映季节性变化
    supervised_data["month"] = supervised_data.index.month
    # 添加离群点标签列（来自任务4/5的检测结果），用于后续清洗
    supervised_data["anomaly_flag"] = anomaly_flag.reindex(supervised_data.index).fillna(0).astype(int)
    # 删除因shift产生的NaN行（前lag_hours行没有完整的历史数据）
    supervised_data = supervised_data.dropna()
    print(f"监督学习数据集形状: {supervised_data.shape}（样本数 × 特征数+目标）")

    # 分离特征矩阵和目标向量
    # 特征列：除"target"外的所有列（滞后特征+时间特征+异常标签）
    feature_columns = [col for col in supervised_data.columns if col != "target"]
    features = supervised_data[feature_columns].copy()  # 特征矩阵X
    target = supervised_data["target"].copy()  # 目标向量y
    print(f"特征矩阵形状: {features.shape}，包含 {len(feature_columns)} 个特征")
    print(f"目标向量形状: {target.shape}")

    # ===== 6.2 按时序划分训练集和测试集 =====
    # 时间序列预测必须保持时序顺序，不能随机打乱
    # 前80%作为训练集，后20%作为测试集
    train_ratio = 0.8  # 训练集比例
    split_index = int(len(features) * train_ratio)  # 计算划分位置
    # 按时间索引切片：前split_index行为训练集
    x_train = features.iloc[:split_index].copy()
    y_train = target.iloc[:split_index].copy()
    # 后部分为测试集
    x_test = features.iloc[split_index:].copy()
    y_test = target.iloc[split_index:].copy()

    print(f"\n训练集样本数: {len(x_train)}（{len(x_train)/len(features)*100:.1f}%）")
    print(f"测试集样本数: {len(x_test)}（{len(x_test)/len(features)*100:.1f}%）")
    print(f"训练集时间范围: {x_train.index[0]} ~ {x_train.index[-1]}")
    print(f"测试集时间范围: {x_test.index[0]} ~ {x_test.index[-1]}")

    return supervised_data, features, target, x_train, x_test, y_train, y_test


# 任务6调用
(supervised_data, features, target,
 x_train, x_test, y_train, y_test) = task6_build_supervised_dataset(
    hourly_series, LAG_HOURS, combined_anomaly_flag
)


# ==================== 任务7：数据标准化与离群点处理 ====================

def task7_standardize_and_clean(x_train, x_test, y_train):
    """
    任务7：数据标准化与离群点处理
    1. 对特征进行Z-Score标准化（StandardScaler）
    2. 剔除训练集中的离群点，观察对模型的影响
    """
    print("=" * 60)
    print("任务7：数据标准化与离群点处理")
    print("=" * 60)

    # ===== 7.1 特征标准化（全数据集） =====
    # 创建StandardScaler实例：Z-Score标准化，使每个特征均值为0、标准差为1
    scaler_all = StandardScaler()
    # 在训练集上拟合scaler（计算均值和标准差），然后转换训练集
    x_train_scaled_all = scaler_all.fit_transform(x_train)
    # 使用训练集的均值和标准差转换测试集（避免数据泄露）
    x_test_scaled_all = scaler_all.transform(x_test)

    # 输出标准化后的统计信息，验证标准化效果
    print("\n>>> 全量数据标准化结果：")
    # 每列均值应接近0（浮点数误差可忽略）
    print(f"  训练集各特征均值范围: [{x_train_scaled_all.mean(axis=0).min():.6f}, {x_train_scaled_all.mean(axis=0).max():.6f}]")
    # 每列标准差应接近1
    print(f"  训练集各特征标准差范围: [{x_train_scaled_all.std(axis=0).min():.6f}, {x_train_scaled_all.std(axis=0).max():.6f}]")

    # ===== 7.2 剔除训练集中被标记为离群点的样本 =====
    # 创建过滤器：仅保留anomaly_flag=0的正常样本
    # anomaly_flag列位于x_train的最后一列（任务6中添加）
    filtered_mask = x_train["anomaly_flag"] == 0
    # 应用过滤器到训练集特征和目标
    filtered_x_train = x_train.loc[filtered_mask].copy()
    filtered_y_train = y_train.loc[filtered_mask].copy()
    print(f"\n>>> 剔除离群点：训练集从 {len(x_train)} 缩减至 {len(filtered_x_train)} 样本")
    print(f"  移除样本数: {len(x_train) - len(filtered_x_train)}")

    # 对过滤后的数据重新标准化（新的训练集有不同的均值和标准差）
    scaler_filtered = StandardScaler()
    filtered_x_train_scaled = scaler_filtered.fit_transform(filtered_x_train)
    # 测试集仍用过滤后的scaler转换（保持一致性）
    x_test_scaled_filtered = scaler_filtered.transform(x_test)

    print(f"\n>>> 过滤后数据标准化结果：")
    print(f"  过滤后训练集各特征均值范围: [{filtered_x_train_scaled.mean(axis=0).min():.6f}, {filtered_x_train_scaled.mean(axis=0).max():.6f}]")
    print(f"  过滤后训练集各特征标准差范围: [{filtered_x_train_scaled.std(axis=0).min():.6f}, {filtered_x_train_scaled.std(axis=0).max():.6f}]")

    return (scaler_all, x_train_scaled_all, x_test_scaled_all,
            scaler_filtered, filtered_x_train_scaled, x_test_scaled_filtered,
            filtered_y_train)


# 任务7调用
(scaler_all, x_train_scaled_all, x_test_scaled_all,
 scaler_filtered, filtered_x_train_scaled, x_test_scaled_filtered,
 filtered_y_train) = task7_standardize_and_clean(x_train, x_test, y_train)


# ==================== 任务8：构建BP神经网络 ====================

def task8_build_mlp(x_train_scaled_all, y_train, filtered_x_train_scaled, filtered_y_train):
    """
    任务8：构建BP神经网络
    1. 使用MLPRegressor构建多层感知机
    2. 设置合适的超参数
    3. 分别在全量数据和过滤数据上拟合模型
    4. 记录并绘制损失下降曲线
    """
    print("=" * 60)
    print("任务8：构建BP神经网络（MLP）")
    print("=" * 60)

    # ===== 8.1 定义MLP模型超参数 =====
    # MLPRegressor超参数说明：
    # hidden_layer_sizes=(100, 64, 32) - 3层隐藏层，神经元数分别为100、64、32
    # activation="relu" - ReLU激活函数，计算高效且能缓解梯度消失
    # solver="adam" - Adam优化器，自适应学习率，收敛速度快
    # alpha=0.0005 - L2正则化系数，防止过拟合（越大惩罚越强）
    # learning_rate_init=0.001 - 初始学习率
    # max_iter=300 - 最大迭代次数（epochs）
    # early_stopping=True - 启用早停机制，验证集loss不再下降时提前停止
    # validation_fraction=0.1 - 从训练集中划分10%作为验证集
    # random_state=42 - 固定随机种子

    # ===== 8.2 训练基准模型（使用全量训练数据，含离群点） =====
    print("\n>>> 训练基准模型（含离群点）...")
    baseline_model = MLPRegressor(
        hidden_layer_sizes=(100, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    # 在全量标准化训练集上拟合模型
    baseline_model.fit(x_train_scaled_all, y_train)
    # 输出基准模型的训练信息
    print(f"  基准模型迭代次数: {baseline_model.n_iter_}")
    print(f"  基准模型最终损失: {baseline_model.loss_:.6f}")

    # ===== 8.3 训练过滤模型（剔除离群点后的训练数据） =====
    print("\n>>> 训练过滤模型（剔除离群点）...")
    filtered_model = MLPRegressor(
        hidden_layer_sizes=(100, 64, 32),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )
    # 在过滤后的标准化训练集上拟合模型
    filtered_model.fit(filtered_x_train_scaled, filtered_y_train)
    print(f"  过滤模型迭代次数: {filtered_model.n_iter_}")
    print(f"  过滤模型最终损失: {filtered_model.loss_:.6f}")

    # ===== 8.4 绘制损失下降曲线 =====
    fig, ax = plt.subplots(figsize=(10, 5))  # 创建画布
    # 绘制基准模型每轮迭代的损失值，loss_curve_存储了每次迭代的loss
    ax.plot(baseline_model.loss_curve_, label="基准模型（含离群点）", linewidth=1.5)
    # 绘制过滤后模型的损失曲线
    ax.plot(filtered_model.loss_curve_, label="过滤模型（剔除离群点）", linewidth=1.5)
    ax.set_title("MLP 训练损失曲线")
    ax.set_xlabel("迭代次数")
    ax.set_ylabel("Loss（损失值）")
    ax.grid(alpha=0.3)  # 添加半透明网格
    ax.legend()  # 显示图例
    fig.tight_layout()
    plt.show()

    return baseline_model, filtered_model


# 任务8调用
baseline_model, filtered_model = task8_build_mlp(
    x_train_scaled_all, y_train, filtered_x_train_scaled, filtered_y_train
)


# ==================== 任务9：模型评估与对比 ====================

def task9_evaluate_and_compare(
    baseline_model, x_test_scaled_all, y_test,
    filtered_model, x_test_scaled_filtered
):
    """
    任务9：模型评估与对比
    1. 计算RMSE、MAE、R2
    2. 绘制真实值与预测值的对比折线图
    """
    print("=" * 60)
    print("任务9：模型评估与对比")
    print("=" * 60)

    # ===== 9.1 基准模型预测与评估 =====
    # 基准模型在测试集上进行预测
    baseline_prediction = baseline_model.predict(x_test_scaled_all)
    # 计算基准模型的三项评估指标
    baseline_metrics = {
        # 均方根误差RMSE：预测值与真实值偏差的平方均值再开方，单位与原始值一致
        "RMSE": float(np.sqrt(mean_squared_error(y_test, baseline_prediction))),
        # 平均绝对误差MAE：预测误差绝对值的平均，对异常值不敏感
        "MAE": float(mean_absolute_error(y_test, baseline_prediction)),
        # 决定系数R2：1减去残差平方和与总方差之比（0~1，越大越好）
        "R2": float(r2_score(y_test, baseline_prediction)),
    }

    # ===== 9.2 过滤模型预测与评估 =====
    # 过滤后模型在测试集上进行预测
    filtered_prediction = filtered_model.predict(x_test_scaled_filtered)
    # 计算过滤模型的三项评估指标
    filtered_metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, filtered_prediction))),
        "MAE": float(mean_absolute_error(y_test, filtered_prediction)),
        "R2": float(r2_score(y_test, filtered_prediction)),
    }

    # ===== 9.3 打印评估指标对比 =====
    print("\n>>> 模型评估指标对比：")
    print("--" * 30)
    # 打印基准模型的各项指标
    print("基准模型（含离群点训练）：")
    print(f"  RMSE: {baseline_metrics['RMSE']:.4f} kW")
    print(f"  MAE : {baseline_metrics['MAE']:.4f} kW")
    print(f"  R2  : {baseline_metrics['R2']:.4f}")
    print("--" * 30)
    # 打印过滤模型的各项指标
    print("过滤模型（剔除离群点训练）：")
    print(f"  RMSE: {filtered_metrics['RMSE']:.4f} kW")
    print(f"  MAE : {filtered_metrics['MAE']:.4f} kW")
    print(f"  R2  : {filtered_metrics['R2']:.4f}")
    print("--" * 30)

    # ===== 9.4 绘制真实值与预测值对比折线图 =====
    fig, ax = plt.subplots(figsize=(15, 5))  # 创建宽画布，容纳时序数据
    # 绘制真实值（黑色线，linewidth=1.0较粗以示强调）
    ax.plot(y_test.index, y_test.values, color="black", linewidth=1.0, label="真实值")
    # 绘制基准模型预测值（半透明蓝色线）
    ax.plot(y_test.index, baseline_prediction, linewidth=0.9,
            label=f"基准模型预测 (R2={baseline_metrics['R2']:.4f})")
    # 绘制过滤模型预测值（半透明橙色线）
    ax.plot(y_test.index, filtered_prediction, linewidth=0.9,
            label=f"过滤模型预测 (R2={filtered_metrics['R2']:.4f})")
    ax.set_title("测试集真实值与预测值对比")
    ax.set_xlabel("时间")
    ax.set_ylabel("功率（kW）")
    ax.grid(alpha=0.3)  # 添加半透明网格
    ax.legend()  # 显示图例
    fig.tight_layout()
    plt.show()

    # 汇总模型指标
    metrics_summary = {
        "基准模型": baseline_metrics,
        "剔除离群点模型": filtered_metrics,
    }

    return baseline_prediction, baseline_metrics, filtered_prediction, filtered_metrics, metrics_summary


# 任务9调用
(baseline_prediction, baseline_metrics,
 filtered_prediction, filtered_metrics, metrics_summary) = task9_evaluate_and_compare(
    baseline_model, x_test_scaled_all, y_test,
    filtered_model, x_test_scaled_filtered
)


# ==================== 任务10：扩展任务（LSTM+DBSCAN） ====================

def task10_extensions(hourly_series, feature_data_from_iforest,
                      x_train_scaled_all, x_test_scaled_all, y_train, y_test,
                      x_train, x_test):
    """
    任务10：扩展任务（可选，选做且正确加10分）
    1. 使用Keras构建LSTM进行预测，与MLP对比
    2. 应用DBSCAN进行基于聚类的离群点检测，并比较效果
    """
    print("=" * 60)
    print("任务10：扩展任务（LSTM + DBSCAN）")
    print("=" * 60)

    # ===== 10.1 使用Keras构建LSTM模型 =====
    print("\n>>> 10.1 LSTM模型预测")
    try:
        # 导入Keras相关模块（需要先安装TensorFlow/Keras）
        from keras.models import Sequential  # 序贯模型，逐层堆叠
        from keras.layers import LSTM, Dense, Dropout  # LSTM层、全连接层、Dropout层
        from keras.optimizers import Adam  # Adam优化器
        from keras.callbacks import EarlyStopping  # 早停回调

        # LSTM要求输入形状为 (样本数, 时间步, 特征数)
        # 原始特征包含滞后特征(lag_1~lag_6)、时间特征和异常标签
        # 将6个滞后特征作为时间步重塑：(样本数, 6, 1)，即6个时间步每步1个特征
        # 加上时间特征和异常标签作为额外特征
        n_train = len(x_train_scaled_all)
        n_test = len(x_test_scaled_all)

        # 提取滞后特征列（lag_1 ~ lag_6，前6列）作为LSTM的时间步输入
        lstm_lag_features_train = x_train_scaled_all[:, :LAG_HOURS].reshape(n_train, LAG_HOURS, 1)
        lstm_lag_features_test = x_test_scaled_all[:, :LAG_HOURS].reshape(n_test, LAG_HOURS, 1)

        # 构建LSTM序贯模型
        lstm_model = Sequential([
            # 第1层LSTM：50个单元，return_sequences=True返回每个时间步的输出
            # input_shape=(时间步数, 特征数)=(6, 1)
            LSTM(50, return_sequences=True, input_shape=(LAG_HOURS, 1)),
            Dropout(0.2),  # Dropout层，20%的神经元随机失活，防止过拟合
            # 第2层LSTM：30个单元，return_sequences=False仅返回最后时间步的输出
            LSTM(30, return_sequences=False),
            Dropout(0.2),  # 再次Dropout防止过拟合
            # 全连接输出层：1个神经元，linear激活函数（回归任务）
            Dense(1, activation="linear")
        ])
        # 编译模型：optimizer=Adam(0.001)学习率0.001，loss="mse"均方误差损失
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
        # 早停回调：监控验证集loss，patience=10即10轮无改善则停止
        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0)

        # 输出模型结构摘要
        print("LSTM模型结构：")
        lstm_model.summary()

        # 训练LSTM模型：epochs=50最大训练轮次，batch_size=32每批32样本
        # validation_split=0.1从训练集中分出10%作为验证集，verbose=0静默训练
        lstm_history = lstm_model.fit(
            lstm_lag_features_train, y_train.values,
            epochs=50, batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0
        )

        # LSTM模型在测试集上预测
        lstm_prediction = lstm_model.predict(lstm_lag_features_test, verbose=0).flatten()
        # 计算LSTM模型的三项评估指标
        lstm_rmse = float(np.sqrt(mean_squared_error(y_test, lstm_prediction)))
        lstm_mae = float(mean_absolute_error(y_test, lstm_prediction))
        lstm_r2 = float(r2_score(y_test, lstm_prediction))

        print(f"\nLSTM模型评估结果：")
        print(f"  RMSE: {lstm_rmse:.4f} kW")
        print(f"  MAE : {lstm_mae:.4f} kW")
        print(f"  R2  : {lstm_r2:.4f}")

        # ===== 绘制MLP vs LSTM预测对比图 =====
        fig, axes = plt.subplots(2, 1, figsize=(15, 10))

        # 上图：预测值vs真实值对比
        axes[0].plot(y_test.index, y_test.values, color="black", linewidth=1.0, label="真实值")
        axes[0].plot(y_test.index, baseline_prediction, linewidth=0.9,
                     label=f"MLP预测 (R2={baseline_metrics['R2']:.4f})")
        axes[0].plot(y_test.index, lstm_prediction, linewidth=0.9,
                     label=f"LSTM预测 (R2={lstm_r2:.4f})")
        axes[0].set_title("MLP vs LSTM 预测对比")
        axes[0].set_ylabel("功率（kW）")
        axes[0].grid(alpha=0.3)
        axes[0].legend()

        # 下图：LSTM训练损失曲线
        axes[1].plot(lstm_history.history["loss"], label="训练损失", linewidth=1.5)
        axes[1].plot(lstm_history.history["val_loss"], label="验证损失", linewidth=1.5)
        axes[1].set_title("LSTM 训练/验证损失曲线")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Loss（MSE）")
        axes[1].grid(alpha=0.3)
        axes[1].legend()

        fig.tight_layout()
        plt.show()

        # 对比MLP和LSTM性能
        print(f"\n>>> MLP vs LSTM 对比：")
        print(f"  模型        RMSE(kW)    MAE(kW)     R2")
        print(f"  MLP基准     {baseline_metrics['RMSE']:.4f}      {baseline_metrics['MAE']:.4f}      {baseline_metrics['R2']:.4f}")
        print(f"  LSTM        {lstm_rmse:.4f}      {lstm_mae:.4f}      {lstm_r2:.4f}")

        lstm_available = True  # 标记LSTM是否成功运行
    except ImportError:
        print("\n注意：Keras/TensorFlow未安装，跳过LSTM部分。")
        print("可使用 pip install tensorflow 安装后再运行。")
        lstm_available = False
        lstm_rmse, lstm_mae, lstm_r2 = None, None, None

    # ===== 10.2 使用DBSCAN进行基于密度的离群点检测 =====
    print("\n" + "=" * 30)
    print(">>> 10.2 DBSCAN离群点检测")
    from sklearn.cluster import DBSCAN  # DBSCAN基于密度的聚类算法
    from sklearn.preprocessing import StandardScaler as SS2  # 为DBSCAN单独导入标准化器

    # 使用任务5构建的时序特征数据进行DBSCAN聚类
    # DBSCAN对特征尺度敏感，先标准化
    dbscan_scaler = SS2()
    iforest_features_scaled = dbscan_scaler.fit_transform(feature_data_from_iforest)

    # 创建DBSCAN实例：
    # eps=0.5 - 邻域半径，决定两个样本被视为邻居的最大距离
    # min_samples=10 - 核心点至少需要10个邻域样本
    # eps和min_samples需根据数据分布调整
    dbscan = DBSCAN(eps=0.5, min_samples=10)
    # 在标准化特征上执行聚类：返回标签数组（-1表示噪声/离群点，>=0表示簇编号）
    dbscan_labels = dbscan.fit_predict(iforest_features_scaled)

    # 创建完整长度的标签系列（与hourly_series索引对齐）
    dbscan_flag = pd.Series(0, index=hourly_series.index, name="dbscan_outlier")
    # DBSCAN标签中-1表示噪声点（离群点），将其转换为1
    dbscan_noise = pd.Series(
        np.where(dbscan_labels == -1, 1, 0),
        index=feature_data_from_iforest.index
    )
    dbscan_flag.loc[dbscan_noise.index] = dbscan_noise

    # 绘制DBSCAN离群点检测结果
    outlier_points_dbscan = hourly_series[dbscan_flag == 1]
    fig, ax = plt.subplots(figsize=(15, 5))
    # 绘制原始序列
    ax.plot(hourly_series.index, hourly_series.values,
            color="tab:blue", linewidth=0.8, label="原始序列")
    # 绘制DBSCAN识别为离群的点（紫色散点）
    ax.scatter(
        outlier_points_dbscan.index,
        outlier_points_dbscan.values,
        color="purple",
        s=20,
        label="DBSCAN离群点",
    )
    ax.set_title("基于 DBSCAN 的离群点检测")
    ax.set_xlabel("时间")
    ax.set_ylabel("功率（kW）")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.show()

    # 统计DBSCAN各簇信息
    n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)  # 除噪声外的簇数
    n_noise = int((dbscan_labels == -1).sum())  # 噪声点数量
    print(f"\nDBSCAN离群点检测结果：")
    print(f"  发现的簇数量: {n_clusters}")
    print(f"  噪声点（离群点）数量: {n_noise}")
    print(f"  噪声点占比: {n_noise / len(dbscan_labels) * 100:.2f}%")

    # ===== 三种离群点检测方法对比 =====
    print(f"\n>>> 三种离群点检测方法对比：")
    print(f"  Z-score离群点数量:     {int(zscore_flag.sum())}")
    print(f"  孤立森林离群点数量:     {int(iforest_flag.sum())}")
    print(f"  DBSCAN离群点数量:       {int(dbscan_flag.sum())}")
    print(f"\n方法适用场景总结：")
    print(f"  Z-score  ：适合近似正态分布、单变量检测，计算快、可解释性强")
    print(f"  孤立森林：适合多维特征、不依赖分布假设，能捕获复杂异常模式")
    print(f"  DBSCAN  ：基于密度聚类，能发现任意形状的簇，适合空间分布异常检测")

    # 返回扩展任务的结果
    return {
        "lstm_available": lstm_available,
        "lstm_rmse": lstm_rmse, "lstm_mae": lstm_mae, "lstm_r2": lstm_r2,
        "dbscan_n_clusters": n_clusters, "dbscan_n_noise": n_noise,
        "dbscan_flag": dbscan_flag,
    }


# 任务10调用
ext_results = task10_extensions(
    hourly_series, iforest_feature_data,
    x_train_scaled_all, x_test_scaled_all, y_train, y_test,
    x_train, x_test
)

# ============================================================
# 所有任务执行完毕
# ============================================================
print("\n" + "=" * 60)
print("实验4：所有任务执行完毕！")
print("=" * 60)
print(f"最终模型性能总结：")
print(f"  基准MLP模型: RMSE={baseline_metrics['RMSE']:.4f}, MAE={baseline_metrics['MAE']:.4f}, R2={baseline_metrics['R2']:.4f}")
print(f"  过滤MLP模型: RMSE={filtered_metrics['RMSE']:.4f}, MAE={filtered_metrics['MAE']:.4f}, R2={filtered_metrics['R2']:.4f}")
if ext_results.get("lstm_available"):
    print(f"  LSTM模型   : RMSE={ext_results['lstm_rmse']:.4f}, MAE={ext_results['lstm_mae']:.4f}, R2={ext_results['lstm_r2']:.4f}")