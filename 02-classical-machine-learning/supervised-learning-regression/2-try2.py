# ==================== 任务1：数据加载与初步概览 ====================

def task1_load_and_overview():
    # 导入所需库：pandas用于数据读取与处理
    import pandas as pd
    # 忽略警告信息，避免输出干扰
    import warnings
    warnings.filterwarnings("ignore")

    # 指定数据集文件路径（分号分隔的CSV格式）
    file_path = "winequality-red.csv"
    # 使用pandas读取CSV文件，sep=";"表示以分号作为分隔符
    df = pd.read_csv(file_path, sep=";")

    # 打印任务标题，方便在Jupyter中定位输出
    print("=" * 50)
    print("任务1：数据加载与初步概览")
    print("=" * 50)

    # 调用info()查看数据基本信息：列名、非空数量、数据类型、内存占用
    print("\n>>> df.info()")
    df.info()

    # 调用describe()查看各特征的统计摘要：均值、标准差、最小值、四分位数、最大值
    print("\n>>> df.describe()")
    print(df.describe())

    # 调用head()查看前5行数据，快速了解数据样貌
    print("\n>>> df.head()")
    print(df.head())

    # 检查缺失值：isnull().sum()统计每列的缺失数量
    print("\n>>> 缺失值统计")
    missing = df.isnull().sum()
    print(missing)
    # 输出总缺失值数量，确认数据是否完整
    print(f"总缺失值数量: {missing.sum()}")

    # 返回DataFrame供后续任务使用
    return df

# 数据加载与初步概览，并将返回的DataFrame保存到全局变量df中
df = task1_load_and_overview()


# ==================== 任务2：可视化探索 ====================

def task2_visual_exploration(df):
    # 导入matplotlib用于绑图，seaborn用于更美观的统计可视化
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    # 设置中文字体，使图表标题和标签能正常显示中文
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 解决matplotlib中负号显示为方块的问题
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 50)
    print("任务2：可视化探索")
    print("=" * 50)

    # --- 绘制目标变量quality的直方图 ---
    plt.figure(figsize=(8, 5))  # 创建画布，设置宽8英寸高5英寸
    # 绘制直方图：bins=10将评分分为10个区间，edgecolor设边框色，alpha设透明度
    plt.hist(df["quality"], bins=10, edgecolor="black", alpha=0.75)
    plt.title("葡萄酒质量评分分布直方图")  # 设置图表标题
    plt.xlabel("质量评分（quality）")  # 设置x轴标签
    plt.ylabel("样本数量")  # 设置y轴标签
    plt.tight_layout()  # 自动调整子图间距，防止标签被裁剪
    plt.show()  # 在Jupyter中直接显示图表（不保存到文件）

    # --- 绘制特征与目标的相关性热力图 ---
    # 计算所有数值列之间的Pearson相关系数矩阵
    corr_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(11, 8))  # 创建较大画布以容纳热力图
    # 绘制热力图：cmap="coolwarm"表示蓝(-)到红(+)的色带，center=0以0为中心，annot=False不显示数值
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
    plt.title("特征与目标的相关性热力图")  # 设置标题
    plt.tight_layout()  # 自动调整间距
    plt.show()  # 直接在Jupyter中显示

    # 提取与quality列的相关系数，按绝对值降序排列，用于初步筛选重要特征
    corr_to_quality = corr_matrix["quality"].sort_values(ascending=False)
    print("\n>>> 各特征与quality的相关系数（降序）：")
    print(corr_to_quality)

    # --- 绘制酒精含量与质量的箱线图和散点图 ---
    # 散点图：直观展示alcohol与quality的分布关系
    plt.figure(figsize=(8, 5))
    # sns.scatterplot绘制散点图，alpha=0.6使点半透明避免重叠遮挡
    sns.scatterplot(data=df, x="alcohol", y="quality", alpha=0.6)
    plt.title("alcohol 与 quality 的散点图")
    plt.xlabel("alcohol（酒精含量）")
    plt.ylabel("quality")
    plt.tight_layout()
    plt.show()

    # 箱线图：按quality分组展示alcohol的分布，便于观察各组差异
    plt.figure(figsize=(8, 5))
    # sns.boxplot按quality的不同取值分组，绘制alcohol的箱线图
    sns.boxplot(data=df, x="quality", y="alcohol")
    plt.title("按 quality 分组的 alcohol 箱线图")
    plt.xlabel("quality")
    plt.ylabel("alcohol（酒精含量）")
    plt.tight_layout()
    plt.show()

    # 返回相关系数序列供后续分析使用
    return corr_to_quality

# 可视化探索，传入任务1返回的df，保存相关系数到全局变量
corr_to_quality = task2_visual_exploration(df)


# ==================== 任务3：特征工程 ====================

def task3_feature_engineering(df):
    import pandas as pd  # pandas用于DataFrame操作

    print("=" * 50)
    print("任务3：特征工程")
    print("=" * 50)

    # 复制原始DataFrame，避免修改原始数据
    df_new = df.copy()
    # 将quality转换为二分类标签good：评分>=6标记为1（好酒），否则标记为0（普通酒）
    df_new["good"] = (df_new["quality"] >= 6).astype(int)
    # 特征矩阵X：去除quality列（原始评分）和good列（二分类标签），仅保留理化指标
    X = df_new.drop(columns=["quality", "good"])
    # 回归目标向量y_reg：保留原始评分quality，用于线性回归任务
    y_reg = df_new["quality"]
    # 分类目标向量y_clf：使用二分类标签good，用于分类任务
    y_clf = df_new["good"]

    # 输出二分类标签的分布情况，检查类别是否平衡
    print("\n>>> 二分类标签 good 分布：")
    print(y_clf.value_counts())
    # 计算各类别占比，观察不平衡程度
    print("\n>>> good 标签占比：")
    print(y_clf.value_counts(normalize=True).sort_index())
    # 输出特征矩阵的形状，确认特征数量和样本数量
    print(f"\n特征矩阵X形状: {X.shape}")
    print(f"回归目标y_reg形状: {y_reg.shape}")
    print(f"分类目标y_clf形状: {y_clf.shape}")

    # 返回处理后的DataFrame、特征矩阵和两个目标向量
    return df_new, X, y_reg, y_clf

# 特征工程，保存结果到全局变量供后续任务使用
df_new, X, y_reg, y_clf = task3_feature_engineering(df)


# ==================== 任务4：数据划分与标准化 ====================

def task4_split_and_scale(X, y_reg, y_clf):
    # 导入train_test_split用于划分训练集和测试集
    from sklearn.model_selection import train_test_split
    # 导入StandardScaler用于Z-Score标准化
    from sklearn.preprocessing import StandardScaler

    print("=" * 50)
    print("任务4：数据划分与标准化")
    print("=" * 50)

    # 划分数据集：test_size=0.3表示30%作为测试集，random_state=42保证可重复性
    # stratify=y_clf确保训练集和测试集中各类别比例与原始数据一致（分层抽样）
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.3, random_state=42, stratify=y_clf
    )

    # 创建StandardScaler实例，用于Z-Score标准化（均值归零、标准差归一）
    scaler = StandardScaler()
    # 在训练集上拟合scaler并转换：fit_transform同时计算均值/标准差并进行转换
    X_train_scaled = scaler.fit_transform(X_train)
    # 在测试集上仅转换（使用训练集的均值和标准差），避免数据泄露
    X_test_scaled = scaler.transform(X_test)

    # 输出划分结果信息
    print(f"\n>>> 训练集样本数: {X_train.shape[0]}")
    print(f">>> 测试集样本数: {X_test.shape[0]}")
    print(f">>> 特征数量: {X_train.shape[1]}")
    # 输出标准化后的均值（应接近0）和标准差（应接近1），验证标准化效果
    print(f"\n>>> 标准化后训练集均值范围: [{X_train_scaled.mean(axis=0).min():.6f}, {X_train_scaled.mean(axis=0).max():.6f}]")
    print(f">>> 标准化后训练集标准差范围: [{X_train_scaled.std(axis=0).min():.6f}, {X_train_scaled.std(axis=0).max():.6f}]")

    # 将所有划分和标准化结果打包为字典，方便后续任务统一引用
    split_dict = {
        "X_train": X_train,              # 原始训练集特征（未标准化）
        "X_test": X_test,                # 原始测试集特征（未标准化）
        "X_train_scaled": X_train_scaled, # 标准化后的训练集特征
        "X_test_scaled": X_test_scaled,   # 标准化后的测试集特征
        "y_reg_train": y_reg_train,       # 训练集回归目标
        "y_reg_test": y_reg_test,         # 测试集回归目标
        "y_clf_train": y_clf_train,       # 训练集分类目标
        "y_clf_test": y_clf_test,         # 测试集分类目标
        "feature_names": X.columns.tolist(), # 特征名称列表，用于后续系数分析
        "scaler": scaler,                 # 标准化器实例，可供后续使用
    }

    # 返回数据字典
    return split_dict

# 数据划分与标准化，保存结果到全局变量
split_dict = task4_split_and_scale(X, y_reg, y_clf)


# ==================== 任务5：线性回归建模 ====================

def task5_linear_regression(split_dict):
    # 导入LinearRegression线性回归模型
    from sklearn.linear_model import LinearRegression

    print("=" * 50)
    print("任务5：线性回归建模")
    print("=" * 50)

    # 创建LinearRegression实例，使用默认参数（无正则化）
    lr_model = LinearRegression()
    # 在标准化后的训练集上拟合模型：学习各特征的权重系数和截距
    lr_model.fit(split_dict["X_train_scaled"], split_dict["y_reg_train"])

    # 输出模型截距（即所有特征为0时的预测值，相当于基线评分）
    print(f"\n>>> 模型截距（intercept）: {lr_model.intercept_:.4f}")
    # 输出模型是否成功拟合
    print(f">>> 模型已训练完成，特征数量: {len(lr_model.coef_)}")

    # 返回训练好的线性回归模型
    return lr_model

# 线性回归建模，保存模型到全局变量
lr_model = task5_linear_regression(split_dict)


# ==================== 任务6：回归模型评估 ====================

def task6_evaluate_regression(lr_model, split_dict):
    # 导入numpy用于数学计算
    import numpy as np
    # 导入评估指标函数
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # 导入pandas用于构建系数DataFrame
    import pandas as pd
    # 导入matplotlib用于绑图
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 50)
    print("任务6：回归模型评估")
    print("=" * 50)

    # 使用训练好的模型在标准化测试集上进行预测
    y_pred = lr_model.predict(split_dict["X_test_scaled"])
    # y_reg_test为测试集的真实评分值
    y_true = split_dict["y_reg_test"]

    # 计算均方误差MSE：预测值与真实值之差的平方的平均，衡量预测偏差大小
    mse_value = mean_squared_error(y_true, y_pred)
    # 计算均方根误差RMSE：MSE的平方根，单位与原始评分一致，更直观
    rmse_value = np.sqrt(mse_value)
    # 计算平均绝对误差MAE：预测值与真实值之差的绝对值的平均，对异常值不敏感
    mae_value = mean_absolute_error(y_true, y_pred)
    # 计算决定系数R²：1减去残差平方和与总方差之比，衡量模型解释方差的比例（0~1，越大越好）
    r2_value = r2_score(y_true, y_pred)

    # 输出各项评估指标
    print(f"\n>>> MSE  (均方误差)   : {mse_value:.4f}")
    print(f">>> RMSE (均方根误差)  : {rmse_value:.4f}")
    print(f">>> MAE  (平均绝对误差) : {mae_value:.4f}")
    print(f">>> R²   (决定系数)    : {r2_value:.4f}")

    # --- 绘制残差图：检查线性假设是否成立 ---
    # 计算残差 = 真实值 - 预测值，若线性假设成立，残差应随机分布在0附近
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    # 以预测值为x轴、残差为y轴绘制散点图，alpha=0.6半透明
    plt.scatter(y_pred, residuals, alpha=0.6)
    # 绘制红色虚线作为零残差参考线
    plt.axhline(y=0, color="red", linestyle="--")
    plt.title("残差图（预测值 vs 残差）")  # 设置标题
    plt.xlabel("预测值")  # x轴标签
    plt.ylabel("残差（真实值 - 预测值）")  # y轴标签
    plt.tight_layout()
    plt.show()  # 在Jupyter中直接显示

    # --- 输出特征系数，分析哪些理化指标对酒质影响最大 ---
    # 构建DataFrame：每行对应一个特征及其回归系数
    coef_df = pd.DataFrame({
        "feature": split_dict["feature_names"],  # 特征名称列
        "coef": lr_model.coef_                   # 线性回归系数列
    })
    # 添加绝对值系数列，用于按影响力大小排序
    coef_df["abs_coef"] = coef_df["coef"].abs()
    # 按绝对值系数降序排序，排在前面的特征对quality影响最大
    coef_df = coef_df.sort_values(by="abs_coef", ascending=False)

    # 输出系数排名，展示对酒质影响最大的理化指标
    print("\n>>> 特征系数（按绝对值降序，即对quality影响最大的特征）：")
    print(coef_df.to_string(index=False))

    # 返回评估结果字典和系数DataFrame
    return {
        "mse": mse_value,
        "rmse": rmse_value,
        "mae": mae_value,
        "r2": r2_value,
        "coef_df": coef_df,
        "y_pred": y_pred,
    }

# 回归模型评估，保存结果到全局变量
reg_result = task6_evaluate_regression(lr_model, split_dict)


# ==================== 任务7：分类数据准备 ====================

def task7_classification_data_prep(split_dict):
    print("=" * 50)
    print("任务7：分类数据准备")
    print("=" * 50)

    # 提取标准化后的特征矩阵X_train_scaled和X_test_scaled（已在前述任务中完成）
    # 提取二分类标签y_clf_train和y_clf_test（已在前述任务中完成）
    # 逻辑回归对特征尺度敏感，必须使用标准化后的特征
    # KNN基于距离计算，标准化可避免大尺度特征主导距离度量
    # 决策树虽不严格要求标准化，但统一标准化便于公平对比

    # 输出分类任务使用的数据概要
    print(f"\n>>> 训练集特征形状: {split_dict['X_train_scaled'].shape}")
    print(f">>> 测试集特征形状: {split_dict['X_test_scaled'].shape}")
    print(f">>> 训练集标签分布: {split_dict['y_clf_train'].value_counts().to_dict()}")
    print(f">>> 测试集标签分布: {split_dict['y_clf_test'].value_counts().to_dict()}")
    print("\n说明：分类任务使用与回归任务相同的标准化特征和划分方案，")
    print("      目标变量为二分类标签y_clf（good>=6为1，否则为0）。")

# 分类数据准备（无需返回值，仅打印确认信息）
task7_classification_data_prep(split_dict)


# ==================== 任务8：逻辑回归 ====================

def task8_logistic_regression(split_dict):
    # 导入逻辑回归模型
    from sklearn.linear_model import LogisticRegression
    # 导入分类评估指标
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_curve, auc
    )
    # 导入matplotlib用于绑图
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 50)
    print("任务8：逻辑回归")
    print("=" * 50)

    # 创建LogisticRegression实例：max_iter=300增加迭代次数确保收敛，random_state=42保证可重复性
    log_model = LogisticRegression(max_iter=300, random_state=42)
    # 在标准化训练集上拟合逻辑回归模型
    log_model.fit(split_dict["X_train_scaled"], split_dict["y_clf_train"])
    # 在标准化测试集上进行类别预测
    y_pred = log_model.predict(split_dict["X_test_scaled"])
    # 获取预测为正类（good=1）的概率，用于绘制ROC曲线
    y_prob = log_model.predict_proba(split_dict["X_test_scaled"])[:, 1]

    # 计算准确率：正确预测的比例
    acc = accuracy_score(split_dict["y_clf_test"], y_pred)
    # 计算精确率：预测为正类中真正为正类的比例（zero_division=0避免无正类预测时报错）
    pre = precision_score(split_dict["y_clf_test"], y_pred, zero_division=0)
    # 计算召回率：真正为正类中被正确预测的比例
    rec = recall_score(split_dict["y_clf_test"], y_pred, zero_division=0)
    # 计算F1分数：精确率和召回率的调和平均，综合衡量分类性能
    f1 = f1_score(split_dict["y_clf_test"], y_pred, zero_division=0)
    # 计算ROC曲线的假阳性率和真阳性率
    fpr, tpr, _ = roc_curve(split_dict["y_clf_test"], y_prob)
    # 计算AUC值：ROC曲线下面积，衡量分类器整体区分能力（0.5~1，越大越好）
    auc_value = auc(fpr, tpr)

    # 输出分类评估指标
    print(f"\n>>> 准确率 (Accuracy) : {acc:.4f}")
    print(f">>> 精确率 (Precision) : {pre:.4f}")
    print(f">>> 召回率 (Recall)    : {rec:.4f}")
    print(f">>> F1分数 (F1-Score)  : {f1:.4f}")
    print(f">>> AUC值              : {auc_value:.4f}")

    # --- 绘制ROC曲线 ---
    plt.figure(figsize=(7, 5))
    # 绘制ROC曲线：x轴为假阳性率FPR，y轴为真阳性率TPR
    plt.plot(fpr, tpr, label=f"逻辑回归 AUC={auc_value:.4f}", linewidth=2)
    # 绘制灰色虚线对角线（随机分类器的ROC曲线，AUC=0.5）
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("假阳性率（FPR）")  # x轴标签
    plt.ylabel("真阳性率（TPR）")  # y轴标签
    plt.title("逻辑回归 ROC曲线")  # 标题
    plt.legend(loc="lower right")  # 图例放在右下角
    plt.tight_layout()
    plt.show()  # 在Jupyter中直接显示

    # 返回模型和评估指标字典，供后续对比使用
    return log_model, {
        "model_name": "逻辑回归_默认",
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "auc": auc_value,
    }

# 逻辑回归，保存模型和指标到全局变量
log_model, logistic_metrics = task8_logistic_regression(split_dict)


# ==================== 任务9：决策树 ====================

def task9_decision_tree(split_dict):
    # 导入决策树分类器和可视化工具
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    # 导入分类评估指标
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_curve, auc
    )
    # 导入pandas用于构建特征重要性DataFrame
    import pandas as pd
    # 导入matplotlib用于绑图
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 50)
    print("任务9：决策树")
    print("=" * 50)

    # 创建DecisionTreeClassifier实例：
    # max_depth=5限制树的最大深度，防止过拟合（树太深会记住训练数据噪声）
    # min_samples_split=10要求节点至少有10个样本才允许分裂，进一步防过拟合
    # random_state=42保证可重复性
    dt_model = DecisionTreeClassifier(
        max_depth=5, min_samples_split=10, random_state=42
    )
    # 在标准化训练集上拟合决策树模型
    dt_model.fit(split_dict["X_train_scaled"], split_dict["y_clf_train"])
    # 在标准化测试集上进行类别预测
    y_pred = dt_model.predict(split_dict["X_test_scaled"])
    # 获取正类概率，用于ROC曲线和AUC计算
    y_prob = dt_model.predict_proba(split_dict["X_test_scaled"])[:, 1]

    # 计算各项评估指标
    acc = accuracy_score(split_dict["y_clf_test"], y_pred)
    pre = precision_score(split_dict["y_clf_test"], y_pred, zero_division=0)
    rec = recall_score(split_dict["y_clf_test"], y_pred, zero_division=0)
    f1 = f1_score(split_dict["y_clf_test"], y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(split_dict["y_clf_test"], y_prob)
    auc_value = auc(fpr, tpr)

    # 输出评估指标
    print(f"\n>>> 准确率 (Accuracy) : {acc:.4f}")
    print(f">>> 精确率 (Precision) : {pre:.4f}")
    print(f">>> 召回率 (Recall)    : {rec:.4f}")
    print(f">>> F1分数 (F1-Score)  : {f1:.4f}")
    print(f">>> AUC值              : {auc_value:.4f}")

    # --- 输出特征重要性 ---
    # 构建DataFrame：每行对应一个特征及其在决策树中的重要性得分
    importance_df = pd.DataFrame({
        "feature": split_dict["feature_names"],   # 特征名称列
        "importance": dt_model.feature_importances_  # 决策树计算的特征重要性列
    })
    # 按重要性降序排序，排在前面的特征对决策树分裂贡献最大
    importance_df = importance_df.sort_values(by="importance", ascending=False)
    print("\n>>> 决策树特征重要性（降序）：")
    print(importance_df.to_string(index=False))

    # --- 可视化决策树结构 ---
    plt.figure(figsize=(20, 10))  # 创建较大画布以容纳树结构
    # plot_tree绘制决策树：feature_names显示特征名，class_names显示类别名
    # filled=True用颜色填充节点表示类别，rounded=True用圆角矩形，max_depth=3仅展示前3层避免过于密集
    plot_tree(
        dt_model,
        feature_names=split_dict["feature_names"],
        class_names=["bad", "good"],
        filled=True,
        rounded=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("决策树可视化（仅展示前3层）")
    plt.tight_layout()
    plt.show()  # 在Jupyter中直接显示

    # 返回模型、评估指标和特征重要性DataFrame
    return dt_model, {
        "model_name": "决策树_默认",
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "auc": auc_value,
    }, importance_df

# 决策树，保存结果到全局变量
dt_model, tree_metrics, tree_importance_df = task9_decision_tree(split_dict)


# ==================== 任务10：K近邻（KNN） ====================

def task10_knn(split_dict):
    # 导入K近邻分类器
    from sklearn.neighbors import KNeighborsClassifier
    # 导入分类评估指标
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_curve, auc
    )
    # 导入matplotlib用于绑图
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 50)
    print("任务10：K近邻（KNN）")
    print("=" * 50)

    # 创建KNeighborsClassifier实例：n_neighbors=5表示取最近的5个邻居投票决定类别
    # KNN基于欧氏距离计算邻近度，因此特征标准化至关重要（否则大尺度特征主导距离）
    knn_model = KNeighborsClassifier(n_neighbors=5)
    # 在标准化训练集上拟合KNN模型（实际上KNN仅存储训练数据，预测时才计算距离）
    knn_model.fit(split_dict["X_train_scaled"], split_dict["y_clf_train"])
    # 在标准化测试集上进行类别预测
    y_pred = knn_model.predict(split_dict["X_test_scaled"])
    # 获取正类概率（KNN通过邻居中正类比例计算概率）
    y_prob = knn_model.predict_proba(split_dict["X_test_scaled"])[:, 1]

    # 计算各项评估指标
    acc = accuracy_score(split_dict["y_clf_test"], y_pred)
    pre = precision_score(split_dict["y_clf_test"], y_pred, zero_division=0)
    rec = recall_score(split_dict["y_clf_test"], y_pred, zero_division=0)
    f1 = f1_score(split_dict["y_clf_test"], y_pred, zero_division=0)
    fpr, tpr, _ = roc_curve(split_dict["y_clf_test"], y_prob)
    auc_value = auc(fpr, tpr)

    # 输出评估指标
    print(f"\n>>> 准确率 (Accuracy) : {acc:.4f}")
    print(f">>> 精确率 (Precision) : {pre:.4f}")
    print(f">>> 召回率 (Recall)    : {rec:.4f}")
    print(f">>> F1分数 (F1-Score)  : {f1:.4f}")
    print(f">>> AUC值              : {auc_value:.4f}")

    # 返回模型和评估指标字典
    return knn_model, {
        "model_name": "KNN_默认",
        "accuracy": acc,
        "precision": pre,
        "recall": rec,
        "f1": f1,
        "auc": auc_value,
    }

# K近邻，保存结果到全局变量
knn_model, knn_metrics = task10_knn(split_dict)


# ==================== 任务11：交叉验证与网格搜索 ====================

def task11_grid_search(split_dict, logistic_metrics, tree_metrics, knn_metrics):
    # 导入网格搜索和交叉验证工具
    from sklearn.model_selection import GridSearchCV
    # 导入三个分类模型（与默认模型相同的类，但需重新创建以搜索最优超参数）
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    # 导入评估指标用于计算调优后的性能
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_curve, auc
    )
    # 导入pandas用于构建对比表格
    import pandas as pd
    # 导入numpy和matplotlib用于绑图
    import numpy as np
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 50)
    print("任务11：交叉验证与网格搜索")
    print("=" * 50)

    # 提取标准化后的训练集特征和分类目标
    X_train_scaled = split_dict["X_train_scaled"]
    y_train = split_dict["y_clf_train"]
    X_test_scaled = split_dict["X_test_scaled"]
    y_test = split_dict["y_clf_test"]

    # --- 逻辑回归网格搜索 ---
    # 定义超参数搜索空间：C为正则化强度的倒数（越小正则化越强），penalty=l2使用L2正则化
    logistic_grid = {
        "C": [0.01, 0.1, 1, 10, 100],  # 尝试5个不同的正则化强度
        "penalty": ["l2"],               # 仅使用L2正则化（与lbfgs求解器兼容）
        "solver": ["lbfgs"],             # 使用lbfgs优化算法（适合小数据集）
    }
    # 创建GridSearchCV实例：scoring="f1"以F1分数为优化目标，cv=5进行5折交叉验证
    # n_jobs=-1使用所有CPU核心加速搜索
    logistic_gs = GridSearchCV(
        estimator=LogisticRegression(max_iter=500, random_state=42),
        param_grid=logistic_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    # 在训练集上执行网格搜索：遍历所有参数组合，用5折CV评估每组参数
    logistic_gs.fit(X_train_scaled, y_train)
    # 输出逻辑回归的最优参数和最优CV得分
    print(f"\n>>> 逻辑回归最优参数: {logistic_gs.best_params_}")
    print(f">>> 逻辑回归最优CV F1分数: {logistic_gs.best_score_:.4f}")

    # --- 决策树网格搜索 ---
    # 定义超参数搜索空间：max_depth控制树深度，min_samples_split控制分裂最小样本数
    # min_samples_leaf控制叶节点最小样本数
    dt_grid = {
        "max_depth": [3, 5, 7, None],         # None表示不限制深度
        "min_samples_split": [2, 5, 10, 20],  # 尝试4个分裂阈值
        "min_samples_leaf": [1, 2, 4, 8],     # 尝试4个叶节点阈值
    }
    dt_gs = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=42),
        param_grid=dt_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    dt_gs.fit(X_train_scaled, y_train)
    print(f"\n>>> 决策树最优参数: {dt_gs.best_params_}")
    print(f">>> 决策树最优CV F1分数: {dt_gs.best_score_:.4f}")

    # --- KNN网格搜索 ---
    # 定义超参数搜索空间：n_neighbors为邻居数量K，weights为权重方式
    knn_grid = {
        "n_neighbors": [3, 5, 7, 9, 11],      # 尝试5个K值
        "weights": ["uniform", "distance"],    # uniform:等权重投票; distance:按距离加权投票
    }
    knn_gs = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=knn_grid,
        scoring="f1",
        cv=5,
        n_jobs=-1,
    )
    knn_gs.fit(X_train_scaled, y_train)
    print(f"\n>>> KNN最优参数: {knn_gs.best_params_}")
    print(f">>> KNN最优CV F1分数: {knn_gs.best_score_:.4f}")

    # --- 获取调优后的最优模型 ---
    best_logistic = logistic_gs.best_estimator_
    best_tree = dt_gs.best_estimator_
    best_knn = knn_gs.best_estimator_

    # --- 计算调优后模型在测试集上的评估指标 ---
    # 逻辑回归调优后评估
    y_pred_log = best_logistic.predict(X_test_scaled)
    y_prob_log = best_logistic.predict_proba(X_test_scaled)[:, 1]
    tuned_log_metrics = {
        "model_name": "逻辑回归_调优",
        "accuracy": accuracy_score(y_test, y_pred_log),
        "precision": precision_score(y_test, y_pred_log, zero_division=0),
        "recall": recall_score(y_test, y_pred_log, zero_division=0),
        "f1": f1_score(y_test, y_pred_log, zero_division=0),
        "auc": auc(*roc_curve(y_test, y_prob_log)[:2]),
    }

    # 决策树调优后评估
    y_pred_tree = best_tree.predict(X_test_scaled)
    y_prob_tree = best_tree.predict_proba(X_test_scaled)[:, 1]
    tuned_tree_metrics = {
        "model_name": "决策树_调优",
        "accuracy": accuracy_score(y_test, y_pred_tree),
        "precision": precision_score(y_test, y_pred_tree, zero_division=0),
        "recall": recall_score(y_test, y_pred_tree, zero_division=0),
        "f1": f1_score(y_test, y_pred_tree, zero_division=0),
        "auc": auc(*roc_curve(y_test, y_prob_tree)[:2]),
    }

    # KNN调优后评估
    y_pred_knn = best_knn.predict(X_test_scaled)
    y_prob_knn = best_knn.predict_proba(X_test_scaled)[:, 1]
    tuned_knn_metrics = {
        "model_name": "KNN_调优",
        "accuracy": accuracy_score(y_test, y_pred_knn),
        "precision": precision_score(y_test, y_pred_knn, zero_division=0),
        "recall": recall_score(y_test, y_pred_knn, zero_division=0),
        "f1": f1_score(y_test, y_pred_knn, zero_division=0),
        "auc": auc(*roc_curve(y_test, y_prob_knn)[:2]),
    }

    # --- 绘制调优前后性能对比图（分组柱状图） ---
    metrics_names = ["Accuracy", "F1", "AUC"]
    models_names = ["逻辑回归", "决策树", "KNN"]
    default_vals = [
        [logistic_metrics["accuracy"], logistic_metrics["f1"], logistic_metrics["auc"]],
        [tree_metrics["accuracy"], tree_metrics["f1"], tree_metrics["auc"]],
        [knn_metrics["accuracy"], knn_metrics["f1"], knn_metrics["auc"]],
    ]
    tuned_vals = [
        [tuned_log_metrics["accuracy"], tuned_log_metrics["f1"], tuned_log_metrics["auc"]],
        [tuned_tree_metrics["accuracy"], tuned_tree_metrics["f1"], tuned_tree_metrics["auc"]],
        [tuned_knn_metrics["accuracy"], tuned_knn_metrics["f1"], tuned_knn_metrics["auc"]],
    ]
    x = np.arange(len(metrics_names))
    width = 0.12
    fig, ax = plt.subplots(figsize=(11, 6))
    colors_default = ["#4ECDC4", "#FFD93D", "#FF6B6B"]
    colors_tuned = ["#2E8B8B", "#C4A800", "#CC3333"]
    for i, model in enumerate(models_names):
        offset = (i - 1) * 2 * width
        bars_d = ax.bar(x + offset, default_vals[i], width,
                    label=f"{model} 默认", color=colors_default[i], edgecolor="black", linewidth=0.5)
        bars_t = ax.bar(x + offset + width, tuned_vals[i], width,
                    label=f"{model} 调优", color=colors_tuned[i], edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.set_ylabel("指标值")
    ax.set_title("调优前后性能对比")
    ax.legend(loc="lower right", fontsize=7)
    ax.set_ylim(0, 1.08)
    for patch in ax.patches:
        h = patch.get_height()
        if h > 0.01:
            ax.annotate(f"{h:.3f}", (patch.get_x() + patch.get_width() / 2, h),
                        ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    plt.show()

    # 返回最优模型字典和对比表格
    best_models = {
        "logistic": best_logistic,
        "tree": best_tree,
        "knn": best_knn,
    }
    return best_models, tuned_log_metrics, tuned_tree_metrics, tuned_knn_metrics

# 交叉验证与网格搜索，传入之前保存的默认模型指标用于对比
best_models, tuned_log_metrics, tuned_tree_metrics, tuned_knn_metrics = task11_grid_search(
    split_dict, logistic_metrics, tree_metrics, knn_metrics
)


# ==================== 任务12：模型综合对比 ====================

def task12_model_comparison(best_models, split_dict):
    # 导入time用于测量训练和预测耗时
    import time
    # 导入评估指标
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_curve, auc,
        confusion_matrix, ConfusionMatrixDisplay
    )
    # 导入pandas用于构建对比表格
    import pandas as pd
    import numpy as np
    # 导入matplotlib用于绑图
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 50)
    print("任务12：模型综合对比")
    print("=" * 50)

    # 提取标准化后的训练集和测试集数据
    X_train_scaled = split_dict["X_train_scaled"]
    X_test_scaled = split_dict["X_test_scaled"]
    y_train = split_dict["y_clf_train"]
    y_test = split_dict["y_clf_test"]

    # 定义模型名称映射，用于输出友好名称
    model_map = {
        "逻辑回归_最优": best_models["logistic"],
        "决策树_最优": best_models["tree"],
        "KNN_最优": best_models["knn"],
    }

    # 收集每个模型的性能数据
    records = []
    for model_name, model_obj in model_map.items():
        # 记录训练开始时间
        start_train = time.perf_counter()
        # 在训练集上重新拟合最优模型（确保计时准确）
        model_obj.fit(X_train_scaled, y_train)
        # 计算训练耗时
        train_time = time.perf_counter() - start_train

        # 记录预测开始时间
        start_pred = time.perf_counter()
        # 在测试集上进行预测
        y_pred = model_obj.predict(X_test_scaled)
        # 计算预测耗时
        pred_time = time.perf_counter() - start_pred

        # 获取正类概率用于AUC计算
        y_prob = model_obj.predict_proba(X_test_scaled)[:, 1]

        # 计算准确率
        acc = accuracy_score(y_test, y_pred)
        # 计算F1分数
        f1 = f1_score(y_test, y_pred, zero_division=0)
        # 计算AUC值
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_value = auc(fpr, tpr)

        # 记录该模型的所有性能数据
        records.append({
            "模型": model_name,
            "准确率": acc,
            "F1分数": f1,
            "AUC值": auc_value,
            "训练时间(秒)": train_time,
            "预测时间(秒)": pred_time,
        })

        # --- 绘制混淆矩阵 ---
        cm = confusion_matrix(y_test, y_pred)  # 计算混淆矩阵
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["bad(0)", "good(1)"]
        )
        disp.plot(cmap="Blues")  # 使用蓝色色带绘制
        plt.title(f"{model_name} 混淆矩阵")
        plt.tight_layout()
        plt.show()  # 在Jupyter中直接显示

    # 构建综合对比表格，按F1和AUC降序排列
    benchmark_df = pd.DataFrame(records).sort_values(
        by=["F1分数", "AUC值"], ascending=False
    )
    # --- 绘制最优模型综合对比图 ---
    metrics_cols = ["准确率", "F1分数", "AUC值"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    # 左图：分类性能分组柱状图
    x = np.arange(len(metrics_cols))
    width = 0.25
    colors = ["#4ECDC4", "#FFD93D", "#FF6B6B"]
    for i, (_, row) in enumerate(benchmark_df.iterrows()):
        vals = [row["准确率"], row["F1分数"], row["AUC值"]]
        bars = axes[0].bar(x + i * width, vals, width, label=row["模型"],
                           color=colors[i % 3], edgecolor="black", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0.01:
                axes[0].annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                                 ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x + width)
    axes[0].set_xticklabels(metrics_cols)
    axes[0].set_ylabel("指标值")
    axes[0].set_title("最优模型分类性能对比")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].set_ylim(0, 1.08)
    # 右图：运行时间柱状图
    time_cols = ["训练时间(秒)", "预测时间(秒)"]
    x2 = np.arange(len(time_cols))
    for i, (_, row) in enumerate(benchmark_df.iterrows()):
        vals2 = [row["训练时间(秒)"], row["预测时间(秒)"]]
        bars2 = axes[1].bar(x2 + i * width, vals2, width, label=row["模型"],
                            color=colors[i % 3], edgecolor="black", linewidth=0.5)
        for bar in bars2:
            h = bar.get_height()
            if h > 0.0001:
                axes[1].annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                                 ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(x2 + width)
    axes[1].set_xticklabels(time_cols)
    axes[1].set_ylabel("时间（秒）")
    axes[1].set_title("最优模型运行时间对比")
    axes[1].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.show()

    # 返回对比表格
    return benchmark_df

# 模型综合对比，保存结果到全局变量
benchmark_df = task12_model_comparison(best_models, split_dict)


# ==================== 任务13：结果分析 ====================

def task13_analysis(reg_result, benchmark_df, y_clf):
    # 导入pandas用于数据分析
    import pandas as pd

    print("=" * 50)
    print("任务13：结果分析")
    print("=" * 50)

    # --- 回归任务关键因素分析 ---
    # 从回归评估结果中提取系数DataFrame
    coef_df = reg_result["coef_df"]
    # 输出对葡萄酒质量影响最大的5个理化指标（按系数绝对值排序）
    print("\n>>> 回归任务中影响葡萄酒质量的关键因素（按系数绝对值降序TOP5）：")
    print(coef_df.head(5)[["feature", "coef"]].to_string(index=False))
    # 分析各关键因素的影响方向（正系数表示正相关，负系数表示负相关）
    print("\n解读：系数为正表示该指标越高quality越高，系数为负则相反。")
    print("例如alcohol系数通常为正，说明酒精含量越高葡萄酒评分倾向更高；")
    print("volatile acidity系数通常为负，说明挥发性酸越高评分倾向更低。")

    # --- 分类算法对比分析 ---
    # 从综合对比表格中提取最优模型信息
    best_row = benchmark_df.iloc[0]  # 取F1最高的模型行
    print(f"\n>>> 分类算法推荐：{best_row['模型']}")
    print(f"    该模型在测试集上F1={best_row['F1分数']:.4f}, AUC={best_row['AUC值']:.4f}")
    print("推荐理由：F1分数综合了精确率和召回率，AUC衡量整体区分能力，")
    print("两者均高的模型更适合本数据集的分类任务。")

    # --- 类别不平衡影响讨论 ---
    # 计算二分类标签的类别比例
    class_ratio = y_clf.value_counts(normalize=True).sort_index()
    print(f"\n>>> 类别分布比例（0=bad, 1=good）：")
    print(class_ratio.to_dict())
    print("\n类别不平衡的影响讨论：")
    print("1. 若good类样本远多于bad类，准确率可能虚高（模型偏向预测多数类）。")
    print("2. 不平衡情况下，应重点关注F1分数、召回率和AUC，而非仅看准确率。")
    print("3. 可能的缓解措施：使用class_weight='balanced'、过采样（SMOTE）、欠采样等。")
    print("4. 逻辑回归可通过class_weight参数自动调整类别权重来应对不平衡。")

# 结果分析
task13_analysis(reg_result, benchmark_df, y_clf)


# ==================== 任务14：扩展思考 ====================

def task14_extensions(split_dict):
    # 导入岭回归和Lasso回归模型，用于与线性回归对比
    from sklearn.linear_model import Ridge, Lasso
    # 导入回归评估指标
    from sklearn.metrics import r2_score, mean_squared_error
    # 导入随机森林分类器（集成学习方法）
    from sklearn.ensemble import RandomForestClassifier
    # 导入分类评估指标
    from sklearn.metrics import (
        accuracy_score, f1_score, roc_curve, auc
    )
    # 导入numpy用于数学计算
    import numpy as np
    # 导入pandas用于构建对比表格
    import pandas as pd
    # 导入matplotlib用于绑图
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    print("=" * 50)
    print("任务14：扩展思考（可选）")
    print("=" * 50)

    # --- 岭回归 vs 线性回归 ---
    # 创建Ridge实例：alpha=1.0控制L2正则化强度（alpha越大正则化越强）
    ridge_model = Ridge(alpha=1.0, random_state=42)
    # 在标准化训练集上拟合岭回归模型
    ridge_model.fit(split_dict["X_train_scaled"], split_dict["y_reg_train"])
    # 在测试集上进行预测
    ridge_pred = ridge_model.predict(split_dict["X_test_scaled"])
    # 计算岭回归的R²决定系数
    ridge_r2 = r2_score(split_dict["y_reg_test"], ridge_pred)
    # 计算岭回归的RMSE
    ridge_rmse = np.sqrt(mean_squared_error(split_dict["y_reg_test"], ridge_pred))

    # --- Lasso回归 vs 线性回归 ---
    # 创建Lasso实例：alpha=0.001控制L1正则化强度（较小值使模型接近普通线性回归）
    # max_iter=5000增加迭代次数确保收敛
    lasso_model = Lasso(alpha=0.001, random_state=42, max_iter=5000)
    # 在标准化训练集上拟合Lasso模型
    lasso_model.fit(split_dict["X_train_scaled"], split_dict["y_reg_train"])
    # 在测试集上进行预测
    lasso_pred = lasso_model.predict(split_dict["X_test_scaled"])
    # 计算Lasso的R²决定系数
    lasso_r2 = r2_score(split_dict["y_reg_test"], lasso_pred)
    # 计算Lasso的RMSE
    lasso_rmse = np.sqrt(mean_squared_error(split_dict["y_reg_test"], lasso_pred))

    # 构建回归模型对比表格，包含线性回归、岭回归、Lasso的R²和RMSE
    reg_compare_df = pd.DataFrame([
        {"模型": "线性回归", "R²": reg_result["r2"], "RMSE": reg_result["rmse"]},
        {"模型": "岭回归", "R²": ridge_r2, "RMSE": ridge_rmse},
        {"模型": "Lasso回归", "R²": lasso_r2, "RMSE": lasso_rmse},
    ])
    # --- 绘制回归模型对比图（R²和RMSE） ---
    reg_models = reg_compare_df["模型"].tolist()
    x = np.arange(len(reg_models))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    bars_r2 = axes[0].bar(x, reg_compare_df["R²"], width,
                          color=["#4ECDC4", "#FFD93D", "#FF6B6B"], edgecolor="black", linewidth=0.5)
    for bar in bars_r2:
        h = bar.get_height()
        axes[0].annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                         ha="center", va="bottom", fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(reg_models)
    axes[0].set_ylabel("R²")
    axes[0].set_title("回归模型 R² 对比")
    bars_rmse = axes[1].bar(x, reg_compare_df["RMSE"], width,
                            color=["#4ECDC4", "#FFD93D", "#FF6B6B"], edgecolor="black", linewidth=0.5)
    for bar in bars_rmse:
        h = bar.get_height()
        axes[1].annotate(f"{h:.4f}", (bar.get_x() + bar.get_width() / 2, h),
                         ha="center", va="bottom", fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(reg_models)
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("回归模型 RMSE 对比")
    plt.tight_layout()
    plt.show()

    # --- 随机森林（集成学习） ---
    # 创建RandomForestClassifier实例：
    # n_estimators=300表示使用300棵决策树，max_depth=10限制树深度防止过拟合
    # min_samples_split=5要求节点至少5个样本才分裂，random_state=42保证可重复性
    rf_model = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_split=5, random_state=42
    )
    # 在标准化训练集上拟合随机森林模型
    rf_model.fit(split_dict["X_train_scaled"], split_dict["y_clf_train"])
    # 在测试集上进行预测
    rf_pred = rf_model.predict(split_dict["X_test_scaled"])
    # 计算随机森林的准确率和F1分数
    rf_acc = accuracy_score(split_dict["y_clf_test"], rf_pred)
    rf_f1 = f1_score(split_dict["y_clf_test"], rf_pred, zero_division=0)
    # 获取正类概率用于AUC计算
    rf_prob = rf_model.predict_proba(split_dict["X_test_scaled"])[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(split_dict["y_clf_test"], rf_prob)
    rf_auc = auc(rf_fpr, rf_tpr)

    # --- XGBoost（集成学习，可能需要额外安装） ---
    # 尝试导入XGBoost，若未安装则跳过
    xgb_acc, xgb_f1, xgb_auc = None, None, None
    try:
        from xgboost import XGBClassifier
        # 创建XGBClassifier实例：
        # n_estimators=300表示300轮提升，max_depth=4限制树深度
        # learning_rate=0.05控制每轮贡献权重，subsample=0.9随机采样90%样本防过拟合
        # colsample_bytree=0.9随机采样90%特征防过拟合，eval_metric="logloss"使用logloss作为评估指标
        xgb_model = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9,
            eval_metric="logloss", random_state=42,
        )
        # 在标准化训练集上拟合XGBoost模型
        xgb_model.fit(split_dict["X_train_scaled"], split_dict["y_clf_train"])
        # 在测试集上进行预测
        xgb_pred = xgb_model.predict(split_dict["X_test_scaled"])
        # 计算XGBoost的准确率和F1分数
        xgb_acc = accuracy_score(split_dict["y_clf_test"], xgb_pred)
        xgb_f1 = f1_score(split_dict["y_clf_test"], xgb_pred, zero_division=0)
        # 获取正类概率用于AUC计算
        xgb_prob = xgb_model.predict_proba(split_dict["X_test_scaled"])[:, 1]
        xgb_fpr, xgb_tpr, _ = roc_curve(split_dict["y_clf_test"], xgb_prob)
        xgb_auc = auc(xgb_fpr, xgb_tpr)
    except ImportError:
        # 若XGBoost未安装，输出提示信息
        print("\n注意：XGBoost未安装，跳过XGBoost评估。可使用 pip install xgboost 安装。")

    # 构建集成学习对比表格
    ensemble_compare_df = pd.DataFrame([
        {"模型": "随机森林", "准确率": rf_acc, "F1分数": rf_f1, "AUC值": rf_auc},
        {"模型": "XGBoost", "准确率": xgb_acc, "F1分数": xgb_f1, "AUC值": xgb_auc},
    ])
    # --- 绘制集成学习方法对比图 ---
    ens_models = ensemble_compare_df["模型"].tolist()
    metrics_cols = ["准确率", "F1分数", "AUC值"]
    x = np.arange(len(metrics_cols))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4ECDC4", "#FFD93D"]
    for i, (_, row) in enumerate(ensemble_compare_df.iterrows()):
        vals = [row["准确率"], row["F1分数"], row["AUC值"]]
        bars = ax.bar(x + i * width, vals, width, label=row["模型"],
                      color=colors[i], edgecolor="black", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h is not None and h > 0.01:
                ax.annotate(f"{h:.3f}", (bar.get_x() + bar.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(metrics_cols)
    ax.set_ylabel("指标值")
    ax.set_title("集成学习方法性能对比")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(0, 1.08)
    plt.tight_layout()
    plt.show()

    # 返回扩展结果
    return {
        "ridge_r2": ridge_r2, "ridge_rmse": ridge_rmse,
        "lasso_r2": lasso_r2, "lasso_rmse": lasso_rmse,
        "rf_acc": rf_acc, "rf_f1": rf_f1, "rf_auc": rf_auc,
        "xgb_acc": xgb_acc, "xgb_f1": xgb_f1, "xgb_auc": xgb_auc,
    }

# 扩展思考，保存结果到全局变量
ext_result = task14_extensions(split_dict)