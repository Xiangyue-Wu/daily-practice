import os
import sys
import site
import warnings
from datetime import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


# 在导入numpy/scipy前移除用户site-packages，避免污染conda环境中的二进制包
USER_SITE = site.getusersitepackages()
if USER_SITE in sys.path:
    sys.path.remove(USER_SITE)

# 将项目内本地依赖目录加入搜索路径，便于脚本和notebook复用
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
LOCAL_PKG_DIR = os.path.join(SCRIPT_DIR, ".py39_pkgs")
if os.path.isdir(LOCAL_PKG_DIR) and LOCAL_PKG_DIR not in sys.path:
    sys.path.insert(0, LOCAL_PKG_DIR)

# 在非交互终端中尽量关闭输出缓冲，避免“程序在跑但迟迟没显示日志”
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True, write_through=True)

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

try:
    import torch

    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


SAMPLE_N = None
DATA_PATH = r"data/online_retail_II.csv"
USE_GPU_IF_AVAILABLE = True
SHOW_PLOTS = True
# 全量数据下约有3.7万笔事务，0.01表示至少在约370笔订单中出现，兼顾效率与业务意义
MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.5
# 课程实验以可解释性为主，限制到3项集通常最均衡
MAX_ITEMSET_LEN = 3
OUTPUT_DIR = r"outputs/re3"


@dataclass
class TorchKMeansModel:
    cluster_centers_: np.ndarray
    labels_: np.ndarray
    inertia_: float
    device: str


def is_notebook() -> bool:
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:
        return False


def configure_chinese_plot() -> None:
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    return None


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)
    return None


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return None


def save_current_figure(file_name: str) -> None:
    ensure_output_dir()
    save_path = os.path.join(OUTPUT_DIR, file_name)
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    log_progress(f"图像已保存：{save_path}")
    return None


def finish_plot(show_plots: bool) -> None:
    if show_plots:
        plt.show()
    else:
        plt.close()
    return None


def get_gpu_status() -> Dict[str, Any]:
    status = {
        "torch_installed": HAS_TORCH,
        "cuda_available": False,
        "device": "cpu",
        "device_name": "CPU",
    }
    if HAS_TORCH and torch.cuda.is_available():
        status["cuda_available"] = True
        status["device"] = "cuda"
        status["device_name"] = torch.cuda.get_device_name(0)
    return status


def torch_kmeans_fit_predict(
    data: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
    max_iter: int = 100,
    tol: float = 1e-4,
) -> TorchKMeansModel:
    if not HAS_TORCH or not torch.cuda.is_available():
        raise RuntimeError("当前环境不可用Torch CUDA。")

    device = torch.device("cuda")
    x = torch.tensor(data, dtype=torch.float32, device=device)
    n_samples = x.shape[0]

    if n_clusters >= n_samples:
        raise ValueError("聚类数必须小于样本数。")

    gen = torch.Generator(device=device)
    gen.manual_seed(random_state)
    init_idx = torch.randperm(n_samples, generator=gen, device=device)[:n_clusters]
    centers = x[init_idx].clone()

    for _ in range(max_iter):
        distances = torch.cdist(x, centers)
        labels = torch.argmin(distances, dim=1)

        new_centers = []
        for cluster_id in range(n_clusters):
            cluster_points = x[labels == cluster_id]
            if cluster_points.numel() == 0:
                new_centers.append(centers[cluster_id])
            else:
                new_centers.append(cluster_points.mean(dim=0))
        new_centers = torch.stack(new_centers)

        shift = torch.norm(new_centers - centers, dim=1).max().item()
        centers = new_centers
        if shift < tol:
            break

    final_distances = torch.cdist(x, centers)
    final_labels = torch.argmin(final_distances, dim=1)
    sq_dist = (x - centers[final_labels]) ** 2
    inertia = float(sq_dist.sum().item())

    return TorchKMeansModel(
        cluster_centers_=centers.detach().cpu().numpy(),
        labels_=final_labels.detach().cpu().numpy(),
        inertia_=inertia,
        device=str(device),
    )


def task1_load_and_clean(file_path: str, sample_n: int = None) -> pd.DataFrame:
    log_progress(f"开始读取CSV文件：{file_path}")
    data = pd.read_csv(file_path, nrows=sample_n)
    log_progress(f"CSV读取完成，原始数据行数：{len(data)}")
    data = data.rename(columns={"Customer ID": "CustomerID", "Price": "UnitPrice"})

    print("\n" + "=" * 60)
    print("任务1：数据加载与概览")
    print("=" * 60)
    print("【数据维度】行数 × 列数：", data.shape)
    print("\n【info() 输出 — 每列的数据类型和非空计数】")
    data.info()
    print("\n【describe() 输出 — 数值列的统计描述】")
    print(data.describe(include="all").T.head(20))
    print("\n【head() 输出 — 前5行原始数据】")
    print(data.head())

    cleaned = data.copy()
    cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
    cleaned["Quantity"] = pd.to_numeric(cleaned["Quantity"], errors="coerce")
    cleaned["UnitPrice"] = pd.to_numeric(cleaned["UnitPrice"], errors="coerce")
    cleaned["CustomerID"] = pd.to_numeric(cleaned["CustomerID"], errors="coerce")
    cleaned["InvoiceNo"] = cleaned["Invoice"].astype(str)
    cleaned = cleaned.dropna(subset=["CustomerID"])
    cleaned = cleaned[cleaned["Quantity"] > 0]
    cleaned = cleaned[cleaned["UnitPrice"] > 0]
    cleaned = cleaned.dropna(subset=["InvoiceDate"])

    log_progress("数据清洗完成")
    print("\n【清洗后数据维度】", cleaned.shape)
    print("清洗前数据量：", len(data))
    print("清洗后数据量：", len(cleaned))
    print("删除无效记录数：", len(data) - len(cleaned))
    return cleaned


def task2_build_transactions(cleaned: pd.DataFrame, item_col: str = "StockCode") -> Tuple[List[List[str]], pd.DataFrame]:
    print("\n" + "=" * 60)
    print("任务2：构造购物篮事务数据")
    print("=" * 60)
    log_progress("开始按订单聚合商品并构造购物篮事务")

    use_col = "StockCode" if item_col not in cleaned.columns else item_col
    temp = cleaned[["InvoiceNo", use_col]].copy()
    temp[use_col] = temp[use_col].astype(str).str.strip()
    temp = temp[temp[use_col] != ""]

    grouped = temp.groupby("InvoiceNo")[use_col].apply(lambda x: list(pd.unique(x)))
    transactions = grouped.tolist()

    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    one_hot_df = pd.DataFrame(te_array, columns=te.columns_)
    log_progress("事务数据One-Hot编码完成")

    print("订单事务数（购物篮数）：", len(transactions))
    print("One-Hot矩阵维度（事务数 × 商品种类数）：", one_hot_df.shape)
    print("示例事务（前3笔订单）：")
    for i, tx in enumerate(transactions[:3]):
        print(f"  订单{i + 1}: {tx}")
    return transactions, one_hot_df


def task3_mine_frequent_itemsets(
    one_hot_df: pd.DataFrame,
    min_support: float = 0.01,
    max_len: int = 3,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("任务3：寻找频繁项集")
    print("=" * 60)

    if one_hot_df.empty:
        print("警告：One-Hot数据为空，无法挖掘频繁项集。")
        return pd.DataFrame()

    log_progress(f"开始执行Apriori，min_support={min_support}，max_len={max_len}")
    frequent_itemsets = apriori(
        one_hot_df,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )
    log_progress("Apriori执行完成")
    frequent_itemsets = frequent_itemsets.sort_values(by="support", ascending=False).reset_index(drop=True)

    print("最小支持度阈值：", min_support)
    print("最大频繁项集长度：", max_len)
    print("找到的频繁项集数量：", len(frequent_itemsets))
    print("\n支持度最高的前10个频繁项集：")
    print(frequent_itemsets.head(10))
    return frequent_itemsets


def task4_generate_rules(frequent_itemsets: pd.DataFrame, min_threshold: float = 0.5) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("任务4：生成关联规则")
    print("=" * 60)

    if frequent_itemsets.empty or len(frequent_itemsets) < 2:
        print("频繁项集不足（至少需要2个频繁项集），无法生成关联规则。")
        return pd.DataFrame()

    log_progress(f"开始生成关联规则，min_confidence={min_threshold}")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
    log_progress("关联规则生成完成")
    rules = rules[rules["lift"] > 1].copy()

    if not rules.empty:
        rules["antecedents_str"] = rules["antecedents"].apply(lambda x: ",".join(list(x)))
        rules["consequents_str"] = rules["consequents"].apply(lambda x: ",".join(list(x)))
        rules = rules.sort_values(by=["lift", "confidence"], ascending=False).reset_index(drop=True)

    print("最小置信度阈值：", min_threshold)
    print("有效规则数量（lift > 1）：", len(rules))
    if not rules.empty:
        print("\n提升度最高的前10条规则：")
        print(rules[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(10))
    return rules


def task5_explain_and_visualize_rules(rules: pd.DataFrame, top_n: int = 5, show_plots: bool = True) -> None:
    print("\n" + "=" * 60)
    print("任务5：规则解读与可视化")
    print("=" * 60)

    if rules.empty:
        print("无关联规则可供解读。")
        return None

    selected = rules.head(top_n).copy()
    print("\n【选取的TOP规则解读】")
    for idx, row in selected.iterrows():
        print(
            f"规则{idx + 1}: 购买「{row['antecedents_str']}」的客户也倾向于购买「{row['consequents_str']}」；"
            f"支持度={row['support']:.4f}；置信度={row['confidence']:.4f}；提升度={row['lift']:.4f}"
        )

    print("\n【指标含义】")
    print("  支持度(Support)：规则中物品组合在全部交易中出现的比例。")
    print("  置信度(Confidence)：购买前件A后再购买后件B的条件概率。")
    print("  提升度(Lift)：置信度与后件独立出现概率之比，Lift > 1 表示正相关。")

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        rules["support"],
        rules["confidence"],
        c=rules["lift"],
        cmap="viridis",
        alpha=0.8,
        edgecolors="k",
    )
    plt.colorbar(scatter, label="提升度 Lift")
    plt.xlabel("支持度 Support")
    plt.ylabel("置信度 Confidence")
    plt.title("关联规则散点图（支持度 vs 置信度，颜色映射提升度）")
    plt.tight_layout()
    save_current_figure("01_association_rules_scatter.png")
    finish_plot(show_plots)
    return None


def task6_build_rfm(cleaned: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp]:
    print("\n" + "=" * 60)
    print("任务6：构建RFM特征")
    print("=" * 60)

    if cleaned.empty:
        print("数据为空，无法构建RFM特征。")
        return pd.DataFrame(), pd.Timestamp.now()

    log_progress("开始构建RFM特征")
    work = cleaned.copy()
    work["Amount"] = work["Quantity"] * work["UnitPrice"]
    ref_date = work["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        work.groupby("CustomerID")
        .agg(
            Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
            Frequency=("InvoiceNo", "nunique"),
            Monetary=("Amount", "sum"),
        )
        .reset_index()
    )

    log_progress("RFM特征构建完成")
    print("参考日期：", ref_date.strftime("%Y-%m-%d"))
    print("客户总数：", len(rfm))
    print("\n【RFM特征前10行】")
    print(rfm.head(10))
    print("\n【RFM统计摘要】")
    print(rfm[["Recency", "Frequency", "Monetary"]].describe())
    return rfm, ref_date


def task7_preprocess_rfm(
    rfm: pd.DataFrame,
    use_gpu: bool = False,
    gpu_status: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, StandardScaler, PCA]:
    print("\n" + "=" * 60)
    print("任务7：数据预处理")
    print("=" * 60)

    if rfm.empty:
        print("RFM数据为空，无法进行预处理。")
        return pd.DataFrame(), np.array([]), np.array([]), StandardScaler(), PCA(n_components=2)

    rfm_features = rfm[["Recency", "Frequency", "Monetary"]].copy()
    log_progress("开始执行RFM对数变换、标准化与PCA")
    print("原始RFM数据偏度（变换前）：")
    print(rfm_features.skew().to_string())

    if use_gpu and gpu_status and gpu_status["cuda_available"]:
        device = torch.device(gpu_status["device"])
        x = torch.tensor(rfm_features.values, dtype=torch.float32, device=device)
        rfm_log_tensor = torch.log1p(x)
        mean = rfm_log_tensor.mean(dim=0, keepdim=True)
        std = rfm_log_tensor.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-8)
        scaled_tensor = (rfm_log_tensor - mean) / std

        rfm_log = pd.DataFrame(rfm_log_tensor.cpu().numpy(), columns=rfm_features.columns, index=rfm_features.index)
        scaled = scaled_tensor.cpu().numpy()
        print(f"\n已使用GPU进行对数变换和标准化：{gpu_status['device_name']}")
    else:
        rfm_log = np.log1p(rfm_features)
        scaler_tmp = StandardScaler()
        scaled = scaler_tmp.fit_transform(rfm_log)
        print("\n未使用GPU，任务7采用CPU计算。")

    print("\n对数变换后RFM数据偏度（变换后）：")
    print(rfm_log.skew().to_string())

    scaler = StandardScaler()
    scaler.fit(rfm_log)
    print("\n标准化后数据均值（应≈0）：", scaled.mean(axis=0).round(6))
    print("标准化后数据标准差（应≈1）：", scaled.std(axis=0).round(6))

    pca = PCA(n_components=2, random_state=42)
    pca_data = pca.fit_transform(scaled)
    log_progress("RFM预处理完成")
    print("\nPCA各主成分解释的方差比例：", pca.explained_variance_ratio_)
    print("累计解释方差比例：", pca.explained_variance_ratio_.sum())
    print("PCA降维后数据维度：", pca_data.shape)
    return rfm_log, scaled, pca_data, scaler, pca


def task8_select_optimal_k(
    scaled_data: np.ndarray,
    min_k: int = 2,
    max_k: int = 10,
    use_gpu: bool = False,
    gpu_status: Dict[str, Any] = None,
    show_plots: bool = True,
) -> Tuple[pd.DataFrame, int]:
    print("\n" + "=" * 60)
    print("任务8：确定最佳聚类数")
    print("=" * 60)

    if scaled_data.size == 0:
        print("数据为空，无法评估K值。")
        return pd.DataFrame(), 2

    n_samples = scaled_data.shape[0]
    use_max_k = max(min_k, min(max_k, n_samples - 1))
    k_list, silhouette_list, inertia_list = [], [], []
    using_gpu_kmeans = use_gpu and gpu_status and gpu_status["cuda_available"]

    for k in range(min_k, use_max_k + 1):
        log_progress(f"正在评估K={k}")
        if using_gpu_kmeans:
            model = torch_kmeans_fit_predict(scaled_data, n_clusters=k, random_state=42)
            labels = model.labels_
            inertia = model.inertia_
        else:
            model = KMeans(n_clusters=k, random_state=42, n_init=20)
            labels = model.fit_predict(scaled_data)
            inertia = model.inertia_

        sil = silhouette_score(scaled_data, labels) if len(set(labels)) > 1 else np.nan
        k_list.append(k)
        silhouette_list.append(sil)
        inertia_list.append(inertia)

    metrics_df = pd.DataFrame({"k": k_list, "silhouette": silhouette_list, "inertia": inertia_list})
    best_row = metrics_df.sort_values(by="silhouette", ascending=False).iloc[0]
    best_k = int(best_row["k"])

    print("各K值的评估指标：")
    print(metrics_df.to_string(index=False))
    print(f"\n建议最佳K值（轮廓系数最大={best_row['silhouette']:.4f}）：K = {best_k}")
    print(f"K-Means评估阶段使用设备：{'GPU' if using_gpu_kmeans else 'CPU'}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(metrics_df["k"], metrics_df["inertia"], marker="o")
    axes[0].set_xlabel("K值（聚类数）")
    axes[0].set_ylabel("簇内误差平方和 Inertia（SSE）")
    axes[0].set_title("肘部图：K值与簇内误差平方和")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(metrics_df["k"], metrics_df["silhouette"], marker="o", color="orange")
    axes[1].set_xlabel("K值（聚类数）")
    axes[1].set_ylabel("轮廓系数 Silhouette Score")
    axes[1].set_title("轮廓系数曲线：K值与聚类质量")
    axes[1].grid(True, alpha=0.3)
    axes[1].annotate(
        f"最佳K={best_k}",
        xy=(best_k, best_row["silhouette"]),
        xytext=(best_k + 0.5, best_row["silhouette"] - 0.02),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=11,
        color="red",
    )
    plt.tight_layout()
    save_current_figure("02_k_selection.png")
    finish_plot(show_plots)
    return metrics_df, best_k


def task9_kmeans_clustering(
    scaled_data: np.ndarray,
    rfm: pd.DataFrame,
    best_k: int,
    use_gpu: bool = False,
    gpu_status: Dict[str, Any] = None,
) -> Tuple[pd.DataFrame, Any, np.ndarray]:
    print("\n" + "=" * 60)
    print("任务9：执行K-Means聚类")
    print("=" * 60)

    if scaled_data.size == 0 or rfm.empty:
        print("数据为空，无法执行K-Means聚类。")
        return pd.DataFrame(), KMeans(n_clusters=2, random_state=42), np.array([])

    using_gpu_kmeans = use_gpu and gpu_status and gpu_status["cuda_available"]
    log_progress(f"开始执行K-Means聚类，K={best_k}")
    if using_gpu_kmeans:
        model = torch_kmeans_fit_predict(scaled_data, n_clusters=best_k, random_state=42)
        labels = model.labels_
        device_name = gpu_status["device_name"]
    else:
        model = KMeans(n_clusters=best_k, random_state=42, n_init=20)
        labels = model.fit_predict(scaled_data)
        device_name = "CPU"

    result = rfm.copy()
    result["KMeansCluster"] = labels
    log_progress("K-Means聚类完成")

    print(f"使用K = {best_k} 进行K-Means聚类")
    print(f"K-Means执行设备：{device_name}")
    print(f"聚类中心坐标（标准化空间）：共{model.cluster_centers_.shape[0]}个簇，每簇{model.cluster_centers_.shape[1]}维")
    print("\n各簇样本数量分布：")
    cluster_counts = pd.Series(labels).value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        print(f"  簇{cluster_id}：{count}位客户")

    print("\n【带聚类标签的客户数据（前10行）】")
    print(result.head(10))
    return result, model, labels


def task10_visualize_and_interpret(
    pca_data: np.ndarray,
    kmeans_result: pd.DataFrame,
    selected_rules: pd.DataFrame,
    optimal_k: int,
    show_plots: bool = True,
) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("任务10：结果可视化与业务解读")
    print("=" * 60)

    if pca_data.size == 0 or kmeans_result.empty:
        print("数据为空，无法进行可视化与业务解读。")
        return pd.DataFrame()

    log_progress("开始绘制K-Means结果并生成业务解读")
    labels = kmeans_result["KMeansCluster"].values

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        pca_data[:, 0],
        pca_data[:, 1],
        c=labels,
        cmap="tab10",
        alpha=0.8,
        edgecolors="k",
    )
    plt.colorbar(scatter, label="簇标签 Cluster")
    plt.xlabel("PCA 第一主成分")
    plt.ylabel("PCA 第二主成分")
    plt.title(f"K-Means聚类结果（PCA降维至2维，K={optimal_k}）")
    plt.tight_layout()
    save_current_figure("03_kmeans_pca_clusters.png")
    finish_plot(show_plots)

    profile = (
        kmeans_result.groupby("KMeansCluster")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
        .sort_values(by="Monetary", ascending=False)
    )

    print("\n【各簇RFM特征均值（原始尺度）】")
    print("（Recency越小表示最近购买过；Frequency越大表示购买频繁；Monetary越大表示消费高）")
    print(profile.to_string(index=False))

    print("\n【营销建议（结合聚类特征与关联规则）】")
    median_monetary = profile["Monetary"].median()
    median_frequency = profile["Frequency"].median()
    median_recency = profile["Recency"].median()

    for _, row in profile.iterrows():
        cid = int(row["KMeansCluster"])
        if row["Monetary"] >= median_monetary and row["Frequency"] >= median_frequency:
            print(f"  簇{cid}：[高价值] 高价值高频客户，建议做会员权益与高端组合推荐。")
        elif row["Recency"] > median_recency:
            print(f"  簇{cid}：[沉睡] 沉睡风险客户，建议做召回券与限时折扣。")
        else:
            print(f"  簇{cid}：[潜力] 潜力成长客户，建议用满减与加价购提升客单。")

    if not selected_rules.empty:
        sample_rule = selected_rules.iloc[0]
        print("\n【联动陈列建议】")
        print(
            f"  基于关联规则「{sample_rule['antecedents_str']}」→「{sample_rule['consequents_str']}」"
            f"（置信度={sample_rule['confidence']:.2f}, 提升度={sample_rule['lift']:.2f}）"
        )
    return profile


def task11_hierarchical_clustering(
    scaled_data: np.ndarray,
    target_k: int = 3,
    method: str = "ward",
    show_plots: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    print("\n" + "=" * 60)
    print("任务11：层次聚类与树状图")
    print("=" * 60)

    if scaled_data.size == 0:
        print("数据为空，无法进行层次聚类。")
        return np.array([]), np.array([])

    print(f"链接方法：{method}，目标簇数：{target_k}")
    log_progress("开始执行层次聚类 linkage 计算")
    z_matrix = linkage(scaled_data, method=method)
    log_progress("层次聚类 linkage 计算完成")
    labels = fcluster(z_matrix, t=target_k, criterion="maxclust") - 1

    cluster_distribution = pd.Series(labels).value_counts().sort_index()
    print("\n层次聚类各簇样本数量：")
    for cluster_id, count in cluster_distribution.items():
        print(f"  簇{cluster_id}：{count}位客户")

    plt.figure(figsize=(12, 5))
    dendrogram(
        z_matrix,
        truncate_mode="lastp",
        p=20,
        leaf_rotation=45,
        leaf_font_size=10,
        show_contracted=True,
    )
    if target_k > 1 and len(z_matrix) >= target_k - 1:
        cut_idx = max(1, target_k - 1)
        cut_height = z_matrix[-cut_idx, 2]
        plt.axhline(y=cut_height, color="red", linestyle="--", alpha=0.7, label=f"切割阈值（K={target_k}）")
        plt.legend()
    plt.title(f"层次聚类树状图（链接方法：{method}，截断显示）")
    plt.xlabel("样本簇（截断后）")
    plt.ylabel("合并距离")
    plt.tight_layout()
    save_current_figure("04_hierarchical_dendrogram.png")
    finish_plot(show_plots)
    return labels, z_matrix


def task12_compare_cluster_results(
    kmeans_labels: np.ndarray,
    hierarchical_labels: np.ndarray,
    scaled_data: np.ndarray,
) -> Dict[str, float]:
    print("\n" + "=" * 60)
    print("任务12：聚类结果对比")
    print("=" * 60)

    result = {"ARI": np.nan, "NMI": np.nan, "Silhouette_KMeans": np.nan, "Silhouette_Hierarchical": np.nan}
    if len(kmeans_labels) == 0 or len(hierarchical_labels) == 0:
        print("聚类标签为空，无法进行比较。")
        return result

    result["ARI"] = adjusted_rand_score(kmeans_labels, hierarchical_labels)
    result["NMI"] = normalized_mutual_info_score(kmeans_labels, hierarchical_labels)
    log_progress("两种聚类结果的一致性指标计算完成")

    print(f"调整兰德指数（ARI）：{result['ARI']:.4f}")
    print(f"归一化互信息（NMI）：{result['NMI']:.4f}")
    print("\n【算法对比分析】")
    print("  K-Means：速度快，适合大规模、近似球状簇。")
    print("  层次聚类：解释性强，能展示层次结构，但复杂度较高。")
    return result


def task13_evaluate_clusters(
    kmeans_labels: np.ndarray,
    hierarchical_labels: np.ndarray,
    scaled_data: np.ndarray,
    comparison_result: Dict[str, float],
) -> None:
    print("\n" + "=" * 60)
    print("任务13：聚类评估（内部指标）")
    print("=" * 60)

    if len(kmeans_labels) == 0 or len(hierarchical_labels) == 0:
        print("聚类标签为空，无法进行评估。")
        return None

    sil_kmeans = silhouette_score(scaled_data, kmeans_labels) if len(set(kmeans_labels)) > 1 else np.nan
    sil_hier = silhouette_score(scaled_data, hierarchical_labels) if len(set(hierarchical_labels)) > 1 else np.nan
    log_progress("聚类内部评估指标计算完成")

    comparison_result["Silhouette_KMeans"] = sil_kmeans
    comparison_result["Silhouette_Hierarchical"] = sil_hier

    print(f"\nK-Means轮廓系数：        {sil_kmeans:.4f}")
    print(f"层次聚类轮廓系数：       {sil_hier:.4f}")
    if not np.isnan(sil_kmeans) and not np.isnan(sil_hier):
        diff = sil_kmeans - sil_hier
        if diff > 0.05:
            print(f"\n结论：K-Means聚类质量更优（轮廓系数高出{diff:.4f}）")
        elif diff < -0.05:
            print(f"\n结论：层次聚类质量更优（轮廓系数高出{-diff:.4f}）")
        else:
            print(f"\n结论：两种方法聚类质量接近（差异={diff:.4f}）")
    return None


def run_analysis(
    sample_n: int = SAMPLE_N,
    data_path: str = DATA_PATH,
    use_gpu: bool = USE_GPU_IF_AVAILABLE,
    show_plots: bool = SHOW_PLOTS,
) -> Dict[str, Any]:
    configure_chinese_plot()
    ensure_output_dir()

    log_progress("分析流程启动")
    print("mlxtend已成功导入")
    print(f"当前采样行数 SAMPLE_N = {sample_n}")
    print(f"当前参数：MIN_SUPPORT={MIN_SUPPORT}, MIN_CONFIDENCE={MIN_CONFIDENCE}, MAX_ITEMSET_LEN={MAX_ITEMSET_LEN}")
    gpu_status = get_gpu_status()
    if use_gpu and gpu_status["cuda_available"]:
        print(f"检测到可用GPU：{gpu_status['device_name']}")
        print("将使用GPU加速任务7中的对数/标准化，以及任务8/9中的K-Means。")
        print("说明：Apriori、关联规则、层次聚类仍主要依赖CPU库。")
    else:
        print("当前将使用CPU执行。若需GPU，请确保Torch CUDA可用。")

    log_progress("进入任务1：数据加载与清洗")
    cleaned_data = task1_load_and_clean(data_path, sample_n=sample_n)
    log_progress("进入任务2：购物篮事务构造")
    transactions, one_hot_data = task2_build_transactions(cleaned_data, item_col="StockCode")
    log_progress("进入任务3：频繁项集挖掘")
    frequent_itemsets = task3_mine_frequent_itemsets(
        one_hot_data,
        min_support=MIN_SUPPORT,
        max_len=MAX_ITEMSET_LEN,
    )
    log_progress("进入任务4：关联规则生成")
    rules = task4_generate_rules(frequent_itemsets, min_threshold=MIN_CONFIDENCE)
    log_progress("进入任务5：规则解读与可视化")
    task5_explain_and_visualize_rules(rules, top_n=5, show_plots=show_plots)

    log_progress("进入任务6：RFM特征构建")
    rfm_features, ref_date = task6_build_rfm(cleaned_data)
    log_progress("进入任务7：RFM预处理")
    rfm_log, scaled_data, pca_data, scaler, pca_model = task7_preprocess_rfm(
        rfm_features,
        use_gpu=use_gpu,
        gpu_status=gpu_status,
    )

    log_progress("进入任务8：选择最佳K")
    k_metrics_df, optimal_k = task8_select_optimal_k(
        scaled_data,
        min_k=2,
        max_k=10,
        use_gpu=use_gpu,
        gpu_status=gpu_status,
        show_plots=show_plots,
    )
    log_progress("进入任务9：K-Means聚类")
    kmeans_result, kmeans_model, kmeans_labels = task9_kmeans_clustering(
        scaled_data,
        rfm_features,
        optimal_k,
        use_gpu=use_gpu,
        gpu_status=gpu_status,
    )
    log_progress("进入任务10：聚类可视化与业务解读")
    profile = task10_visualize_and_interpret(
        pca_data,
        kmeans_result,
        rules,
        optimal_k=optimal_k,
        show_plots=show_plots,
    )
    log_progress("进入任务11：层次聚类")
    hierarchical_labels, z_matrix = task11_hierarchical_clustering(
        scaled_data,
        target_k=optimal_k,
        method="ward",
        show_plots=show_plots,
    )
    log_progress("进入任务12：聚类结果对比")
    comparison_result = task12_compare_cluster_results(kmeans_labels, hierarchical_labels, scaled_data)
    log_progress("进入任务13：聚类评估")
    task13_evaluate_clusters(kmeans_labels, hierarchical_labels, scaled_data, comparison_result)

    log_progress("全部任务执行完毕")
    print("\n" + "=" * 60)
    print("全部分析任务完成！")
    print("=" * 60)

    return {
        "cleaned_data": cleaned_data,
        "transactions": transactions,
        "one_hot_data": one_hot_data,
        "frequent_itemsets": frequent_itemsets,
        "rules": rules,
        "rfm_features": rfm_features,
        "rfm_log": rfm_log,
        "scaled_data": scaled_data,
        "pca_data": pca_data,
        "k_metrics_df": k_metrics_df,
        "optimal_k": optimal_k,
        "kmeans_result": kmeans_result,
        "kmeans_model": kmeans_model,
        "hierarchical_labels": hierarchical_labels,
        "z_matrix": z_matrix,
        "comparison_result": comparison_result,
        "profile": profile,
        "ref_date": ref_date,
        "gpu_status": gpu_status,
        "scaler": scaler,
        "pca_model": pca_model,
    }


def main() -> None:
    run_analysis(sample_n=SAMPLE_N, data_path=DATA_PATH, use_gpu=USE_GPU_IF_AVAILABLE, show_plots=SHOW_PLOTS)
    return None


if __name__ == "__main__":
    main()
