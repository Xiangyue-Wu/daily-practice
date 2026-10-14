"""中文文本相似度实验：TF-IDF、余弦相似度与热力图。"""

import csv
import json
from pathlib import Path

import jieba
import matplotlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

matplotlib.use("Agg")  # 直接保存图片，无须弹出窗口
import matplotlib.pyplot as plt


# 四条实验测试文本；也可通过项目根目录的 data/samples.json 配置
DEFAULT_CORPUS = [
    "人工智能是计算机科学的一个分支，它企图了解智能的实质。",
    "机器学习是人工智能的核心，是使计算机具有智能的根本途径。",
    "篮球是一项全球广泛开展的运动，它强调团队协作与竞技对抗。",
    "现代教育更加注重培养学生的综合能力，包括思维创新、沟通协作与社会责任感。",
]


def load_corpus(data_dir):
    """优先读取外部语料，缺省时使用原实验语料。"""
    path = Path(data_dir) / "samples.json"
    if not path.exists():
        return DEFAULT_CORPUS.copy()

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    # samples.json 可包含其他实验的数据
    corpus = data.get("similarity_corpus", DEFAULT_CORPUS) if isinstance(data, dict) else data
    if not isinstance(corpus, list) or len(corpus) < 2:
        raise ValueError("similarity_corpus 至少需要两条文本。")
    if any(not isinstance(text, str) or not text.strip() for text in corpus):
        raise ValueError("语料必须为非空字符串列表。")
    return corpus


def calculate_similarity(corpus):
    """jieba 分词后，计算 TF-IDF 和余弦相似度。"""
    segmented = [" ".join(jieba.cut(text)) for text in corpus]
    vectorizer = TfidfVectorizer()  # 保持原实验的默认参数
    tfidf_matrix = vectorizer.fit_transform(segmented)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return segmented, tfidf_matrix, similarity_matrix


def save_results(corpus, segmented, matrix, output_dir):
    """保存分词结果、相似度矩阵和排行榜。"""
    output_dir = Path(output_dir)

    with (output_dir / "segmented_texts.txt").open("w", encoding="utf-8") as file:
        for i, text in enumerate(segmented):
            file.write(f"文本{i}: {text}\n")

    with (output_dir / "similarity_matrix.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["文本编号"] + [f"文本{i}" for i in range(len(corpus))])
        for i, row in enumerate(matrix):
            writer.writerow([f"文本{i}"] + [f"{value:.6f}" for value in row])

    with (output_dir / "similarity_ranking.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["源文本", "排名", "目标文本", "余弦相似度"])
        for i in range(len(corpus)):
            # 排除自身后，按相似度降序排列
            ranked = sorted(
                (j for j in range(len(corpus)) if j != i),
                key=lambda j: (-matrix[i, j], j),
            )
            for rank, j in enumerate(ranked, start=1):
                writer.writerow([i, rank, j, f"{matrix[i, j]:.6f}"])
            best = ranked[0]
            print(f"文本{i} 最相似的是文本{best}，相似度：{matrix[i, best]:.4f}")


def save_heatmap(matrix, output_dir):
    """绘制与报告对应的蓝色相似度热力图。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    fig.colorbar(image, ax=ax, label="相似度")

    labels = [f"文本{i}" for i in range(len(matrix))]
    ax.set_xticks(range(len(matrix)), labels=labels)
    ax.set_yticks(range(len(matrix)), labels=labels)
    ax.set_xlabel("文本编号")
    ax.set_ylabel("文本编号")
    ax.set_title("文本相似度热力图", fontweight="bold")

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            value = matrix[i, j]
            ax.text(
                j, i, f"{value:.2f}",
                ha="center", va="center",
                color="white" if value > 0.5 else "black",
            )

    fig.tight_layout()
    fig.savefig(Path(output_dir) / "text_similarity_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_experiment(data_dir, output_dir):
    """供 main.py 调用的统一入口。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(data_dir)
    segmented, tfidf_matrix, matrix = calculate_similarity(corpus)

    print("\n中文分词结果：")
    for i, text in enumerate(segmented):
        print(f"文本{i}: {text}")

    print(f"\nTF-IDF 矩阵形状：{tfidf_matrix.shape}")
    print("\n余弦相似度矩阵：")
    print(np.array2string(matrix, formatter={"float_kind": lambda x: f"{x:.2f}"}))
    print("\n最相似文本：")

    save_results(corpus, segmented, matrix, output_dir)
    save_heatmap(matrix, output_dir)
    print(f"\n结果已保存到：{output_dir}")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    run_experiment(project_dir / "data", project_dir / "results" / "similarity")
