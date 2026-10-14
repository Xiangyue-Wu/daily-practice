"""关键词提取实验：TF-IDF 与 TextRank。"""

import csv
import json
from pathlib import Path

import jieba.analyse
import matplotlib

matplotlib.use("Agg")  # 保存图表，不弹出窗口
import matplotlib.pyplot as plt


# 实验使用的自然语言处理主题文本
DEFAULT_TEXT = (
    "自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。"
    "它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。"
    "自然语言处理是一门融语言学、计算机科学、数学于一体的科学。"
    "因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，"
    "所以它与语言学的研究有着密切的联系。"
)

# 两种方法分别使用的自定义停用词
DEFAULT_STOPWORDS = {
    "tfidf": {"的", "研究", "因此", "处理", "了", "领域"},
    "textrank": {"研究", "的", "了", "因此", "人们"},
}


def load_text(data_dir):
    """优先读取自定义实验文本，否则使用报告原文。"""
    path = Path(data_dir) / "samples.json"
    if not path.exists():
        return DEFAULT_TEXT

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        return DEFAULT_TEXT
    text = data.get("keyword_text", DEFAULT_TEXT)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("samples.json 中的 keyword_text 必须为非空字符串。")
    return text


def load_stopwords(data_dir, method):
    """支持使用 data 目录下的自定义停用词表。"""
    path = Path(data_dir) / f"stopwords_{method}.txt"
    if not path.exists():
        return DEFAULT_STOPWORDS[method].copy()

    return {
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def extract_keywords(text, data_dir):
    """分别计算未设置停用词与设置停用词的结果。"""
    tfidf = jieba.analyse.TFIDF()
    textrank = jieba.analyse.TextRank()

    # 第一组：不额外设置自定义停用词，提取前 5 个关键词
    tfidf_before = tfidf.extract_tags(text, topK=5, withWeight=True)
    textrank_before = textrank.textrank(text, topK=5, withWeight=True)

    # 第二组：增加实验停用词，提取前 4 个关键词
    tfidf.stop_words.update(load_stopwords(data_dir, "tfidf"))
    textrank.stop_words.update(load_stopwords(data_dir, "textrank"))
    tfidf_after = tfidf.extract_tags(text, topK=4, withWeight=True)
    textrank_after = textrank.textrank(text, topK=4, withWeight=True)

    return {
        "tfidf_before": tfidf_before,
        "tfidf_after": tfidf_after,
        "textrank_before": textrank_before,
        "textrank_after": textrank_after,
    }


def save_csv(results, output_dir):
    """将全部实验结果写入 CSV。"""
    path = Path(output_dir) / "keyword_results.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["方法", "实验设置", "排名", "关键词", "权重"])

        for name, keywords in results.items():
            method, stage = name.split("_")
            method_label = "TF-IDF" if method == "tfidf" else "TextRank"
            setting = "未设置自定义停用词 / Top5" if stage == "before" else "指定停用词 / Top4"
            for rank, (word, weight) in enumerate(keywords, start=1):
                writer.writerow([method_label, setting, rank, word, f"{weight:.6f}"])


def save_chart(name, keywords, output_dir):
    """用柱状图展示关键词权重。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"
    ]
    plt.rcParams["axes.unicode_minus"] = False

    method, stage = name.split("_")
    title = "TF-IDF" if method == "tfidf" else "TextRank"
    condition = (
        "未设置自定义停用词，TopK=5"
        if stage == "before"
        else "设置自定义停用词，TopK=4"
    )
    color = "#4CC9C0" if method == "tfidf" else "#FF6B6B"

    words = [word for word, _ in keywords]
    scores = [score for _, score in keywords]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(words, scores, color=color, edgecolor="black", alpha=0.9)
    ax.set_title(f"{title}关键词权重（{condition}）")
    ax.set_ylabel("权重")
    ax.set_ylim(0, max(scores, default=1) * 1.2)

    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{score:.3f}",
            ha="center", va="bottom", fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(
        Path(output_dir) / f"{name}.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)


def run_experiment(data_dir, output_dir):
    """供 main.py 调用的统一入口。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text = load_text(data_dir)
    results = extract_keywords(text, data_dir)

    for name, keywords in results.items():
        method, stage = name.split("_")
        setting = "未设置自定义停用词 / Top5" if stage == "before" else "指定停用词 / Top4"
        print(f"\n{method.upper()}（{setting}）：")
        for word, weight in keywords:
            print(f"  {word:<10} {weight:.4f}")
        save_chart(name, keywords, output_dir)

    save_csv(results, output_dir)
    print(f"\n实验结果已保存到：{output_dir}")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    run_experiment(project_dir / "data", project_dir / "results" / "keywords")
