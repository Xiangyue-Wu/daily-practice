"""编辑距离实验：动态规划、英文拼写纠错与结果可视化。"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 直接保存图片
import matplotlib.pyplot as plt


# 实验词典及拼写错误测试词
DEFAULT_VOCAB = [
    "apple", "banana", "orange", "python",
    "java", "intelligence", "language",
]
DEFAULT_INPUTS = ["aple", "pythoon", "langage", "intelegence"]


def load_data(data_dir):
    """优先读取自定义数据，否则使用原实验数据。"""
    path = Path(data_dir) / "samples.json"
    data = {}
    if path.exists():
        with path.open("r", encoding="utf-8-sig") as file:
            loaded = json.load(file)
        if isinstance(loaded, dict):
            data = loaded

    vocab = data.get("edit_distance_vocab", DEFAULT_VOCAB)
    inputs = data.get("edit_distance_inputs", DEFAULT_INPUTS)
    for label, items in [("词典", vocab), ("测试输入", inputs)]:
        if not isinstance(items, list) or not items:
            raise ValueError(f"{label}必须为非空字符串列表。")
        if any(not isinstance(word, str) or not word.strip() for word in items):
            raise ValueError(f"{label}包含空值或非字符串。")
    return vocab, inputs


def min_edit_distance(word1, word2):
    """使用动态规划计算 Levenshtein 编辑距离。"""
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if word1[i - 1] == word2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,          # 删除
                dp[i][j - 1] + 1,          # 插入
                dp[i - 1][j - 1] + cost,   # 替换或匹配
            )
    return dp[m][n]


def rank_candidates(user_word, vocab):
    """按编辑距离升序排列，距离相同时保留词典原顺序。"""
    results = [
        (word, min_edit_distance(user_word.lower(), word.lower()))
        for word in vocab
    ]
    return sorted(results, key=lambda item: item[1])


def save_csv(results, output_dir):
    """保存全部候选词的距离与排名。"""
    path = Path(output_dir) / "edit_distance_results.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["输入词", "排名", "候选词", "编辑距离", "是否最佳候选"])
        for user_word, candidates in results:
            best_distance = candidates[0][1]
            for rank, (word, distance) in enumerate(candidates, start=1):
                writer.writerow([
                    user_word, rank, word, distance,
                    "是" if distance == best_distance else "否",
                ])


def save_chart(results, output_dir):
    """生成四组编辑距离横向柱状图。"""
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei", "SimHei", "PingFang SC",
        "Noto Sans CJK SC", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    rows = (len(results) + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=(12, 4.5 * rows), squeeze=False)

    for index, (user_word, candidates) in enumerate(results):
        ax = axes[index // 2][index % 2]
        words = [word for word, _ in candidates]
        distances = [distance for _, distance in candidates]
        best_distance = min(distances)
        colors = ["#F16D6D" if d == best_distance else "#52C4BF" for d in distances]

        bars = ax.barh(words, distances, color=colors, edgecolor="black", alpha=0.9)
        ax.invert_yaxis()  # 最接近的候选排在最上面
        ax.set_title(f"用户输入：'{user_word}'")
        ax.set_xlabel("编辑距离")
        ax.set_xlim(0, max(distances) + 1.5)

        for bar, distance in zip(bars, distances):
            ax.text(
                bar.get_width() + 0.1,
                bar.get_y() + bar.get_height() / 2,
                str(distance), va="center", fontsize=9,
            )

    for index in range(len(results), rows * 2):
        fig.delaxes(axes[index // 2][index % 2])

    fig.tight_layout()
    fig.savefig(
        Path(output_dir) / "edit_distance_barcharts.png",
        dpi=300, bbox_inches="tight",
    )
    plt.close(fig)


def run_experiment(data_dir, output_dir):
    """供 main.py 调用的统一入口。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vocab, inputs = load_data(data_dir)
    results = []

    for user_word in inputs:
        candidates = rank_candidates(user_word, vocab)
        results.append((user_word, candidates))
        best_distance = candidates[0][1]
        best_words = [word for word, d in candidates if d == best_distance]
        top3 = "、".join(f"{word}({d})" for word, d in candidates[:3])
        print(f"输入：{user_word} -> 最佳纠错：{', '.join(best_words)}（编辑距离：{best_distance}）")
        print(f"候选 Top3：{top3}")

    save_csv(results, output_dir)
    save_chart(results, output_dir)
    print(f"\n实验结果已保存到：{output_dir}")


if __name__ == "__main__":
    project_dir = Path(__file__).resolve().parent.parent
    run_experiment(project_dir / "data", project_dir / "results" / "edit_distance")
