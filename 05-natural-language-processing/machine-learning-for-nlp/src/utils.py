
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, roc_curve
)
from models import build_logistic_regression, TextCNN, TextLSTM


# ==================== 模型训练 ====================

def train_model(model, loader, device, epochs, lr):
    """训练深度学习模型"""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        losses.append(avg_loss)

        print(
            f"  Epoch {epoch + 1}/{epochs} "
            f"| Loss: {avg_loss:.4f}"
        )

    return losses


# ==================== 模型预测 ====================

def predict_deep(model, loader, device):
    """返回真实标签、预测标签及正类概率"""
    model.eval()
    y_true, y_pred, y_score = [], [], []

    with torch.no_grad():
        for features, labels in loader:
            logits = model(features.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())
            y_score.extend(probs.cpu().numpy().tolist())

    return (
        np.array(y_true),
        np.array(y_pred),
        np.array(y_score)
    )


# ==================== 模型评估 ====================

def get_metrics(y_true, y_pred, y_score):
    """计算分类评估指标"""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, y_score))
    }


# ==================== 实验结果可视化 ====================

def plot_comparison(
    metrics, histories, matrices, scores,
    y_true, df, fig_dir
):
    """生成实验图表"""
    names = list(metrics)

    # 1. 损失曲线、准确率和混淆矩阵
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for name, values in histories.items():
        axes[0].plot(
            range(1, len(values) + 1),
            values,
            marker="o",
            label=name
        )

    axes[0].set(
        title="Training Loss",
        xlabel="Epoch",
        ylabel="Loss"
    )
    axes[0].legend()

    accuracies = [
        metrics[name]["accuracy"] for name in names
    ]

    axes[1].bar(names, accuracies)
    axes[1].set(
        title="Accuracy Comparison",
        ylim=(0, 1.05),
        ylabel="Accuracy"
    )
    axes[1].tick_params(axis="x", labelrotation=15)

    for i, value in enumerate(accuracies):
        axes[1].text(
            i, value + 0.015,
            f"{value:.4f}",
            ha="center"
        )

    matrix = matrices["DL_BiLSTM"]
    axes[2].imshow(matrix, cmap="Blues")
    axes[2].set(
        title="Confusion Matrix (Bi-LSTM)",
        xlabel="Predicted",
        ylabel="True"
    )
    axes[2].set_xticks(
        [0, 1], ["Negative", "Positive"]
    )
    axes[2].set_yticks(
        [0, 1], ["Negative", "Positive"]
    )

    for row in range(2):
        for col in range(2):
            axes[2].text(
                col, row, str(matrix[row, col]),
                ha="center", va="center"
            )

    fig.tight_layout()
    fig.savefig(
        fig_dir / "model_comparison.png",
        dpi=180
    )
    plt.close(fig)

    # 同时保存三张独立图片，便于分别用于实验报告
    fig, ax = plt.subplots(figsize=(7, 5))

    for name, values in histories.items():
        ax.plot(
            range(1, len(values) + 1),
            values,
            marker="o",
            label=name
        )

    ax.set(
        title="Training Loss",
        xlabel="Epoch",
        ylabel="Loss"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "training_loss.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(names, accuracies)
    ax.set(
        title="Accuracy Comparison",
        ylim=(0, 1.05),
        ylabel="Accuracy"
    )
    ax.tick_params(axis="x", labelrotation=15)

    for i, value in enumerate(accuracies):
        ax.text(i, value + 0.015, f"{value:.4f}", ha="center")

    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(matrix, cmap="Blues")
    ax.set(
        title="Confusion Matrix (Bi-LSTM)",
        xlabel="Predicted",
        ylabel="True"
    )
    ax.set_xticks([0, 1], ["Negative", "Positive"])
    ax.set_yticks([0, 1], ["Negative", "Positive"])

    for row in range(2):
        for col in range(2):
            ax.text(
                col, row, str(matrix[row, col]),
                ha="center", va="center"
            )

    fig.tight_layout()
    fig.savefig(
        fig_dir / "bilstm_confusion_matrix.png",
        dpi=180
    )
    plt.close(fig)

    # 2. ROC 曲线
    fig, ax = plt.subplots(figsize=(7, 6))

    for name in names:
        fpr, tpr, _ = roc_curve(y_true, scores[name])
        auc = metrics[name]["roc_auc"]

        ax.plot(
            fpr, tpr,
            label=f"{name} (AUC={auc:.3f})"
        )

    ax.plot(
        [0, 1], [0, 1],
        "--", color="gray"
    )
    ax.set(
        title="ROC Curves",
        xlabel="False Positive Rate",
        ylabel="True Positive Rate"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        fig_dir / "roc_curves.png",
        dpi=180
    )
    plt.close(fig)

    # 3. 文本长度分布
    lengths = df["cut_text"].map(
        lambda text: len(text.split())
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        lengths,
        bins=50,
        range=(0, min(max(int(lengths.max()), 100), 500))
    )
    ax.axvline(
        100,
        linestyle="--",
        color="red",
        label="Max length = 100"
    )
    ax.set(
        title="Text Length Distribution",
        xlabel="Tokens",
        ylabel="Count"
    )
    ax.legend()

    fig.tight_layout()
    fig.savefig(
        fig_dir / "text_length_distribution.png",
        dpi=180
    )
    plt.close(fig)

    # 4. 正负面中文词云
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("未安装 wordcloud，跳过词云生成")
        return

    # 自动查找常见中文字体
    font_candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    ]

    font_path = next(
        (p for p in font_candidates if p.is_file()),
        None
    )

    if font_path is None:
        print("未找到中文字体，跳过词云生成")
        return

    for label, filename in [
        (1, "positive_wordcloud.png"),
        (0, "negative_wordcloud.png")
    ]:
        corpus = " ".join(
            df.loc[df["label"] == label, "cut_text"]
        )

        if not corpus.strip():
            continue

        cloud = WordCloud(
            font_path=str(font_path),
            width=900,
            height=550,
            background_color="white",
            collocations=False
        ).generate(corpus)

        cloud.to_file(str(fig_dir / filename))


# ==================== 完整实验流程 ====================

def run_experiments(data):
    """供 main.py 调用的完整实验入口"""

    output_dir = Path(data["output_dir"])
    model_dir = output_dir / "models"
    fig_dir = output_dir / "figures"

    model_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    device = data["device"]
    y_true = np.asarray(data["y_test"])

    metrics = {}
    matrices = {}
    scores = {}
    histories = {}

    # ---------- 1. 逻辑回归 ----------

    print("\n[1/3] 训练 Logistic Regression")

    lr_model = build_logistic_regression()

    lr_model.fit(
        data["X_train_tfidf"],
        data["y_train"]
    )

    pred = lr_model.predict(data["X_test_tfidf"])
    prob = lr_model.predict_proba(
        data["X_test_tfidf"]
    )[:, 1]

    metrics["ML_LogReg"] = get_metrics(
        y_true, pred, prob
    )

    matrices["ML_LogReg"] = confusion_matrix(
        y_true, pred, labels=[0, 1]
    )

    scores["ML_LogReg"] = prob

    joblib.dump(
        lr_model,
        model_dir / "logistic_regression.joblib"
    )
    joblib.dump(
        data["vectorizer"],
        model_dir / "tfidf_vectorizer.joblib"
    )

    # ---------- 2. TextCNN ----------

    print("\n[2/3] 训练 TextCNN")

    cnn = TextCNN(
        data["vocab_size"],
        data["embed_dim"]
    ).to(device)

    histories["TextCNN"] = train_model(
        cnn,
        data["train_loader"],
        device,
        data["epochs"],
        data["learning_rate"]
    )

    _, pred, prob = predict_deep(
        cnn, data["test_loader"], device
    )

    metrics["DL_TextCNN"] = get_metrics(
        y_true, pred, prob
    )

    matrices["DL_TextCNN"] = confusion_matrix(
        y_true, pred, labels=[0, 1]
    )

    scores["DL_TextCNN"] = prob

    torch.save(
        cnn.state_dict(),
        model_dir / "textcnn.pt"
    )

    # ---------- 3. Bi-LSTM ----------

    print("\n[3/3] 训练 Bi-LSTM")

    lstm = TextLSTM(
        data["vocab_size"],
        data["embed_dim"],
        hidden_dim=64
    ).to(device)

    histories["Bi-LSTM"] = train_model(
        lstm,
        data["train_loader"],
        device,
        data["epochs"],
        data["learning_rate"]
    )

    _, pred, prob = predict_deep(
        lstm, data["test_loader"], device
    )

    metrics["DL_BiLSTM"] = get_metrics(
        y_true, pred, prob
    )

    matrices["DL_BiLSTM"] = confusion_matrix(
        y_true, pred, labels=[0, 1]
    )

    scores["DL_BiLSTM"] = prob

    torch.save(
        lstm.state_dict(),
        model_dir / "bilstm.pt"
    )

    # ---------- 4. 保存评估结果 ----------

    pd.DataFrame.from_dict(
        metrics,
        orient="index"
    ).rename_axis("model").to_csv(
        output_dir / "results.csv",
        encoding="utf-8-sig"
    )

    pd.DataFrame({
        "epoch": range(1, data["epochs"] + 1),
        **histories
    }).to_csv(
        output_dir / "loss_history.csv",
        index=False
    )

    for name, matrix in matrices.items():
        pd.DataFrame(
            matrix,
            index=["true_0", "true_1"],
            columns=["pred_0", "pred_1"]
        ).to_csv(
            output_dir / f"confusion_matrix_{name}.csv"
        )

    # 保存词汇表
    with open(
        model_dir / "word2idx.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            data["word2idx"],
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---------- 5. 生成可视化 ----------

    print("\n正在生成实验图表...")

    plot_comparison(
        metrics,
        histories,
        matrices,
        scores,
        y_true,
        data["df"],
        fig_dir
    )

    # ---------- 6. 输出实验结果 ----------

    print("\n模型评估结果：")

    for name, result in metrics.items():
        print(
            f"{name}: "
            f"Accuracy={result['accuracy']:.4f}, "
            f"F1={result['f1']:.4f}, "
            f"AUC={result['roc_auc']:.4f}"
        )

    print(f"\n实验结果已保存至：{output_dir}")
