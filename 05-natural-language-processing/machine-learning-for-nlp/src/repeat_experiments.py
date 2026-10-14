import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from main import (
    BATCH_SIZE,
    DATA_PATH,
    EMBED_DIM,
    EPOCHS,
    LEARNING_RATE,
    OUTPUT_DIR,
    RANDOM_SEED,
    load_data,
    prepare_data,
    tokenize_data,
)
from models import TextCNN, TextLSTM
from utils import get_metrics, predict_deep, train_model


TRAINING_SEEDS = [42, 123, 2026]
REPEATED_OUTPUT_DIR = OUTPUT_DIR / "repeated_runs"


def set_training_seed(seed):
    """设置模型初始化、训练顺序和 CUDA 使用的随机种子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_train_loader(dataset, seed):
    """为每次训练建立由对应种子控制的全新 DataLoader。"""
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def build_summary(seed_results):
    """按模型计算均值和样本标准差（ddof=1）。"""
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    rows = []

    for model_name, group in seed_results.groupby("model", sort=False):
        row = {
            "model": model_name,
            "runs": len(group),
            "std_ddof": 1,
        }

        for metric in metric_names:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows)


def plot_stability(summary, output_path):
    """绘制 Accuracy、F1 和 ROC-AUC 的均值与标准差。"""
    metrics = ["accuracy", "f1", "roc_auc"]
    labels = ["Accuracy", "F1", "ROC-AUC"]
    model_names = summary["model"].tolist()
    x = np.arange(len(metrics))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9, 6))

    for index, model_name in enumerate(model_names):
        row = summary.loc[summary["model"] == model_name].iloc[0]
        means = [row[f"{metric}_mean"] for metric in metrics]
        stds = [row[f"{metric}_std"] for metric in metrics]
        positions = x + (index - (len(model_names) - 1) / 2) * width

        bars = ax.bar(
            positions,
            means,
            width,
            yerr=stds,
            capsize=6,
            label=model_name,
        )

        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{mean:.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set(
        title="Deep Learning Model Stability (3 Seeds)",
        ylabel="Mean Score (error bars: sample standard deviation)",
        ylim=(0, 1.05),
        xticks=x,
        xticklabels=labels,
    )
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    if RANDOM_SEED != 42:
        raise ValueError("数据集划分随机种子必须保持为 42")

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            "未找到已有数据集，请先运行单次完整实验下载数据"
        )

    REPEATED_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 数据仅准备一次，确保六次训练使用完全相同的划分、词汇表和测试集。
    set_training_seed(RANDOM_SEED)
    df = tokenize_data(load_data())
    data = prepare_data(df)
    train_dataset = data["train_loader"].dataset
    test_loader = DataLoader(
        data["test_loader"].dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    y_true = np.asarray(data["y_test"])

    if len(y_true) != 1553:
        raise ValueError(f"测试集样本数异常：{len(y_true)}")

    model_factories = {
        "TextCNN": lambda: TextCNN(data["vocab_size"], EMBED_DIM),
        "Bi-LSTM": lambda: TextLSTM(
            data["vocab_size"],
            EMBED_DIM,
            hidden_dim=64,
        ),
    }
    result_rows = []
    loss_rows = []

    for model_name, model_factory in model_factories.items():
        for seed in TRAINING_SEEDS:
            print(f"\n训练 {model_name}，seed={seed}")
            set_training_seed(seed)
            train_loader = build_train_loader(train_dataset, seed)
            model = model_factory().to(data["device"])
            losses = train_model(
                model,
                train_loader,
                data["device"],
                EPOCHS,
                LEARNING_RATE,
            )
            y_eval, y_pred, y_score = predict_deep(
                model,
                test_loader,
                data["device"],
            )

            if len(y_eval) != 1553 or not np.array_equal(y_eval, y_true):
                raise ValueError("重复实验的测试集与固定测试集不一致")

            metrics = get_metrics(y_true, y_pred, y_score)
            result_rows.append(
                {
                    "model": model_name,
                    "seed": seed,
                    **metrics,
                }
            )

            for epoch, loss in enumerate(losses, start=1):
                loss_rows.append(
                    {
                        "model": model_name,
                        "seed": seed,
                        "epoch": epoch,
                        "loss": loss,
                    }
                )

            print(
                f"  Accuracy={metrics['accuracy']:.4f}, "
                f"F1={metrics['f1']:.4f}, "
                f"AUC={metrics['roc_auc']:.4f}, "
                f"test_samples={len(y_eval)}"
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    seed_results = pd.DataFrame(result_rows)[
        [
            "model",
            "seed",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ]
    ]
    summary = build_summary(seed_results)

    seed_results.to_csv(
        REPEATED_OUTPUT_DIR / "seed_results.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(
        REPEATED_OUTPUT_DIR / "summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(loss_rows).to_csv(
        REPEATED_OUTPUT_DIR / "loss_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    plot_stability(
        summary,
        REPEATED_OUTPUT_DIR / "stability_comparison.png",
    )

    print("\n三随机种子实验完成")
    print("结果目录：outputs/repeated_runs")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
