
from pathlib import Path
from urllib.request import urlretrieve
import json
import random

import jieba
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset, DataLoader


# ==================== 实验配置 ====================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ChnSentiCorp_htl_all.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

DATA_URL = (
    "https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/"
    "master/datasets/ChnSentiCorp_htl_all/ChnSentiCorp_htl_all.csv"
)

RANDOM_SEED = 42
TEST_SIZE = 0.2
MAX_FEATURES = 5000
MAX_LEN = 100
EMBED_DIM = 100
BATCH_SIZE = 64
EPOCHS = 3
LEARNING_RATE = 0.001


# ==================== 数据集加载 ====================

def download_dataset():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    if DATA_PATH.exists():
        print("检测到本地数据集")
        return

    print("正在下载数据集...")

    try:
        urlretrieve(DATA_URL, DATA_PATH)
        print("数据集下载完成")
    except Exception as e:
        DATA_PATH.unlink(missing_ok=True)
        raise RuntimeError(
            "数据集下载失败，请手动将 CSV 文件放入 data 目录"
        ) from e


def load_data():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # 统一列名
    df.rename(
        columns={
            "review": "text",
            "content": "text",
            "comment": "text",
            "star": "label"
        },
        inplace=True
    )

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("数据集必须包含文本列和标签列")

    original_size = len(df)

    # 删除无效数据
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")

    df = df.dropna(subset=["label"])
    df = df[df["text"] != ""]
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)

    if df.empty:
        raise ValueError("清洗后没有有效样本")

    df.reset_index(drop=True, inplace=True)

    print(f"原始样本数：{original_size}")
    print(f"清洗后样本数：{len(df)}")

    return df


# ==================== 中文分词 ====================

def tokenize_data(df):
    print("正在进行中文分词...")

    df = df.copy()
    df["cut_text"] = df["text"].apply(
        lambda text: " ".join(jieba.lcut(text))
    )

    return df


# ==================== 词汇表与序列处理 ====================

def build_vocab(texts):
    word2idx = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for text in texts:
        for word in text.split():
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    return word2idx


def encode_text(text, word2idx):
    indices = [
        word2idx.get(word, word2idx["<UNK>"])
        for word in text.split()
    ]

    # 截断和填充
    indices = indices[:MAX_LEN]
    indices += [word2idx["<PAD>"]] * (MAX_LEN - len(indices))

    return indices


class TextDataset(Dataset):
    def __init__(self, texts, labels, word2idx):
        self.texts = torch.tensor(
            [encode_text(t, word2idx) for t in texts],
            dtype=torch.long
        )

        self.labels = torch.tensor(
            list(labels),
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.texts[index], self.labels[index]


# ==================== 数据集准备 ====================

def prepare_data(df):
    # 按照 8:2 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        df["cut_text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED
    )

    print(f"训练集样本数：{len(X_train)}")
    print(f"测试集样本数：{len(X_test)}")

    # 逻辑回归使用 TF-IDF 特征
    vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF 特征维度：{X_train_tfidf.shape[1]}")

    # 深度学习模型使用词汇索引
    word2idx = build_vocab(X_train)

    train_dataset = TextDataset(X_train, y_train, word2idx)
    test_dataset = TextDataset(X_test, y_test, word2idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    print(f"词汇表大小：{len(word2idx)}")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"训练设备：{device}")

    return {
        "X_train_tfidf": X_train_tfidf,
        "X_test_tfidf": X_test_tfidf,
        "y_train": y_train.to_numpy(),
        "y_test": y_test.to_numpy(),
        "train_loader": train_loader,
        "test_loader": test_loader,
        "word2idx": word2idx,
        "vocab_size": len(word2idx),
        "vectorizer": vectorizer,
        "df": df,
        "device": device,
        "output_dir": OUTPUT_DIR,
        "embed_dim": EMBED_DIM,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE
    }


# ==================== 主程序 ====================

def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 50)
    print("中文情感分类对比实验")
    print("=" * 50)

    # 1. 下载和加载数据
    download_dataset()
    df = load_data()

    # 2. 中文分词
    df = tokenize_data(df)

    # 3. 特征提取与数据集构建
    data = prepare_data(df)

    # 4. 保存预处理统计
    summary = {
        "total_samples": len(df),
        "train_samples": len(data["y_train"]),
        "test_samples": len(data["y_test"]),
        "vocab_size": data["vocab_size"],
        "tfidf_features": data["X_train_tfidf"].shape[1],
        "max_len": MAX_LEN,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "label_counts": {
            str(k): int(v)
            for k, v in df["label"].value_counts().items()
        }
    }

    summary_path = OUTPUT_DIR / "preprocessing_summary.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("数据预处理完成")
    print(f"统计结果已保存：{summary_path}")

    # 5. 等待模型与训练模块
    if not (
        (BASE_DIR / "models.py").is_file()
        and (BASE_DIR / "utils.py").is_file()
    ):
        print("尚未添加模型和训练模块，跳过模型训练")
        return

    from utils import run_experiments

    # 6. 运行完整实验
    run_experiments(data)

    print("全部实验完成")


if __name__ == "__main__":
    main()
