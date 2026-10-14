import os
import jieba
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import snapshot_download
import torch
from wordcloud import WordCloud

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

def load_or_download_model(model_name, local_dir, force_download=False, use_mirror=True):
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    model_path = os.path.join(local_dir, model_name.replace('/', '--'))
    if os.path.exists(model_path) and not force_download:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForSequenceClassification.from_pretrained(model_path)
            return model, tokenizer
        except Exception as e:
            force_download = True
    if force_download or not os.path.exists(model_path):
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                resume_download=True,
                local_dir_use_symlinks=False
            )
        except Exception as e:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            os.makedirs(model_path, exist_ok=True)
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            return model, tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return model, tokenizer

def tokenize(text):
    if not isinstance(text, str):
        return []
    return [w for w in jieba.lcut(text) if len(w) > 1]

def read_data(csv_path):
    df = pd.read_csv(csv_path)
    return df

def word_frequency_analysis(df):
    df["tokens"] = df["clean_text"].apply(tokenize)
    all_words = []
    df["tokens"].apply(lambda x: all_words.extend(x))
    word_freq = Counter(all_words).most_common(50)
    pd.DataFrame(word_freq, columns=["word", "count"]).to_csv(
        "word_frequency.csv", index=False, encoding="utf-8-sig"
    )
    if word_freq:
        words, counts = zip(*word_freq[:20])
        plt.figure(figsize=(12, 8))
        bars = plt.bar(range(len(words)), counts, color='skyblue')
        plt.xticks(range(len(words)), words, rotation=45, ha='right')
        plt.xlabel('词语')
        plt.ylabel('频次')
        plt.title('微博评论高频词统计 (Top 20)')
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     str(count), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig("word_frequency_chart.png", dpi=300, bbox_inches='tight')
        plt.show()
    return all_words, word_freq, df

def generate_wordcloud(all_words):
    text = " ".join(all_words)
    wc = WordCloud(
        font_path="msyh.ttc",
        width=800, height=500,
        background_color="white"
    ).generate(text)
    plt.figure(figsize=(10, 6))
    plt.imshow(wc)
    plt.axis("off")
    plt.title("微博评论词云图")
    plt.savefig("wordcloud_weibo.png", dpi=300, bbox_inches='tight')
    plt.show()
    return None

def predict_sentiment_with_model(df, model, tokenizer):
    texts = df["clean_text"].fillna("").tolist()
    probs = []
    model.eval()
    for t in texts:
        inputs = tokenizer(t, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            if logits.shape[-1] == 2:
                prob = torch.softmax(logits, dim=-1)[0, 1].item()
            else:
                prob = torch.softmax(logits, dim=-1)[0].max().item()
        probs.append(prob)
    df["情感偏向(越接近1越积极)"] = probs
    df.to_csv("sentiment_weibo.csv", index=False, encoding="utf-8-sig")
    print(df[["clean_text", "情感偏向(越接近1越积极)"]].head())
    return df

def plot_sentiment_distribution(df):
    positive = sum(df["情感偏向(越接近1越积极)"] > 0.6)
    negative = sum(df["情感偏向(越接近1越积极)"] < 0.4)
    neutral = len(df) - positive - negative
    plt.figure(figsize=(6,4))
    plt.bar(["积极", "中性", "消极"], [positive, neutral, negative], 
            color=['green', 'gray', 'red'])
    plt.title("微博评论情感分布")
    plt.xlabel("情绪类型")
    plt.ylabel("数量")
    plt.text(0, positive + 2, str(positive), ha='center', va='bottom')
    plt.text(1, neutral + 2, str(neutral), ha='center', va='bottom')
    plt.text(2, negative + 2, str(negative), ha='center', va='bottom')
    plt.savefig("sentiment_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    return None

def main():
    # 1.加载或下载模型
    model_name = "voidful/albert_chinese_small_sentiment"
    local_dir = r"models/m2"
    model, tokenizer = load_or_download_model(model_name, local_dir, force_download=False, use_mirror=False)
    print(f"加载{model_name}成功")

    # 2.读取数据
    csv_path = r"data/clean_weibo_comments.csv"
    df = read_data(csv_path)

    # 3.词频统计与可视化
    all_words, word_freq, df = word_frequency_analysis(df)

    # 4.词云可视化
    generate_wordcloud(all_words)

    # 5.模型预测情感
    df = predict_sentiment_with_model(df, model, tokenizer)

    # 6.情感分布可视化
    plot_sentiment_distribution(df)
    return None

if __name__ == "__main__":
    main()
