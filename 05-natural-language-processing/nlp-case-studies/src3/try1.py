import jieba
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from wordcloud import WordCloud

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

def tokenize(text):
    if not isinstance(text, str):
        return []
    return [w for w in jieba.lcut(text) if len(w) > 1]

def build_vectorizer():
    v = TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None, ngram_range=(1, 2), min_df=1, max_df=0.95)
    return v

def split_train_test(df):
    df_use = df.copy()
    df_use = df_use[df_use["态度"].isin([1, 2, 3])]
    X = df_use["clean_text"].fillna("").astype(str).values
    y = df_use["态度"].astype(int).values
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        print("警告：某些类别样本过少，改为非分层划分。")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, vectorizer):
    Xv_train = vectorizer.fit_transform(list(X_train))
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", multi_class="ovr")
    clf.fit(Xv_train, y_train)
    return clf

def evaluate_model(clf, vectorizer, X_test, y_test):
    Xv_test = vectorizer.transform(list(X_test))
    y_pred = clf.predict(Xv_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred, labels=[1,2,3])
    print("测试集准确率:", acc)
    print("分类报告:\n", report)
    print("混淆矩阵:")
    print(cm)
    return None

def compute_tokens_and_frequency(df):
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
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(count), ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig("word_frequency_chart.png", dpi=300, bbox_inches='tight')
        plt.show()
    return all_words

def visualize_wordcloud(all_words):
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

def visualize_sentiment_distribution(df):
    positive = sum(df["情感偏向(越接近1越积极)"] > 0.6)
    negative = sum(df["情感偏向(越接近1越积极)"] < 0.4)
    neutral = len(df) - positive - negative
    plt.figure(figsize=(6,4))
    plt.bar(["积极", "中性", "消极"], [positive, neutral, negative], color=['green', 'gray', 'red'])
    plt.title("微博评论情感分布")
    plt.xlabel("情绪类型")
    plt.ylabel("数量")
    plt.text(0, positive + 2, str(positive), ha='center', va='bottom')
    plt.text(1, neutral + 2, str(neutral), ha='center', va='bottom')
    plt.text(2, negative + 2, str(negative), ha='center', va='bottom')
    plt.savefig("sentiment_distribution.png", dpi=300, bbox_inches='tight')
    plt.show()
    return None

def run_full_model_and_visualize(df):
    texts_all = df["clean_text"].fillna("").astype(str).values
    mask_label = df["态度"].isin([1,2,3])
    vectorizer_full = build_vectorizer()
    Xv_all = vectorizer_full.fit_transform(list(texts_all))
    y_all = df.loc[mask_label, "态度"].astype(int).values
    clf_full = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", multi_class="ovr")
    clf_full.fit(Xv_all[mask_label.values], y_all)
    proba_all = clf_full.predict_proba(Xv_all)
    idx_pos = list(clf_full.classes_).index(3) if 3 in clf_full.classes_ else 0
    df["预测态度"] = clf_full.predict(Xv_all)
    df["情感偏向(越接近1越积极)"] = proba_all[:, idx_pos]
    df.to_csv("sentiment_weibo.csv", index=False, encoding="utf-8-sig")
    print(df[["clean_text", "情感偏向(越接近1越积极)"]].head())
    all_words = compute_tokens_and_frequency(df)
    visualize_wordcloud(all_words)
    visualize_sentiment_distribution(df)
    return None

def main(df):
    # 1.划分训练集与测试集
    X_train, X_test, y_train, y_test = split_train_test(df)
    # 2.训练监督模型
    vectorizer = build_vectorizer()
    clf = train_model(X_train, y_train, vectorizer)
    # 3.评估模型
    evaluate_model(clf, vectorizer, X_test, y_test)
    # 4.用全量数据拟合并可视化
    run_full_model_and_visualize(df)
    return None

if __name__ == "__main__":
    DATA_PATH = r"data/clean_weibo_comments.csv"
    df = pd.read_csv(DATA_PATH)
    main(df)
