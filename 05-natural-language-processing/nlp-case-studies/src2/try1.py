import jieba
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False               #设置中文字体，对于数据可视化非常重要

df = pd.read_csv(r"data/clean_weibo_comments.csv")


#第一种数据形式的探索，词频统计与量化分析
def tokenize(text):
    if not isinstance(text, str):
        return []
    return [w for w in jieba.lcut(text) if len(w) > 1]

df["tokens"] = df["clean_text"].apply(tokenize)

# 统计词频
all_words = []
df["tokens"].apply(lambda x: all_words.extend(x))
word_freq = Counter(all_words).most_common(50)

# 保存词频到CSV文件
pd.DataFrame(word_freq, columns=["word", "count"]).to_csv(
    "word_frequency.csv", index=False, encoding="utf-8-sig"
)

# 词频可视化
if word_freq:
    words, counts = zip(*word_freq[:20])  
    
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(words)), counts, color='skyblue')
    plt.xticks(range(len(words)), words, rotation=45, ha='right')
    plt.xlabel('词语')
    plt.ylabel('频次')
    plt.title('微博评论高频词统计 (Top 20)')
    
    # 在每个柱子上显示具体数值
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(count), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig("word_frequency_chart.png", dpi=300, bbox_inches='tight')
    plt.show()


pos_words = set([
    "好", "喜欢", "支持", "赞", "满意", "开心", "高兴", "棒", "优秀", "美好",
    "值得", "推荐", "期待", "祝福", "温暖", "感谢", "幸福", "厉害", "给力", "安慰"
])
neg_words = set([
    "差", "讨厌", "不行", "失望", "垃圾", "生气", "愤怒", "糟糕", "恐惧", "难受",
    "痛苦", "不满", "黑心", "不值", "反对", "谴责", "伤心", "恶心", "没用", "崩溃"
])

def _lexicon_counts(text, pos_words, neg_words):
    if not isinstance(text, str):
        return 0, 0
    tokens = jieba.lcut(text)
    pos = sum(1 for w in tokens if w in pos_words)
    neg = sum(1 for w in tokens if w in neg_words)
    return pos, neg

def _build_pseudo_label(text, pos_words, neg_words):
    pos, neg = _lexicon_counts(text, pos_words, neg_words)
    score = pos - neg
    if score >= 1:
        return 1
    elif score <= -1:
        return 0
    else:
        return None

df["_pseudo_label"] = df["clean_text"].apply(lambda t: _build_pseudo_label(t, pos_words, neg_words))
texts = df["clean_text"].fillna("")

vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, ngram_range=(1, 2), min_df=1, max_df=0.95)
X_all = vectorizer.fit_transform(texts)

mask = df["_pseudo_label"].notnull()
if mask.sum() >= 10 and len(set(df.loc[mask, "_pseudo_label"])) >= 2:
    y = df.loc[mask, "_pseudo_label"].astype(int).values
    X_train = X_all[mask.values]
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    clf.fit(X_train, y)
    probs = clf.predict_proba(X_all)[:, 1]
    df["情感偏向(越接近1越积极)"] = probs
else:
    probs = []
    for t in texts:
        p, n = _lexicon_counts(t, pos_words, neg_words)
        proba = (p + 1.0) / (p + n + 2.0)
        probs.append(proba)
    df["情感偏向(越接近1越积极)"] = probs

df.to_csv("sentiment_weibo.csv", index=False, encoding="utf-8-sig")

print(df[["clean_text", "情感偏向(越接近1越积极)"]].head())

from wordcloud import WordCloud

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

positive = sum(df["情感偏向(越接近1越积极)"] > 0.6)
negative = sum(df["情感偏向(越接近1越积极)"] < 0.4)
neutral = len(df) - positive - negative

plt.figure(figsize=(6,4))
plt.bar(["积极", "中性", "消极"], [positive, neutral, negative], 
        color=['green', 'gray', 'red'])
plt.title("微博评论情感分布")
plt.xlabel("情绪类型")
plt.ylabel("数量")

# 添加数值标签
plt.text(0, positive + 2, str(positive), ha='center', va='bottom')
plt.text(1, neutral + 2, str(neutral), ha='center', va='bottom')
plt.text(2, negative + 2, str(negative), ha='center', va='bottom')

plt.savefig("sentiment_distribution.png", dpi=300, bbox_inches='tight')
plt.show()
