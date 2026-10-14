import jieba
import pandas as pd
from collections import Counter
from snownlp import SnowNLP
import matplotlib.pyplot as plt

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


#第二种形式，词云可视化跟情感分析
def sentiment_score(text):
    try:
        return SnowNLP(text).sentiments  # 0~1，越高越积极
    except:
        return 0.5

#上面这部分是使用合适的模型进行情感分析，使用SnowNLP模型计算每条评论的情感分数（sentiments方法，返回0~1值，越接近1越积极）

df["情感偏向(越接近1越积极)"] = df["clean_text"].apply(sentiment_score)

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
