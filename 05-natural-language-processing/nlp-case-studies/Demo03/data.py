#数据预处理部分，指导书中第三点：进行数据预处理

import pandas as pd
import re

# 读取微博评论
df = pd.read_csv("weibo_comments.csv")

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = re.sub(r'@\S+', '', text)  # 去掉 @用户
    text = re.sub(r'http\S+|www.\S+', '', text)  # 去掉链接
    text = re.sub(r'[^\w\u4e00-\u9fa5]', ' ', text)  # 只保留中文/英文字符
    text = re.sub(r'\s+', ' ', text).strip()  # 多余空格
    return text

df["clean_text"] = df["评论内容"].apply(clean_text)
df.to_csv("clean_weibo_comments.csv", index=False, encoding="utf-8-sig")

print("数据清洗完成！已保存为 clean_weibo_comments.csv")
