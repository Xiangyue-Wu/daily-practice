import collections
import re

import jieba
import matplotlib.pyplot as plt
import requests
import wordcloud
from bs4 import BeautifulSoup
from collections import Counter


# 1.步骤一：获取网页数据
def get_data(links):
    # 初始化存储网页解析结果的列表
    soups = []
    # 遍历每个链接，获取网页内容
    for i in links:
        # 发送GET请求获取网页内容
        res = requests.get(i)
        # 根据网页的实际编码方式设置解码方式，避免乱码
        res.encoding = res.apparent_encoding
        # 使用html5lib解析器解析网页内容，得到BeautifulSoup对象并添加到列表
        soups.append(BeautifulSoup(res.text, 'html5lib'))
    # 打印最后一个网页的解析结果（查看页面源代码）

    print(soups[-1])
    
    return soups


# 2.步骤二：清洗网页数据
def wash_data(soups):
    # 初始化存储标题和对应内容的字典
    t_c = {}
    # 遍历每个解析后的网页对象
    for soup in soups:
        # 提取标题（假设标题在<h1>标签中），去除首尾空白
        title = soup.find('h1').text.strip()
        # 提取内容（假设内容在id为'detail'的<div>标签中），去除首尾空白
        content = soup.find('div', id='detail').text.strip()
        # 进一步清洗内容：移除HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        # 移除特殊符号和多余空白（将多个空白替换为单个空格）
        content = re.sub(r'\s+', ' ', content)
        # 移除特定干扰文本（如"【纠错】"、"【责任编辑...】"）
        content = re.sub(r'【纠错】|【责任编辑.*】', '', content)
        # 再次去除首尾空白
        content = content.strip()
        # 将标题和清洗后的内容存入字典
        t_c[title] = content
    return t_c


# 3.步骤三：文本统计分析
def statistic(t_c):
    # 拼接所有标题和内容（仅用于后续处理，此处未直接使用）
    result = ''.join(str(key) + str(value) for key, value in t_c.items())
    # 定义简单停用词列表（可根据需求扩展）
    stop_words = set(['的', '是', '在', '了', '和', '与', '等', '对', '如', '有', '这', '为', '于', '中'])
    # 拼接所有内容文本
    all_content = " ".join(t_c.values())
    
    # 打印统计分析标题
    print(f"\n=== 综合统计分析 ===")
    # 词频统计
    # 使用jieba对文本进行分词
    words = jieba.lcut(all_content)
    # 过滤停用词和长度为1的词
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    # 统计词频
    word_freq = Counter(filtered_words)
    
    # 打印前10个高频词
    print("词频统计 (前10个):")
    for word, freq in word_freq.most_common(10):
        print(f"{word}: {freq}")
    
    # 句子长度分析
    # 使用标点符号分割句子
    sentences = re.split(r'[。！？]', all_content)
    # 过滤空句子并去除首尾空白
    sentences = [s.strip() for s in sentences if s.strip()]
    # 计算每个句子的长度
    lengths = [len(s) for s in sentences]
    return all_content


# 4.步骤四：生成词云
def cloud(string_data):
    # 文本预处理：去除制表符、换行符、标点等干扰字符
    pattern = re.compile(r'\t|\n|\.|-|:|;|\)|\(|\?|"')
    string_data = re.sub(pattern, '', string_data)

    # 文本分词：使用jieba精确模式分词
    seg_list_exact = jieba.cut(string_data, cut_all=False)
    object_list = []
    # 自定义需要去除的词（扩展停用词）
    remove_words = [u'的', u'，', u'和', u'是', u'随着', u'对于', u'对', u'等', u'能', u'都', u'。', u' ', u'、', u'中', u'在', u'了',
                    u'通常', u'如果', u'我们', u'需要']
    
    # 过滤去除词，保留有效分词
    for word in seg_list_exact:
        if word not in remove_words:
            object_list.append(word)
    
    # 词频统计：对有效分词进行词频统计
    word_counts = collections.Counter(object_list)
    # 获取前10个高频词并打印（用于验证）
    word_counts_top10 = word_counts.most_common(10)
    print(word_counts_top10)
    
    # 生成词云
    wc = wordcloud.WordCloud(
        font_path='C:/Windows/Fonts/simhei.ttf',  # 设置中文字体（确保字体存在）
        max_words=200,  # 最多显示的词数
        max_font_size=100,  # 最大字体大小
        background_color='white'  # 背景颜色为白色
    )
    # 根据词频生成词云
    wc.generate_from_frequencies(word_counts)
    
    # 显示词云
    plt.imshow(wc)
    plt.axis('off')  # 关闭坐标轴
    plt.show()  # 显示图像
    
    return None


# 5.步骤五：主程序与调用
def main():
    # 1. 定义需要爬取的文章链接列表
    links = ['https://www.news.cn/politics/leaders/20251118/e5fcb29e94854583ac274feef1358848/c.html',
             'https://www.news.cn/politics/leaders/20251118/cec4c71ac8c6417fa930aa0c915e9174/c.html',
             'https://www.news.cn/politics/20251118/0c9595753db2423aa9ec472de7028950/c.html',
             'https://www.news.cn/fortune/20251118/8acdaa74b5e74c19b7be5bc58157b9f4/c.html',
             ]
    
    # 2. 获取未清洗的网页内容（解析为BeautifulSoup对象）
    soups = get_data(links)
    
    # 3. 清洗内容（提取标题和正文，去除HTML标签、特殊符号等）
    t_c = wash_data(soups)
    
    # 4. 统计分析（词频统计、句子长度分析等）
    all_content = statistic(t_c)
    
    # 5. 生成并显示词云
    cloud(all_content)

    return None

if __name__ == "__main__":
    main()
