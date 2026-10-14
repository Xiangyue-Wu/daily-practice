from bs4 import BeautifulSoup
import requests
import re
import jieba
from collections import Counter
    # 导入扩展库
import re # 正则表达式库
import collections # 词频统计库
import numpy as np # numpy数据处理库
import jieba # 结巴分词
import wordcloud # 词云展示库
import matplotlib.pyplot as plt # 图像展示库

def get_data(links):
    soups=[]
    for i in links:
        res = requests.get(i)#新闻的网址
        res.encoding = res.apparent_encoding  
        # 根据网站的编码方式修改解码方式，因为网页的文字编码方式多种多样有UTF-8 GBK这些解码方式如果不一致容易发生乱码，所以解码方式最好不要统一，而是交给网页自己来决定
        soups.append(BeautifulSoup(res.text, 'html5lib'))#使用html5lib样式来解析网页，看不懂没关系
    print(soups[-1])#查看页面源代码

    return soups

def wash_data(soups):
    t_c={}
    for soup in soups:
        title = soup.find('h1').text.strip()
        content = soup.find('div', id='detail').text.strip()
        
        # 进一步清洗内容
        # 移除HTML标签
        content = re.sub(r'<[^>]+>', '', content)
        # 移除特殊符号和多余空白
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'【纠错】|【责任编辑.*】', '', content)
        content = content.strip()
        t_c[title]=content
        print("=== 标题 ===")
        print(title)
        print("\n=== 内容 ===")
        print(content)
        print("\n" + "="*50 + "\n")

    return t_c

def statistic(t_c):
    result = ''.join(str(key) + str(value) for key, value in t_c.items())
    # 简单停用词列表（可扩展）
    stop_words = set(['的', '是', '在', '了', '和', '与', '等', '对', '如', '有', '这', '为', '于', '中'])
    
    all_content = " ".join(t_c.values())
    
    print(f"\n=== 综合统计分析 ===")
    
    # 词频统计
    words = jieba.lcut(all_content)
    filtered_words = [word for word in words if word not in stop_words and len(word) > 1]
    word_freq = Counter(filtered_words)
    print("词频统计 (前10个):")
    for word, freq in word_freq.most_common(10):
        print(f"{word}: {freq}")
    
    # 句子长度分析
    sentences = re.split(r'[。！？]', all_content)
    sentences = [s.strip() for s in sentences if s.strip()]
    lengths = [len(s) for s in sentences]
    if lengths:
        avg_length = sum(lengths) / len(lengths)
        print(f"\n句子数量: {len(sentences)}")
        print(f"平均句子长度: {avg_length:.2f} 字符")
        print(f"最长句子长度: {max(lengths)} 字符")
        print(f"最短句子长度: {min(lengths)} 字符")
    else:
        print("无有效句子")
    
    print("=" * 50)
    
    return all_content

def cloud(string_data):
# # 读取文件
#     fn = open('article.txt') # 打开文件
#     string_data = fn.read() # 读出整个文件
#     fn.close() # 关闭文件

# 文本预处理
    pattern = re.compile(r'\t|\n|\.|-|:|;|\)|\(|\?|"') # 定义正则表达式匹配模式
    string_data = re.sub(pattern, '', string_data) # 将符合模式的字符去除

# 文本分词
    seg_list_exact = jieba.cut(string_data, cut_all = False) # 精确模式分词
    object_list = []
    remove_words = [u'的', u'，',u'和', u'是', u'随着', u'对于', u'对',u'等',u'能',u'都',u'。',u' ',u'、',u'中',u'在',u'了',
                u'通常',u'如果',u'我们',u'需要'] # 自定义去除词库

    for word in seg_list_exact: # 循环读出每个分词
        if word not in remove_words: # 如果不在去除词库中
            object_list.append(word) # 分词追加到列表

# 词频统计
    word_counts = collections.Counter(object_list) # 对分词做词频统计
    word_counts_top10 = word_counts.most_common(10) # 获取前10最高频的词
    print (word_counts_top10) # 输出检查

# 词频展示
    wc = wordcloud.WordCloud(
    font_path='C:/Windows/Fonts/simhei.ttf', # 设置字体格式
    max_words=200, # 最多显示词数
    max_font_size=100, # 字体最大值
    background_color='white' # 设置背景颜色为白色
)

    wc.generate_from_frequencies(word_counts) # 从字典生成词云
    
    plt.imshow(wc) # 显示词云
    plt.axis('off') # 关闭坐标轴
    plt.show() # 显示图像

    return None


def main():
    # 1. 文章链接
    links = ['https://www.news.cn/politics/leaders/20251118/e5fcb29e94854583ac274feef1358848/c.html',
    'https://www.news.cn/politics/leaders/20251118/cec4c71ac8c6417fa930aa0c915e9174/c.html',
    'https://www.news.cn/politics/20251118/0c9595753db2423aa9ec472de7028950/c.html',
    'https://www.news.cn/fortune/20251118/8acdaa74b5e74c19b7be5bc58157b9f4/c.html',
    
    ]

    # 2.获取未清洗的内容
    soups = get_data(links)

    # 3. 清洗内容（采集文本进行清洗，包括去除 HTML 标签、特殊符号与停用词）
    t_c=wash_data(soups)
    
    # 4. 统计分析（对清洗后的文本进行统计分析，如词频统计、句子长度分析等）
    all_content=statistic(t_c)

    # 5. 生成词云（对清洗后的文本进行词云生成）
    cloud(all_content)
    
    return None

if __name__ == "__main__":
    main()
