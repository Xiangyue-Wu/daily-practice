import jieba
import jieba.analyse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import matplotlib

# 设置matplotlib字体，防止中文乱码 (根据系统字体自行调整，如 SimHei, Arial Unicode MS)
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

def print_separator(title):
    print(f"\n{'='*20} {title} {'='*20}")

# ==========================================
# 实验内容一：文本相似度实验
# ==========================================
def experiment_text_similarity():
    print_separator("实验一：文本相似度")
    
    # 1. 准备语料 (模拟几条关于科技和体育的短文本)
    corpus = [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质。",  # 文本 0 (AI)
        "机器学习是人工智能的核心，是使计算机具有智能的根本途径。", # 文本 1 (AI)
        "足球是一项全球性的体育运动，深受大众喜爱。",             # 文本 2 (Sports)
        "世界杯足球赛是世界上最高荣誉、最高规格的足球比赛。"       # 文本 3 (Sports)
    ]
    
    print("原始语料：")
    for i, text in enumerate(corpus):
        print(f"文本 {i}: {text}")

    # 2. 分词处理
    corpus_cut = [" ".join(jieba.cut(text)) for text in corpus]
    
    # 3. 计算 TF-IDF 向量
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus_cut)
    
    # 4. 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(tfidf_matrix)
    
    print("\n相似度矩阵 (Cosine Similarity):")
    print(similarity_matrix)
    
    # 5. 结果分析 (找出最相似的文本对，排除自身)
    print("\n文本相似度分析：")
    rows, cols = similarity_matrix.shape
    for i in range(rows):
        # 找到当前文本最相似的另一个文本 (排除自身 1.0)
        # argsort 返回索引，[-2] 是第二大的值（第一大是自身）
        most_similar_index = similarity_matrix[i].argsort()[-2] 
        similarity_score = similarity_matrix[i][most_similar_index]
        print(f"与 [文本 {i}] 最相似的是 [文本 {most_similar_index}]，相似度为: {similarity_score:.4f}")

    # 6. 可视化 (简单的热力图)
    try:
        plt.figure(figsize=(8, 6))
        plt.imshow(similarity_matrix, cmap='Blues')
        plt.colorbar()
        plt.title('文本相似度热力图')
        plt.xlabel('文本编号')
        plt.ylabel('文本编号')
        # plt.show() # 在本地运行时取消注释以显示图表
        print("\n[提示] 已生成相似度热力图代码，本地运行可查看可视化结果。")
    except Exception as e:
        print(f"可视化出错 (可能是字体原因): {e}")

# ==========================================
# 实验内容二：关键词提取
# ==========================================
def experiment_keyword_extraction():
    print_separator("实验二：关键词提取")
    
    # 样本文本 (关于自然语言处理的介绍)
    text = (
        "自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。"
        "它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。"
        "自然语言处理是一门融语言学、计算机科学、数学于一体的科学。"
        "因此，这一领域的研究将涉及自然语言，即人们日常使用的语言，"
        "所以它与语言学的研究有着密切的联系。"
    )
    print(f"待提取文本：\n{text[:50]}... (略)\n")

    # 方法 1: TF-IDF 算法提取
    print("--- 基于 TF-IDF 算法提取关键词 ---")
    # topK: 返回几个关键词, withWeight: 是否返回权重
    keywords_tfidf = jieba.analyse.extract_tags(text, topK=5, withWeight=True)
    for word, weight in keywords_tfidf:
        print(f"关键词: {word}, 权重: {weight:.4f}")

    # 方法 2: TextRank 算法提取
    print("\n--- 基于 TextRank 算法提取关键词 ---")
    keywords_textrank = jieba.analyse.textrank(text, topK=5, withWeight=True)
    for word, weight in keywords_textrank:
        print(f"关键词: {word}, 权重: {weight:.4f}")
        
    print("\n分析：TF-IDF 偏向于在当前文档频率高但在语料库频率低的词；TextRank 偏向于词与词之间的共现关系。")

# ==========================================
# 实验内容三：编辑距离 (算法实现与应用)
# ==========================================
def min_edit_distance(word1, word2):
    """
    手动实现编辑距离算法 (动态规划)
    """
    m = len(word1)
    n = len(word2)
    
    # 初始化 DP 表
    # dp[i][j] 表示 word1的前i个字符 转换到 word2的前j个字符 所需的最少操作数
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # 初始化第一列 (word1 变为空字符串，需要删除 i 次)
    for i in range(m + 1):
        dp[i][0] = i
        
    # 初始化第一行 (空字符串 变为 word2，需要插入 j 次)
    for j in range(n + 1):
        dp[0][j] = j
        
    # 填充 DP 表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                cost = 0
            else:
                cost = 1
            
            dp[i][j] = min(
                dp[i-1][j] + 1,      # 删除 (Deletion)
                dp[i][j-1] + 1,      # 插入 (Insertion)
                dp[i-1][j-1] + cost  # 替换 (Substitution)
            )
            
    return dp[m][n]

def experiment_edit_distance_app():
    print_separator("实验三：编辑距离应用 (简易纠错)")
    
    # 场景：简单的单词拼写纠错
    # 词典库
    vocab = ["apple", "banana", "orange", "python", "java", "intelligence", "language"]
    
    # 用户输入的错误单词
    user_inputs = ["aple", "pythoon", "langage", "intelegence"]
    
    print(f"标准词典: {vocab}")
    
    for user_word in user_inputs:
        print(f"\n用户输入: '{user_word}'")
        
        # 计算该输入与词典中每个词的编辑距离
        distances = []
        for real_word in vocab:
            dist = min_edit_distance(user_word, real_word)
            distances.append((real_word, dist))
        
        # 排序，找出距离最近的词
        distances.sort(key=lambda x: x[1])
        
        # 输出结果
        best_guess = distances[0]
        print(f" -> 猜测您想输入的是: '{best_guess[0]}' (编辑距离: {best_guess[1]})")
        print(f" -> 候选词 Top 3: {distances[:3]}")

if __name__ == "__main__":
    print("开始运行《自然语言处理》实验二代码...")
    
    experiment_text_similarity()
    experiment_keyword_extraction()
    experiment_edit_distance_app()
    
    print("\n所有实验步骤执行完毕。")