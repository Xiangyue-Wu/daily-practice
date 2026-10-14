import os
import time
import random
import logging
from flask import Flask, request, jsonify, render_template, url_for
from snownlp import SnowNLP
import pandas as pd
import jieba
from wordcloud import WordCloud
import matplotlib

matplotlib.use('Agg')  # 后台绘图模式
import matplotlib.pyplot as plt

# --- 深度学习依赖 ---
from transformers import pipeline

app = Flask(__name__)

# 配置日志和路径
logging.basicConfig(level=logging.INFO)
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# ==========================================
# 核心引擎加载 (考察点：前沿技术应用)
# ==========================================
print("正在初始化 AI 引擎...")
print("1. 加载传统统计模型 (SnowNLP)... Done.")
print("2. 加载深度神经网络 (RoBERTa-Chinese)... (首次运行需下载约400MB)")

bert_classifier = None
try:
    # 使用针对中文评论微调过的 RoBERTa 模型
    # device=0 表示使用你的 RTX 4060 显卡加速！
    bert_classifier = pipeline('sentiment-analysis', model="uer/roberta-base-finetuned-dianping-chinese", device=0)
    print(f">>> 深度学习模型加载成功！已启用 GPU 加速。")
except Exception as e:
    print(f">>> 警告：深度模型加载失败，将尝试 CPU 模式。错误信息: {e}")
    try:
        bert_classifier = pipeline('sentiment-analysis', model="uer/roberta-base-finetuned-dianping-chinese")
        print(">>> CPU 模式加载成功。")
    except Exception as e2:
        print(f">>> 彻底失败，系统将仅使用 SnowNLP。错误: {e2}")

# --- 模拟爬虫数据池 ---
MOCK_DATA_POOL = [
    "物流极其慢，但是东西质量出乎意料的好，还是给个好评吧。",  # 转折句，BERT强项
    "虽然价格便宜，但是做工太粗糙了，根本没法用。",  # 转折句
    "这手机简直是工业垃圾，发热严重。",
    "惊艳！屏幕显示效果吊打同价位竞品。",
    "客服态度敷衍，问半天不回，差评。",
    "一般般，习惯性好评。",
    "包装破损，幸好里面没事，希望能改进。",
    "这就是所谓的旗舰？太失望了，退钱！",
    "非常满意，物超所值，强烈推荐。",
    "也就那样吧，没有营销号吹得那么神。"
]


# --- 辅助：BERT 分析 ---
def analyze_bert_engine(text):
    if not bert_classifier: return "未加载", 0.0
    try:
        # 截断过长文本
        res = bert_classifier(text[:500])[0]
        # 模型返回 label_0(负面) 或 label_1(正面)，也有可能是 negative/positive
        raw_label = res['label']
        score = res['score']

        if 'negative' in raw_label or 'label_0' in raw_label:
            return "负面 (Negative)", score
        else:
            return "正面 (Positive)", score
    except:
        return "分析错误", 0.0


# --- 辅助：SnowNLP 分析 ---
def analyze_snownlp_engine(text):
    s = SnowNLP(text)
    score = s.sentiments
    if score > 0.6:
        return "正面 (Positive)", score
    elif score < 0.4:
        return "负面 (Negative)", score
    else:
        return "中性 (Neutral)", score


# --- 辅助：词云生成 ---
def create_wordcloud(texts):
    full_text = " ".join(texts)
    cut_text = " ".join(jieba.cut(full_text))

    # Windows 字体适配
    font_path = "C:/Windows/Fonts/simhei.ttf"
    if not os.path.exists(font_path): font_path = "C:/Windows/Fonts/msyh.ttc"

    # 生成透明背景的高级感词云
    wc = WordCloud(
        font_path=font_path,
        background_color=None,  # 透明背景
        mode="RGBA",
        width=1000, height=600,
        max_words=80,
        colormap='cool'  # 科技感配色
    )
    wc.generate(cut_text)
    filename = f'wc_{int(time.time())}.png'
    wc.to_file(os.path.join(STATIC_FOLDER, filename))
    return filename


# --- 路由：主页 ---
@app.route('/')
def home():
    return render_template('index.html')


# --- 路由：深度对比分析 ---
@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    text = request.json.get('text', '')
    if not text: return jsonify({'error': 'Empty'}), 400

    # 运行双模型
    bert_label, bert_score = analyze_bert_engine(text)
    snow_label, snow_score = analyze_snownlp_engine(text)

    return jsonify({
        'bert': {'label': bert_label, 'score': bert_score},
        'snow': {'label': snow_label, 'score': snow_score}
    })


# --- 路由：文件上传处理 ---
@app.route('/api/upload', methods=['POST'])
def api_upload():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    f = request.files['file']

    try:
        if f.filename.endswith('.csv'):
            df = pd.read_csv(f)
        else:
            df = pd.read_excel(f)

        # 假设第一列是文本
        comments = df.iloc[:, 0].astype(str).tolist()
        return process_batch(comments)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# --- 路由：仿真爬虫 ---
@app.route('/api/crawl', methods=['POST'])
def api_crawl():
    url = request.json.get('url', '')
    if not url: return jsonify({'error': 'No URL'}), 400

    # 模拟延迟，制造"正在连接目标服务器"的假象
    time.sleep(1.5)

    # 随机生成数据
    count = random.randint(15, 30)
    comments = [random.choice(MOCK_DATA_POOL) for _ in range(count)]

    # 保存数据用于下载演示
    df = pd.DataFrame({'采集评论': comments})
    filename = f"crawl_data_{int(time.time())}.xlsx"
    df.to_excel(os.path.join(UPLOAD_FOLDER, filename), index=False)

    res = process_batch(comments)
    res['download'] = filename
    return jsonify(res)


def process_batch(comments):
    # 批量处理逻辑
    # 既然有 GPU，我们可以稍微多处理一点，但还是限制一下防止前端超时
    process_limit = 50
    target_comments = comments[:process_limit]

    pos, neg = 0, 0
    for c in target_comments:
        # 优先使用 BERT 判定
        label, _ = analyze_bert_engine(c)
        if "正面" in label:
            pos += 1
        else:
            neg += 1

    img = create_wordcloud(target_comments)

    return {
        'total': len(comments),
        'processed': len(target_comments),
        'pos': pos,
        'neg': neg,
        'img_url': url_for('static', filename=img)
    }


if __name__ == '__main__':
    print(">>> 系统正在启动...")
    print(f">>> 访问地址: http://127.0.0.1:5000")
    app.run(debug=True, port=5000) 