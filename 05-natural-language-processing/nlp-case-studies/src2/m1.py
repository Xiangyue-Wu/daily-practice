import os
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

def load_or_download_model(model_name, local_dir, force_download=False, use_mirror=True):
    """
    加载或下载Hugging Face模型
    
    Args:
        model_name (str): Hugging Face模型名称，如 "bert-base-uncased"
        local_dir (str): 本地保存路径
        force_download (bool): 是否强制重新下载
        use_mirror (bool): 是否使用国内镜像加速
    
    Returns:
        model, tokenizer: 加载的模型和分词器
    """
    
    # # 设置镜像源（国内用户推荐）
    if use_mirror:
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 检查本地模型是否存在 
    model_path = os.path.join(local_dir, model_name.replace('/', '--'))
    
    if os.path.exists(model_path) and not force_download:
        print(f"✅ 找到本地模型: {model_path}")
        try:
            # 加载本地模型 [5](@ref)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            print("✅ 本地模型加载成功")
            return model, tokenizer
        except Exception as e:
            print(f"❌ 本地模型损坏，重新下载: {e}")
            force_download = True
    
    # 下载模型 [1,4](@ref)
    if force_download or not os.path.exists(model_path):
        print(f"📥 开始下载模型: {model_name}")
        
        try:
            # 方法1: 使用snapshot_download下载完整模型 [1,4](@ref)
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
                resume_download=True,
                local_dir_use_symlinks=False
            )
            print(f"✅ 模型下载完成: {model_path}")
            
        except Exception as e:
            print(f"❌ 下载失败: {e}")
            # 方法2: 使用transformers自动下载 [3,5](@ref)
            print("尝试使用transformers自动下载...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            
            # 保存到本地 [5](@ref)
            os.makedirs(model_path, exist_ok=True)
            tokenizer.save_pretrained(model_path)
            model.save_pretrained(model_path)
            print(f"✅ 模型保存到: {model_path}")
            return model, tokenizer
    
    # 加载下载的模型
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    return model, tokenizer

def check_model_files(model_path):
    """
    检查模型文件是否完整 [3,8](@ref)
    """
    required_files = ['config.json', 'pytorch_model.bin', 'vocab.txt']
    if not os.path.exists(model_path):
        return False
    
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            return False
    return True

# 使用示例
if __name__ == "__main__":
    # 模型配置
    MODEL_NAME = "voidful/albert_chinese_small_sentiment"  # 可以替换为任何Hugging Face模型
    LOCAL_DIR = r'models/m2'  # 本地保存目录
    
    # 加载或下载模型
    model, tokenizer = load_or_download_model(
        model_name=MODEL_NAME,
        local_dir=LOCAL_DIR,
        force_download=False,  # 设置为True可强制重新下载
        use_mirror=False       # 国内用户建议开启
    )
    
    print("🎉 模型准备就绪！")
    print(f"模型架构: {model.__class__.__name__}")
    print(f"分词器: {tokenizer.__class__.__name__}")
    
    # 测试模型
    test_text = "Hello, this is a test sentence."
    inputs = tokenizer(test_text, return_tensors="pt")
    outputs = model(**inputs)
    
    print(f"输入文本: {test_text}")
    print(f"输出形状: {outputs.last_hidden_state.shape}")