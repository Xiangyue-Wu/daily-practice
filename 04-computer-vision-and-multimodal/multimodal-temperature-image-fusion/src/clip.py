import os
import pandas as pd
import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import math

# 预训练模型在 Hugging Face 上的名称；脚本会自动下载并缓存到本地
MODEL_ID = "openai/clip-vit-base-patch32"
# 本地缓存目录（可离线加载）；若无法联网，可手动将模型文件放到该目录
CACHE_DIR = os.path.join(os.path.dirname(__file__), "model_hub", "openai", "clip-vit-base-patch32")
# 根据环境自动选择设备：优先使用 CUDA，其次 CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def read_excel(mn_path):
    mn_df = pd.read_excel(mn_path)
    fire_df = mn_df[mn_df['type'] == 'fire']
    fire_test_df = mn_df[mn_df['type'] == 'fire_test']
    normal_df = mn_df[mn_df['type'] == 'normal']
    return fire_df, fire_test_df, normal_df

def read_clean_csv(csv_path):
    return pd.read_csv(csv_path)

def ensure_dir(p):
    # 确保目录存在（用于本地模型缓存）
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)
    return p

def download_pretrained(model_id=MODEL_ID, cache_dir=CACHE_DIR):
    # 在线下载预训练模型与处理器，并缓存到本地 cache_dir；首次运行需要联网
    ensure_dir(cache_dir)
    model = CLIPModel.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=False, use_safetensors=True)
    processor = CLIPProcessor.from_pretrained(model_id, cache_dir=cache_dir, local_files_only=False)
    model = model.to(DEVICE)
    return model, processor, cache_dir

def load_local_pretrained(local_dir=CACHE_DIR):
    def _find_snapshot(p):
        cand = []
        base = os.path.join(p, "models--openai--clip-vit-base-patch32", "snapshots")
        if os.path.isdir(base):
            for d in os.listdir(base):
                sd = os.path.join(base, d)
                has_cfg = os.path.isfile(os.path.join(sd, "config.json"))
                has_safe = os.path.isfile(os.path.join(sd, "model.safetensors"))
                has_bin = os.path.isfile(os.path.join(sd, "pytorch_model.bin"))
                if has_cfg and (has_safe or has_bin):
                    cand.append((sd, 0 if has_safe else 1))
        if not cand:
            return None
        cand.sort(key=lambda x: x[1])
        return cand[0][0]
    snap = _find_snapshot(local_dir)
    if snap is None:
        raise OSError("no_local_snapshot")
    model = CLIPModel.from_pretrained(snap, local_files_only=True, use_safetensors=True)
    model = model.to(DEVICE)
    processor = None
    return model, processor

def _pick_col(df, names):
    # 在给定备选列名中，选择数据表实际存在的一列
    for n in names:
        if n in df.columns:
            return n
    return None

def build_pairs(df, template):
    # 构建 (图像路径, 文本提示) 对；文本由温度填充模板生成
    ic = _pick_col(df, ["co_im", "image", "image_path", "path", "filename"])
    if ic is None:
        return []
    pairs = []
    for i in range(len(df)):
        row = df.iloc[i]
        temps = []
        for col in ['temp1', 'temp2', 'temp3']:
            if col in df.columns:
                temps.append(row[col])
        if not temps:
            continue
        t = str(max(temps))
        txt = template.format(t)
        pairs.append((str(row[ic]), txt))
    return pairs

def _map_path_and_label(co_im, image_root, t):
    if isinstance(co_im, str):
        if co_im.startswith("normal_"):
            p = os.path.join(image_root, "normal", f"{co_im}.png")
            return p, 0
        if co_im.startswith("img_"):
            p = os.path.join(image_root, "fire", "images", f"{co_im}.jpg")
            return p, 1
        if co_im.startswith("val_img_"):
            p = os.path.join(image_root, "fire_test", "images", f"{co_im}.jpg")
            return p, 1
    return None, None

def build_train_test_sets(df, image_root):
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []
    for i in range(len(df)):
        r = df.iloc[i]
        p, lab = _map_path_and_label(r.get("co_im"), image_root, r.get("type"))
        if p is None or lab is None:
            continue
        if not os.path.exists(p):
            continue
        if str(r.get("type")).lower() == "test":
            test_paths.append(p)
            test_labels.append(lab)
        elif str(r.get("type")).lower() == "train":
            train_paths.append(p)
            train_labels.append(lab)
    return train_paths, np.array(train_labels, dtype=np.int64), test_paths, np.array(test_labels, dtype=np.int64)

def encode_images_batch(model, processor, paths, batch_size=32):
    feats = []
    n = len(paths)
    for i in range(0, n, batch_size):
        batch_paths = paths[i:i+batch_size]
        images = [Image.open(p).convert("RGB") for p in batch_paths]
        if hasattr(model, "get_image_features"):
            if processor is not None:
                inputs = processor(images=images, return_tensors="pt")
                pixel_values = inputs["pixel_values"].to(DEVICE)
            else:
                pv_list = []
                for im in images:
                    w, h = im.size
                    s = 224 / float(min(w, h))
                    nw, nh = int(round(w * s)), int(round(h * s))
                    im = im.resize((nw, nh), Image.BICUBIC)
                    lx = (nw - 224) // 2
                    ty = (nh - 224) // 2
                    im = im.crop((lx, ty, lx + 224, ty + 224))
                    x = np.asarray(im).astype(np.float32) / 255.0
                    x = np.transpose(x, (2, 0, 1))
                    mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)[:, None, None]
                    std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)[:, None, None]
                    x = (x - mean) / std
                    pv_list.append(torch.from_numpy(x))
                pixel_values = torch.stack(pv_list, dim=0).to(DEVICE)
            with torch.no_grad():
                img = model.get_image_features(pixel_values=pixel_values)
        else:
            ims = [processor(im).to(DEVICE) for im in images]
            pv = torch.stack(ims, dim=0)
            with torch.no_grad():
                img = model.encode_image(pv)
        img = img / img.norm(dim=-1, keepdim=True)
        feats.append(img.detach().cpu())
    if len(feats) == 0:
        return np.zeros((0, 512), dtype=np.float32)
    return torch.cat(feats, dim=0).numpy()

def train_logreg(x, y):
    clf = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=0)
    clf.fit(x, y)
    return clf

def _save_hist(proba, out_path):
    plt.figure(figsize=(6,4))
    plt.hist(proba, bins=30, color="#4c72b0")
    plt.xlabel("prob")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _save_cm(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0,1])
    fig, ax = plt.subplots(figsize=(4,4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _save_top_grid(paths, scores, out_path, top_k=16):
    n = len(paths)
    if n == 0:
        return
    k = min(top_k, n)
    idx = np.argsort(-scores)[:k]
    cols = 4
    rows = math.ceil(k/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3, rows*3))
    axes = np.array(axes).reshape(rows, cols)
    for j in range(rows*cols):
        r = j // cols
        c = j % cols
        ax = axes[r, c]
        if j < k:
            p = paths[idx[j]]
            im = Image.open(p).convert("RGB")
            ax.imshow(im)
            ax.set_title(f"p={scores[idx[j]]:.3f}")
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def save_test_visualizations(test_paths, test_labels, proba, y_pred, base_dir):
    out_dir = os.path.join(base_dir, "vis")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    _save_hist(proba, os.path.join(out_dir, "test_proba_hist.png"))
    _save_cm(test_labels, y_pred, os.path.join(out_dir, "confusion_matrix.png"))
    _save_top_grid(test_paths, proba, os.path.join(out_dir, "top_predictions.png"), top_k=16)

def encode_similarity(model, processor, image_path, texts, image_root=None):
    # 计算单张图与多条文本的相似度：图/文分别编码，L2 归一化后以点积作为余弦相似度
    p = image_path
    if image_root and not os.path.isabs(p):
        p = os.path.join(image_root, p)
    image = Image.open(p).convert("RGB")
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        img = model.get_image_features(pixel_values=inputs["pixel_values"]) 
        txt = model.get_text_features(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]) 
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)
    sims = txt @ img.T
    return sims.detach().cpu().numpy()

def run_fusion(model, processor, fire_pairs, fire_test_pairs, normal_pairs, image_root=None):
    # 批量运行三组数据（训练火灾、测试火灾、训练无火灾）的相似度计算；返回每组相似度数组
    out = {}
    for name, pairs in [("train_fire", fire_pairs), ("test_fire", fire_test_pairs), ("train_normal", normal_pairs)]:
        sims = []
        for img, txt in pairs:
            s = encode_similarity(model, processor, img, [txt], image_root=image_root)
            sims.append(float(s[0][0]))
        out[name] = np.array(sims, dtype=np.float32)
    return out

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(project_dir, 'data', 'cleaned_dataset.csv')
    image_root = os.path.join(project_dir, 'data', 'dataset')
    try:
        model, processor, _ = download_pretrained(MODEL_ID, CACHE_DIR)
    except Exception:
        try:
            model, processor = load_local_pretrained(CACHE_DIR)
        except Exception:
            import sys
            from importlib.machinery import SourceFileLoader
            sp = os.path.join(sys.prefix, 'lib', 'site-packages', 'clip', '__init__.py')
            if os.path.isfile(sp):
                mod = SourceFileLoader('openai_clip', sp).load_module()
                model, processor = mod.load('ViT-B/32', device=DEVICE)
            else:
                raise
    df = read_clean_csv(csv_path)
    train_paths, train_labels, test_paths, test_labels = build_train_test_sets(df, image_root)
    print("train", len(train_paths), "test", len(test_paths))
    x_train = encode_images_batch(model, processor, train_paths, batch_size=32)
    x_test = encode_images_batch(model, processor, test_paths, batch_size=32)
    clf = train_logreg(x_train, train_labels)
    y_pred = clf.predict(x_test)
    proba = clf.predict_proba(x_test)[:, 1] if len(x_test) else np.array([], dtype=np.float32)
    acc = float((y_pred == test_labels).mean()) if len(test_labels) > 0 else 0.0
    print("test_accuracy", f"{acc:.4f}")
    save_test_visualizations(test_paths, test_labels, proba, y_pred, os.path.dirname(__file__))
    print("saved", os.path.join(os.path.dirname(__file__), "vis", "test_proba_hist.png"))
    print("saved", os.path.join(os.path.dirname(__file__), "vis", "confusion_matrix.png"))
    print("saved", os.path.join(os.path.dirname(__file__), "vis", "top_predictions.png"))
    return None

if __name__ == '__main__':
    main()

