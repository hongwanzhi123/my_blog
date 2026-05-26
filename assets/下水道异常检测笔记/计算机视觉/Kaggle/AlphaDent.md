# AlphaDent: Teeth marking

这是整理 AlphaDent 牙齿标注分割任务时的项目笔记。这个任务从任务角度看重点不是单纯跑通 YOLO，而是把一个真实分割比赛的流程串起来：数据定位、标签检查、yaml 构建、训练、推理、后处理和提交文件生成。

本方案先追求稳定跑通，再考虑冲分。整体思路是用 YOLO segmentation 作为第一版 baseline，先把训练和提交链路打通。

## 一、Kaggle 设置

这个任务是分割任务，在 Kaggle 右侧这样设置：

Accelerator: GPU T4 x2
Internet: On

此前遇到过 P100 和 PyTorch 版本不兼容的问题，所以这里直接选 T4 x2。

这份代码默认只用第一张 T4，也就是 device=0。这样最稳定，后面如果要加速，再考虑双卡。

## 二、完整 Notebook 代码

按 Cell 把流程拆开，方便后面单独排查每一步的问题。
```
Cell 1：安装和导入库
# =========================
# 1. Install / Import
# =========================

import os
import sys
import subprocess
from pathlib import Path
from collections import Counter
import random
import warnings

warnings.filterwarnings("ignore")

# 如果 Kaggle 环境没有 ultralytics，则自动安装
try:
    import ultralytics
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

import cv2
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO

print("Ultralytics version:", ultralytics.__version__)

# 查看 GPU
try:
    import torch
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
except Exception as e:
    print("Torch check failed:", e)
Cell 2：全局配置
# =========================
# 2. Config
# =========================

class CFG:
    # 模型选择：
    # 入门稳定：yolo11s-seg.pt
    # 更强但更吃显存：yolo11m-seg.pt / yolo11l-seg.pt / yolo11x-seg.pt
    MODEL_NAME = "yolo11s-seg.pt"

    # 牙齿病灶区域可能较小，960 通常比 640 更适合细粒度分割
    # 如果显存爆了，改成 640
    IMG_SIZE = 960

    # T4 上 yolo11s-seg + 960 可以先试 batch=4
    # 如果 OOM，改成 2 或 1
    BATCH = 4

    EPOCHS = 80
    PATIENCE = 20

    # 单卡最稳：0
    # 如果后面想尝试双卡，再改成 "0,1"；第一版先用单卡保证稳定
    DEVICE = 0

    SEED = 42
    WORK_DIR = Path("/kaggle/working")
    PROJECT_DIR = WORK_DIR / "alphadent_yolo_runs"
    RUN_NAME = f"{MODEL_NAME.replace('.pt', '')}_img{IMG_SIZE}"

    # 推理阈值，可以后续调
    CONF_THRES = 0.20
    IOU_THRES = 0.60
    MAX_DET = 300

    # 测试时增强，可能提高分数，但推理更慢
    USE_TTA = True

    # 是否简化多边形，避免 poly 太长
    SIMPLIFY_POLYGON = True
    POLY_EPS = 0.0015


random.seed(CFG.SEED)
np.random.seed(CFG.SEED)

print("Config loaded.")
Cell 3：自动定位数据集目录
# =========================
# 3. Find Dataset Root
# =========================

def find_alphadent_root():
    """
    自动寻找包含 images/train 和 labels/train 的 AlphaDent 根目录。
    常见路径：
    /kaggle/input/alpha-dent/AlphaDent
    """
    input_root = Path("/kaggle/input")

    candidates = []
    for p in input_root.rglob("images"):
        parent = p.parent
        if (parent / "images" / "train").exists() and (parent / "labels" / "train").exists():
            candidates.append(parent)

    if len(candidates) == 0:
        raise FileNotFoundError(
            "没有找到 AlphaDent 数据集目录。需要先确认已经 Add Data：alpha-dent。"
        )

    # 优先选择路径里包含 AlphaDent 的
    for c in candidates:
        if "AlphaDent" in str(c):
            return c

    return candidates[0]


DATA_ROOT = find_alphadent_root()

IMAGE_TRAIN_DIR = DATA_ROOT / "images" / "train"
LABEL_TRAIN_DIR = DATA_ROOT / "labels" / "train"
IMAGE_VALID_DIR = DATA_ROOT / "images" / "valid"
LABEL_VALID_DIR = DATA_ROOT / "labels" / "valid"
IMAGE_TEST_DIR = DATA_ROOT / "images" / "test"

print("DATA_ROOT:", DATA_ROOT)
print("Train images:", IMAGE_TRAIN_DIR)
print("Train labels:", LABEL_TRAIN_DIR)
print("Valid images:", IMAGE_VALID_DIR)
print("Test images:", IMAGE_TEST_DIR)
print("Train image count:", len(list(IMAGE_TRAIN_DIR.glob("*"))))
print("Train label count:", len(list(LABEL_TRAIN_DIR.glob("*.txt"))))
print("Valid image count:", len(list(IMAGE_VALID_DIR.glob("*"))))
print("Valid label count:", len(list(LABEL_VALID_DIR.glob("*.txt"))))
print("Test image count:", len(list(IMAGE_TEST_DIR.glob("*"))))
Cell 4：创建 YOLO 数据配置 yaml
# =========================
# 4. Build YOLO YAML
# =========================

def collect_class_ids(label_dir):
    class_ids = []
    for label_path in Path(label_dir).glob("*.txt"):
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    class_ids.append(int(float(parts[0])))
    return class_ids


# 尝试读取原始 yaml
original_yaml_candidates = list(DATA_ROOT.glob("*.yaml")) + list(DATA_ROOT.glob("*.yml"))
print("Original yaml candidates:", original_yaml_candidates)

names = None
nc = None

if len(original_yaml_candidates) > 0:
    original_yaml_path = original_yaml_candidates[0]
    print("Use original yaml:", original_yaml_path)

    with open(original_yaml_path, "r") as f:
        original_cfg = yaml.safe_load(f)

    if isinstance(original_cfg, dict):
        names = original_cfg.get("names", None)
        nc = original_cfg.get("nc", None)

# 如果原 yaml 没有 names/nc，就从标签里推断
class_ids = collect_class_ids(LABEL_TRAIN_DIR)

if nc is None:
    nc = max(class_ids) + 1 if len(class_ids) > 0 else 9

if names is None:
    names = [f"class_{i}" for i in range(nc)]

if isinstance(names, dict):
    # 有些 yaml 的 names 是 {0: xxx, 1: xxx}
    names = [names[i] for i in sorted(names.keys())]

print("nc:", nc)
print("names:", names)

# 写一个干净的 yaml 到 /kaggle/working
YOLO_YAML = CFG.WORK_DIR / "alphadent_yolo_seg.yaml"

yolo_cfg = {
    "path": str(DATA_ROOT),
    "train": "images/train",
    "val": "images/valid",
    "test": "images/test",
    "nc": int(nc),
    "names": names,
}

with open(YOLO_YAML, "w") as f:
    yaml.dump(yolo_cfg, f, sort_keys=False)

print("Created yaml:", YOLO_YAML)
print(open(YOLO_YAML).read())
Cell 5：类别分布统计
# =========================
# 5. Class Distribution
# =========================

class_counter = Counter()

for label_path in LABEL_TRAIN_DIR.glob("*.txt"):
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                class_id = int(float(parts[0]))
                class_counter[class_id] += 1

print("Class distribution:")
for k in sorted(class_counter.keys()):
    print(f"class {k}: {class_counter[k]}")

plt.figure(figsize=(10, 4))
xs = sorted(class_counter.keys())
ys = [class_counter[x] for x in xs]
plt.bar(xs, ys)
plt.xlabel("Class ID")
plt.ylabel("Instance Count")
plt.title("AlphaDent Train Class Distribution")
plt.xticks(xs)
plt.show()
Cell 6：标签可视化函数
# =========================
# 6. Visualization Utils
# =========================

def read_image_rgb(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def yolo_seg_label_to_mask(label_path, img_shape, alpha=0.45):
    """
    将 YOLO segmentation 标签转成彩色 mask。
    标签格式：
    class_id x1 y1 x2 y2 ... xn yn
    坐标是归一化 0~1。
    """
    h, w = img_shape[:2]
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    if not Path(label_path).exists():
        return mask

    with open(label_path, "r") as f:
        lines = f.readlines()

    rng = np.random.default_rng(123)

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue

        class_id = int(float(parts[0]))
        coords = np.array(list(map(float, parts[1:])), dtype=np.float32)

        if len(coords) % 2 != 0:
            coords = coords[:-1]

        points = coords.reshape(-1, 2)
        if len(points) < 3:
            continue

        points[:, 0] *= w
        points[:, 1] *= h

        points = np.clip(points, [0, 0], [w - 1, h - 1]).astype(np.int32)

        color = rng.integers(30, 255, size=3).tolist()
        cv2.fillPoly(mask, [points], color)

    return mask


def show_train_samples(num_samples=6):
    image_paths = sorted([
        p for p in IMAGE_TRAIN_DIR.glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    selected = random.sample(image_paths, min(num_samples, len(image_paths)))

    rows = len(selected)
    plt.figure(figsize=(12, 4 * rows))

    for i, img_path in enumerate(selected):
        img = read_image_rgb(img_path)
        label_path = LABEL_TRAIN_DIR / f"{img_path.stem}.txt"
        mask = yolo_seg_label_to_mask(label_path, img.shape)
        overlay = cv2.addWeighted(img, 1.0, mask, 0.45, 0)

        plt.subplot(rows, 2, 2 * i + 1)
        plt.imshow(img)
        plt.title(f"Image: {img_path.name}")
        plt.axis("off")

        plt.subplot(rows, 2, 2 * i + 2)
        plt.imshow(overlay)
        plt.title("GT Mask Overlay")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


show_train_samples(num_samples=5)
Cell 7：训练 YOLO segmentation 模型
# =========================
# 7. Train YOLO Segmentation
# =========================

model = YOLO(CFG.MODEL_NAME)

train_results = model.train(
    data=str(YOLO_YAML),
    epochs=CFG.EPOCHS,
    imgsz=CFG.IMG_SIZE,
    batch=CFG.BATCH,
    device=CFG.DEVICE,
    workers=2,
    seed=CFG.SEED,

    # 优化相关
    optimizer="AdamW",
    lr0=0.002,
    lrf=0.01,
    cos_lr=True,
    weight_decay=0.0005,
    patience=CFG.PATIENCE,

    # 分割相关
    overlap_mask=True,
    mask_ratio=4,

    # 数据增强：先用 YOLO 默认增强，不要一开始改太复杂
    close_mosaic=10,

    # 输出
    project=str(CFG.PROJECT_DIR),
    name=CFG.RUN_NAME,
    exist_ok=True,
    plots=True,
    verbose=True
)

BEST_PT = CFG.PROJECT_DIR / CFG.RUN_NAME / "weights" / "best.pt"
LAST_PT = CFG.PROJECT_DIR / CFG.RUN_NAME / "weights" / "last.pt"

print("Best model:", BEST_PT)
print("Best exists:", BEST_PT.exists())
print("Last model:", LAST_PT)
print("Last exists:", LAST_PT.exists())

如果这里显存不够，优先这样改：

IMG_SIZE: 960 → 640
BATCH: 4 → 2 → 1
MODEL_NAME: yolo11s-seg.pt 不要先换大模型
Cell 8：验证集评估
# =========================
# 8. Validation
# =========================

best_model = YOLO(str(BEST_PT))

val_metrics = best_model.val(
    data=str(YOLO_YAML),
    split="val",
    imgsz=CFG.IMG_SIZE,
    batch=CFG.BATCH,
    device=CFG.DEVICE,
    project=str(CFG.PROJECT_DIR),
    name=f"{CFG.RUN_NAME}_val",
    exist_ok=True,
    plots=True,
    verbose=True
)

print(val_metrics)

主要观察训练输出里的这些指标：

Box Precision / Recall
Mask Precision / Recall
mAP50
mAP50-95

对于这个比赛，mask 指标更重要，因为最终提交的是多边形分割结果。

Cell 9：验证集预测可视化
# =========================
# 9. Predict Valid Samples
# =========================

valid_images = sorted([
    str(p) for p in IMAGE_VALID_DIR.glob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

sample_valid_images = valid_images[:12]

pred_results = best_model.predict(
    source=sample_valid_images,
    imgsz=CFG.IMG_SIZE,
    conf=CFG.CONF_THRES,
    iou=CFG.IOU_THRES,
    device=CFG.DEVICE,
    save=True,
    project=str(CFG.PROJECT_DIR),
    name=f"{CFG.RUN_NAME}_valid_predict",
    exist_ok=True,
    verbose=False,
    retina_masks=True,
)

print("Saved valid predictions to:")
print(CFG.PROJECT_DIR / f"{CFG.RUN_NAME}_valid_predict")
Cell 10：显示验证集预测图
# =========================
# 10. Show Valid Prediction Images
# =========================

pred_dir = CFG.PROJECT_DIR / f"{CFG.RUN_NAME}_valid_predict"

pred_imgs = sorted([
    p for p in pred_dir.glob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

num_show = min(8, len(pred_imgs))

plt.figure(figsize=(16, 4 * num_show))

for i, p in enumerate(pred_imgs[:num_show]):
    img = read_image_rgb(p)
    plt.subplot(num_show, 1, i + 1)
    plt.imshow(img)
    plt.title(p.name)
    plt.axis("off")

plt.tight_layout()
plt.show()
Cell 11：寻找 sample_submission.csv
# =========================
# 11. Find Sample Submission
# =========================

sample_candidates = list(Path("/kaggle/input").rglob("*submission*.csv"))
print("Sample submission candidates:")
for p in sample_candidates:
    print(p)

SAMPLE_SUBMISSION_PATH = sample_candidates[0] if len(sample_candidates) > 0 else None

if SAMPLE_SUBMISSION_PATH is not None:
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    print("Sample submission path:", SAMPLE_SUBMISSION_PATH)
    print("Sample columns:", sample_sub.columns.tolist())
    print(sample_sub.head())
else:
    print("No sample_submission.csv found. Use default columns.")
Cell 12：多边形处理函数
# =========================
# 12. Polygon Utils
# =========================

def simplify_polygon_norm(poly, eps=0.0015):
    """
    poly: normalized polygon, shape [N, 2], x/y in [0, 1]
    使用 approxPolyDP 简化多边形，减少 submission.csv 体积。
    """
    if poly is None or len(poly) < 3:
        return None

    poly = np.asarray(poly, dtype=np.float32)
    poly = np.clip(poly, 0, 1)

    if not CFG.SIMPLIFY_POLYGON:
        return poly

    contour = poly.reshape(-1, 1, 2).astype(np.float32)
    approx = cv2.approxPolyDP(contour, epsilon=eps, closed=True)
    approx = approx.reshape(-1, 2)

    if len(approx) < 3:
        return poly

    return approx


def polygon_to_string(poly):
    """
    将 [N, 2] 的归一化 polygon 转成：
    x1 y1 x2 y2 ... xn yn
    """
    poly = np.asarray(poly, dtype=np.float32)
    poly = np.clip(poly, 0, 1)

    return " ".join([f"{x:.6f} {y:.6f}" for x, y in poly])
Cell 13：测试集全量推理并生成 submission.csv
# =========================
# 13. Full Test Inference + Submission
# =========================

test_images = sorted([
    str(p) for p in IMAGE_TEST_DIR.glob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

print("Number of test images:", len(test_images))
print("First 5 test images:", test_images[:5])

# 推理
test_results = best_model.predict(
    source=test_images,
    imgsz=CFG.IMG_SIZE,
    conf=CFG.CONF_THRES,
    iou=CFG.IOU_THRES,
    max_det=CFG.MAX_DET,
    device=CFG.DEVICE,
    save=False,
    stream=True,
    verbose=False,
    retina_masks=True,
    augment=CFG.USE_TTA,
)

rows = []

for r in test_results:
    patient_id = Path(r.path).stem

    # 没有预测结果，直接跳过
    if r.boxes is None or r.masks is None:
        continue

    if len(r.boxes) == 0:
        continue

    classes = r.boxes.cls.detach().cpu().numpy().astype(int)
    confs = r.boxes.conf.detach().cpu().numpy()

    # r.masks.xyn 是归一化多边形坐标，适合提交
    polygons = r.masks.xyn

    for class_id, conf, poly in zip(classes, confs, polygons):
        if poly is None or len(poly) < 3:
            continue

        poly = simplify_polygon_norm(poly, eps=CFG.POLY_EPS)

        if poly is None or len(poly) < 3:
            continue

        poly_str = polygon_to_string(poly)

        rows.append({
            "patient_id": patient_id,
            "class_id": int(class_id),
            "confidence": float(conf),
            "poly": poly_str,
        })

submission = pd.DataFrame(
    rows,
    columns=["patient_id", "class_id", "confidence", "poly"]
)

SUBMISSION_PATH = CFG.WORK_DIR / "submission.csv"
submission.to_csv(SUBMISSION_PATH, index=False)

print("Submission saved to:", SUBMISSION_PATH)
print("Submission shape:", submission.shape)
submission.head()
Cell 14：提交文件格式检查
# =========================
# 14. Check Submission
# =========================

sub = pd.read_csv(SUBMISSION_PATH)

print("Shape:", sub.shape)
print("Columns:", sub.columns.tolist())
print("Null count:")
print(sub.isnull().sum())

print("Class distribution in submission:")
print(sub["class_id"].value_counts().sort_index())

print("Confidence describe:")
print(sub["confidence"].describe())

print("First rows:")
display(sub.head())

# 基础断言
assert list(sub.columns) == ["patient_id", "class_id", "confidence", "poly"]
assert sub["patient_id"].notna().all()
assert sub["class_id"].notna().all()
assert sub["confidence"].notna().all()
assert sub["poly"].notna().all()

print("Submission check passed.")
Cell 15：可选，测试集预测图可视化
# =========================
# 15. Optional: Save and Show Some Test Predictions
# =========================

sample_test_images = test_images[:12]

_ = best_model.predict(
    source=sample_test_images,
    imgsz=CFG.IMG_SIZE,
    conf=CFG.CONF_THRES,
    iou=CFG.IOU_THRES,
    device=CFG.DEVICE,
    save=True,
    project=str(CFG.PROJECT_DIR),
    name=f"{CFG.RUN_NAME}_test_preview",
    exist_ok=True,
    verbose=False,
    retina_masks=True,
    augment=False,
)

preview_dir = CFG.PROJECT_DIR / f"{CFG.RUN_NAME}_test_preview"

preview_imgs = sorted([
    p for p in preview_dir.glob("*")
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
])

num_show = min(8, len(preview_imgs))

plt.figure(figsize=(16, 4 * num_show))

for i, p in enumerate(preview_imgs[:num_show]):
    img = read_image_rgb(p)
    plt.subplot(num_show, 1, i + 1)
    plt.imshow(img)
    plt.title(p.name)
    plt.axis("off")

plt.tight_layout()
plt.show()
```
## 三、这份代码的运行顺序

Notebook 新建后，按顺序运行：

Cell 1：安装和导入库
Cell 2：配置参数
Cell 3：定位数据集
Cell 4：生成 yaml
Cell 5：类别分布
Cell 6：标签可视化
Cell 7：训练模型
Cell 8：验证评估
Cell 9-10：验证预测可视化
Cell 11：查看 sample_submission
Cell 12：多边形处理
Cell 13：测试集推理并生成 submission.csv
Cell 14：检查提交文件
Cell 15：可选测试集可视化

最终提交文件会在：

/kaggle/working/submission.csv

Kaggle 页面右侧 Output 里可以看到它，然后提交。

## 四、最可能遇到的问题和解决方法
1. 显存爆了

报错类似：

CUDA out of memory

按这个顺序改：

CFG.BATCH = 2

还不行：

CFG.BATCH = 1

再不行：

CFG.IMG_SIZE = 640

不要一开始就用 yolo11x-seg.pt。先用：

CFG.MODEL_NAME = "yolo11s-seg.pt"
2. 模型下载失败

如果 YOLO("yolo11s-seg.pt") 下载失败，说明 Kaggle Internet 没开。

解决：

右侧 Notebook Settings
Internet: On

如果比赛最终提交要求关闭 Internet，可以先训练好模型，把 best.pt 保存成 Kaggle Dataset，再在最终推理 Notebook 里从 Dataset 加载权重。

3. submission.csv 是空的

如果输出：

Submission shape: (0, 4)

说明模型没有预测出任何 mask。可能原因：

conf 太高
模型没训练好
路径不对
权重没加载正确
测试集不是预期图片

先把阈值调低：

CFG.CONF_THRES = 0.05

然后重新运行 Cell 13。

4. 提交后格式错误

优先检查：

sub.head()
sub.columns

比赛要求一般是：

patient_id,class_id,confidence,poly

如果官方 sample_submission.csv 的列名和这里不同，就以官方为准，把 Cell 13 的列名改掉。

## 五、后续实验计划

第一版先跑通 yolo11s-seg + imgsz=960 + conf=0.20 + iou=0.60 + TTA。

跑通以后可以继续做这些实验：

1. imgsz=640 vs 960
2. conf=0.10 / 0.15 / 0.20 / 0.25 / 0.30
3. iou=0.50 / 0.60 / 0.70
4. TTA=False vs True
5. yolo11s-seg vs yolo11m-seg
6. epochs=80 vs 120
7. 不简化 polygon vs 简化 polygon

每次只改一个参数，记录：

本地验证 mAP
提交分数
推理时间
submission.csv 大小

## 六、项目总结

这次 AlphaDent 将它当成一个完整的实例分割项目来整理。相比手写数字识别，这个任务更接近真实工程：数据路径更复杂，标签格式需要检查，模型训练会受到显存限制，最终提交还要处理 mask 多边形格式。

当前策略是先用 yolo11s-seg 跑通 baseline，再围绕输入尺寸、置信度阈值、TTA、模型大小和 polygon 后处理做单变量实验。这样每次分数变化都能对应到一个明确改动，后续总结也更清楚。
