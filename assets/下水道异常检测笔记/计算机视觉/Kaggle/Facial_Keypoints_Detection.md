# Facial Keypoints Detection

这个比赛本质是一个 人脸关键点回归任务。

## 一、任务解释：Facial Keypoints Detection 是什么？

这个 Kaggle 比赛给你一张 96×96 的灰度人脸图像，让模型预测人脸上的关键点坐标。

每张图需要预测 15 个关键点，每个关键点有 x 和 y 两个坐标，所以一共输出：

15 个关键点 × 2 = 30 个数值

例如：

left_eye_center_x
left_eye_center_y
right_eye_center_x
right_eye_center_y
nose_tip_x
nose_tip_y
mouth_left_corner_x
mouth_left_corner_y
...

输入是图片，输出是坐标，所以它不是分类任务，而是：

图像回归任务

也可以理解为：

输入：96×96 人脸灰度图
输出：30 个关键点坐标
## 二、这个任务的难点

这个比赛有几个典型难点：

1. 标签有缺失值
2. 输出不是类别，而是连续坐标
3. 图片较小，只有 96×96
4. 人脸姿态、表情、光照不同
5. 左右关键点需要注意翻转增强
6. 提交格式不是简单预测表，需要用 IdLookupTable 映射

其中最重要的是：

训练集中很多关键点坐标是缺失的，所以 loss 不能直接对所有 30 个坐标计算，否则 NaN 会导致训练失败。

所以我们要设计一个：

Masked Loss

只对存在标签的关键点计算损失。

## 三、整体解决思路

我们的解决流程是：

1. 读取 training.csv / test.csv / IdLookupTable.csv
2. 解析 Image 字符串为 96×96 灰度图
3. 提取 30 个关键点坐标作为标签
4. 对坐标进行归一化：0~96 → 0~1
5. 构建 Dataset 和 DataLoader
6. 使用 CNN 回归 30 个关键点坐标
7. 使用 Masked SmoothL1Loss 处理缺失标签
8. 训练并验证模型
9. 对 test.csv 预测坐标
10. 根据 IdLookupTable 生成 Kaggle submission.csv
## 四、完整 Kaggle 代码
```python
1. 导入库
import os
import random
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
2. 固定随机种子
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = True


seed_everything(42)

作用：

尽量保证每次训练结果稳定
3. 配置参数
class CFG:
    data_dir = Path("/kaggle/input/facial-keypoints-detection")

    img_size = 96
    num_keypoint_values = 30

    seed = 42
    valid_size = 0.15

    batch_size = 128
    num_epochs = 60
    learning_rate = 1e-3
    weight_decay = 1e-4

    best_model_path = "/kaggle/working/best_facial_keypoint_cnn.pth"
    submission_path = "/kaggle/working/submission.csv"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
4. 读取 Kaggle 数据

这个比赛的数据有时候是 .zip 文件，所以这里写一个兼容读取函数。

def read_competition_csv(csv_name, zip_name=None):
    csv_path = CFG.data_dir / csv_name

    if csv_path.exists():
        return pd.read_csv(csv_path)

    if zip_name is None:
        zip_name = csv_name.replace(".csv", ".zip")

    zip_path = CFG.data_dir / zip_name

    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as z:
            names = z.namelist()
            target_name = None

            for name in names:
                if name.endswith(csv_name):
                    target_name = name
                    break

            if target_name is None:
                target_name = names[0]

            with z.open(target_name) as f:
                return pd.read_csv(f)

    raise FileNotFoundError(f"Cannot find {csv_name} or {zip_name}")


train_df = read_competition_csv("training.csv", "training.zip")
test_df = read_competition_csv("test.csv", "test.zip")
lookup_df = read_competition_csv("IdLookupTable.csv")
sample_submission = read_competition_csv("SampleSubmission.csv")

print("train_df:", train_df.shape)
print("test_df:", test_df.shape)
print("lookup_df:", lookup_df.shape)

train_df.head()
五、理解数据格式

运行下面代码查看列名：

print(train_df.columns.tolist())

你会看到：

left_eye_center_x
left_eye_center_y
right_eye_center_x
right_eye_center_y
...
Image

其中：

Image：一张 96×96 图像，被存成一个很长的字符串
其他 30 列：关键点坐标

例如 Image 里面是：

238 236 237 238 240 ...

我们需要把它转成：

[96, 96]

的二维图像。

6. 解析图片
target_cols = [col for col in train_df.columns if col != "Image"]

print("Number of target columns:", len(target_cols))
print(target_cols)
def parse_images(image_series):
    images = []

    for img_str in image_series:
        img = np.fromstring(img_str, sep=" ", dtype=np.float32)
        img = img.reshape(CFG.img_size, CFG.img_size)
        images.append(img)

    images = np.stack(images)
    images = images / 255.0

    return images.astype(np.float32)


X_all = parse_images(train_df["Image"])
X_test = parse_images(test_df["Image"])

print("X_all:", X_all.shape)
print("X_test:", X_test.shape)

输出应该类似：

X_all: (7049, 96, 96)
X_test: (1783, 96, 96)
7. 处理标签和缺失值

重点来了。

训练标签里有 NaN，所以我们需要两个东西：

y_all：标签值，NaN 先填成 0
mask_all：是否有真实标签，有标签为 1，缺失为 0
y_raw = train_df[target_cols].values.astype(np.float32)

mask_all = ~np.isnan(y_raw)

# NaN 先填 0，后面通过 mask 避免参与 loss
y_all = np.nan_to_num(y_raw, nan=0.0)

# 坐标归一化：0~96 -> 0~1
y_all = y_all / CFG.img_size

mask_all = mask_all.astype(np.float32)

print("y_all:", y_all.shape)
print("mask_all:", mask_all.shape)
print("Missing label ratio:", 1.0 - mask_all.mean())

为什么要归一化？

原始坐标范围大概是：

0 ~ 96

归一化后变成：

0 ~ 1

这样模型更容易训练。

8. 可视化一张图片和关键点
def show_image_with_keypoints(image, targets=None, title=""):
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray")

    if targets is not None:
        points = targets.reshape(-1, 2) * CFG.img_size
        for x, y in points:
            if x > 0 and y > 0:
                plt.scatter(x, y, c="red", s=12)

    plt.title(title)
    plt.axis("off")
    plt.show()


show_image_with_keypoints(X_all[0], y_all[0], title="Training Sample")
六、划分训练集和验证集
indices = np.arange(len(X_all))

train_idx, valid_idx = train_test_split(
    indices,
    test_size=CFG.valid_size,
    random_state=CFG.seed
)

X_train = X_all[train_idx]
y_train = y_all[train_idx]
mask_train = mask_all[train_idx]

X_valid = X_all[valid_idx]
y_valid = y_all[valid_idx]
mask_valid = mask_all[valid_idx]

print("X_train:", X_train.shape)
print("X_valid:", X_valid.shape)
七、构建 Dataset

这里我们加入一个非常重要的数据增强：

水平翻转 Horizontal Flip

人脸左右翻转后：

x 坐标需要变成 1 - x
左眼要和右眼交换
左嘴角要和右嘴角交换
1. 构建左右关键点交换表
name_to_idx = {name: i for i, name in enumerate(target_cols)}

swap_pairs = []

for name in target_cols:
    if "left" in name:
        right_name = name.replace("left", "right")
        if right_name in name_to_idx:
            i = name_to_idx[name]
            j = name_to_idx[right_name]
            swap_pairs.append((i, j))

x_indices = [i for i, name in enumerate(target_cols) if name.endswith("_x")]

print("swap_pairs:")
for i, j in swap_pairs:
    print(target_cols[i], "<->", target_cols[j])

print("x_indices:", x_indices)
2. Dataset 类
class FacialKeypointsDataset(Dataset):
    def __init__(
        self,
        images,
        targets=None,
        masks=None,
        augment=False,
        target_cols=None,
        swap_pairs=None,
        x_indices=None
    ):
        self.images = images
        self.targets = targets
        self.masks = masks
        self.augment = augment
        self.target_cols = target_cols
        self.swap_pairs = swap_pairs or []
        self.x_indices = x_indices or []

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)

        if self.targets is None:
            return image

        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        mask = torch.tensor(self.masks[idx], dtype=torch.float32)

        if self.augment:
            # 水平翻转增强
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])

                # x 坐标翻转
                target[self.x_indices] = 1.0 - target[self.x_indices]

                # 左右关键点交换
                for i, j in self.swap_pairs:
                    target_i = target[i].clone()
                    target_j = target[j].clone()
                    mask_i = mask[i].clone()
                    mask_j = mask[j].clone()

                    target[i] = target_j
                    target[j] = target_i
                    mask[i] = mask_j
                    mask[j] = mask_i

        return image, target, mask
3. DataLoader
train_dataset = FacialKeypointsDataset(
    X_train,
    y_train,
    mask_train,
    augment=True,
    target_cols=target_cols,
    swap_pairs=swap_pairs,
    x_indices=x_indices
)

valid_dataset = FacialKeypointsDataset(
    X_valid,
    y_valid,
    mask_valid,
    augment=False,
    target_cols=target_cols,
    swap_pairs=swap_pairs,
    x_indices=x_indices
)

test_dataset = FacialKeypointsDataset(
    X_test,
    targets=None,
    masks=None,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=(device.type == "cuda")
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda")
)

test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=(device.type == "cuda")
)
八、构建 CNN 回归模型

这个模型输入：

[B, 1, 96, 96]

输出：

[B, 30]

每个输出是归一化坐标，范围在 0~1。

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class FacialKeypointCNN(nn.Module):
    def __init__(self, num_outputs=30):
        super().__init__()

        self.features = nn.Sequential(
            ConvBlock(1, 32),
            ConvBlock(32, 32),
            nn.MaxPool2d(2),          # 96 -> 48
            nn.Dropout2d(0.05),

            ConvBlock(32, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2),          # 48 -> 24
            nn.Dropout2d(0.10),

            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2),          # 24 -> 12
            nn.Dropout2d(0.15),

            ConvBlock(128, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2),          # 12 -> 6
            nn.Dropout2d(0.20),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 6 * 6, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),

            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),

            nn.Linear(256, num_outputs),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x
初始化模型
model = FacialKeypointCNN(num_outputs=CFG.num_keypoint_values).to(device)

print(model)
九、Masked Loss：处理缺失标签

因为标签有缺失值，所以不能直接：

loss = MSE(pred, target)

我们要写：

只对 mask = 1 的位置计算 loss

这里使用 SmoothL1Loss，它比 MSE 对异常值更稳一点。

class MaskedSmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target, mask):
        loss = F.smooth_l1_loss(pred, target, reduction="none")
        loss = loss * mask

        return loss.sum() / mask.sum().clamp_min(1.0)


criterion = MaskedSmoothL1Loss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CFG.learning_rate,
    weight_decay=CFG.weight_decay
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=CFG.num_epochs
)
十、训练和验证函数
1. 训练一个 epoch
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_count = 0

    for images, targets, masks in loader:
        images = images.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        preds = model(images)

        loss = criterion(preds, targets, masks)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_count += batch_size

    return total_loss / total_count
2. 验证函数

这里我们同时计算：

验证 loss
验证 RMSE 像素误差

RMSE 是 Kaggle 这个任务更直观的指标。

def validate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_count = 0

    sq_error_sum = 0.0
    mask_sum = 0.0

    with torch.no_grad():
        for images, targets, masks in loader:
            images = images.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            preds = model(images)

            loss = criterion(preds, targets, masks)

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_count += batch_size

            sq_error = ((preds - targets) ** 2) * masks
            sq_error_sum += sq_error.sum().item()
            mask_sum += masks.sum().item()

    valid_loss = total_loss / total_count

    # 坐标已经归一化，因此乘 96 转回像素误差
    rmse_pixel = np.sqrt(sq_error_sum / max(mask_sum, 1.0)) * CFG.img_size

    return valid_loss, rmse_pixel
十一、开始训练
best_rmse = float("inf")

train_losses = []
valid_losses = []
valid_rmses = []

for epoch in range(1, CFG.num_epochs + 1):
    train_loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    valid_loss, valid_rmse = validate(
        model,
        valid_loader,
        criterion,
        device
    )

    scheduler.step()

    train_losses.append(train_loss)
    valid_losses.append(valid_loss)
    valid_rmses.append(valid_rmse)

    if valid_rmse < best_rmse:
        best_rmse = valid_rmse
        torch.save(model.state_dict(), CFG.best_model_path)

    print(
        f"Epoch [{epoch:03d}/{CFG.num_epochs}] "
        f"Train Loss: {train_loss:.5f} "
        f"Valid Loss: {valid_loss:.5f} "
        f"Valid RMSE(px): {valid_rmse:.4f} "
        f"Best RMSE(px): {best_rmse:.4f}"
    )

print("Best validation RMSE:", best_rmse)
十二、绘制训练曲线
plt.figure(figsize=(10, 4))
plt.plot(train_losses, label="Train Loss")
plt.plot(valid_losses, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(valid_rmses, label="Valid RMSE(px)")
plt.xlabel("Epoch")
plt.ylabel("RMSE in pixels")
plt.title("Validation RMSE")
plt.legend()
plt.show()
十三、验证集预测可视化
model.load_state_dict(torch.load(CFG.best_model_path, map_location=device))
model.eval()

def show_valid_prediction(index=0):
    image = X_valid[index]
    target = y_valid[index]

    x = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")

    target_points = target.reshape(-1, 2) * CFG.img_size
    pred_points = pred.reshape(-1, 2) * CFG.img_size

    for x_gt, y_gt in target_points:
        if x_gt > 0 and y_gt > 0:
            plt.scatter(x_gt, y_gt, c="lime", s=18, label="GT")

    for x_pred, y_pred in pred_points:
        plt.scatter(x_pred, y_pred, c="red", s=12, label="Pred")

    plt.title("Green: Ground Truth, Red: Prediction")
    plt.axis("off")
    plt.show()


show_valid_prediction(0)
show_valid_prediction(1)
show_valid_prediction(2)
十四、预测测试集
model.load_state_dict(torch.load(CFG.best_model_path, map_location=device))
model.to(device)
model.eval()

test_preds = []

with torch.no_grad():
    for images in test_loader:
        images = images.to(device)

        preds = model(images)
        preds = preds.cpu().numpy()

        test_preds.append(preds)

test_preds = np.concatenate(test_preds, axis=0)

# 0~1 -> 0~96
test_preds = test_preds * CFG.img_size

# 坐标裁剪到合理范围
test_preds = np.clip(test_preds, 0, CFG.img_size)

print("test_preds:", test_preds.shape)
print(test_preds[:2])
十五、生成 Kaggle 提交文件

这个比赛的提交格式比较特殊。

不能直接提交：

ImageId, left_eye_center_x, ...

而是要根据 IdLookupTable.csv 生成：

RowId, Location

IdLookupTable.csv 会告诉你：

第几张测试图
需要提交哪个关键点
对应的 RowId 是多少

所以我们要查表。

pred_df = pd.DataFrame(test_preds, columns=target_cols)

locations = []

for _, row in lookup_df.iterrows():
    image_id = int(row["ImageId"])
    feature_name = row["FeatureName"]

    # ImageId 是从 1 开始，DataFrame index 从 0 开始
    pred_value = pred_df.loc[image_id - 1, feature_name]

    locations.append(pred_value)

submission = pd.DataFrame({
    "RowId": lookup_df["RowId"],
    "Location": locations
})

submission.to_csv(CFG.submission_path, index=False)

print("Saved submission to:", CFG.submission_path)
print(submission.shape)
submission.head()
检查提交文件
sub = pd.read_csv(CFG.submission_path)

print(sub.head())
print(sub.tail())
print(sub.isnull().sum())
print(sub["Location"].describe())

右侧 Output 里应该出现：

/kaggle/working/submission.csv

提交这个文件即可。
```
## 十六、核心知识点笔记
### 1. 这是分类任务吗？

不是。

这是：

关键点坐标回归任务

分类任务输出类别：

猫 / 狗 / 汽车

这个任务输出连续坐标：

left_eye_center_x = 66.03
left_eye_center_y = 39.00

所以最后一层不是 softmax，而是输出 30 个数值。

### 2. 为什么最后用 Sigmoid？

因为我们把坐标归一化到了：

0 ~ 1

所以模型最后用：

nn.Sigmoid()

让输出自然落在 0~1 区间。

最后提交前再乘以：

96

恢复到原图坐标。

### 3. 为什么要用 Masked Loss？

因为训练集中有很多关键点缺失。

例如某一张图片可能只有：

left_eye_center
right_eye_center
nose_tip

其他点是 NaN。

如果直接计算所有点 loss：

NaN 会传播，训练失败

所以我们构造 mask：

有标签的位置：1
缺失的位置：0

loss 只在 mask=1 的位置计算。

### 4. 为什么要做水平翻转增强？

人脸左右基本对称。

水平翻转后可以得到更多训练样本。

但关键点也要同步变换：

x 坐标：x -> 1 - x
left_eye -> right_eye
left_mouth_corner -> right_mouth_corner

如果只翻转图片，不交换标签，模型会学错。

### 5. 为什么不用 CrossEntropyLoss？

因为 CrossEntropyLoss 用于分类。

这个任务输出的是连续坐标，所以用：

MSELoss
SmoothL1Loss
L1Loss

更合适。

这里用 SmoothL1Loss，它对异常标签更稳。

十七、模型结构总结

这份 CNN 模型可以理解为：

输入 96×96 灰度图
↓
多层卷积提取局部特征
↓
池化逐步扩大感受野
↓
全连接层融合全局人脸结构
↓
输出 30 个关键点坐标

卷积层负责提取：

眼睛边缘
鼻子轮廓
嘴角形状
眉毛纹理
脸部结构

全连接层负责根据整体人脸结构回归关键点位置。

## 十八、如何继续提升分数？

如果你想进一步提升，可以做这些优化：

### 1. 使用更强模型

可以换成：

ResNet18
EfficientNet-B0
MobileNetV3
HRNet
UNet heatmap regression

对于关键点检测，heatmap 方法通常比直接坐标回归更强。

### 2. 两阶段训练

第一阶段：

使用所有有标签样本，训练基础模型

第二阶段：

只使用完整标签较多的样本，微调模型
### 3. 使用 heatmap 监督

直接回归坐标是：

图片 → 30 个数

Heatmap 方法是：

图片 → 每个关键点一张热力图

例如：

15 个关键点 → 15 张 96×96 heatmap

每个 heatmap 上关键点位置是一个高斯峰。

这种方法更符合关键点检测任务。

### 4. 更丰富的数据增强

可以加入：

随机旋转
随机平移
随机缩放
亮度变化
对比度变化
随机噪声

但要注意：

几何增强必须同步变换关键点坐标。

### 5. 缺失标签填补

可以尝试用已有标签训练一个模型，然后预测缺失标签，构造伪标签，再继续训练。

这属于：

Pseudo Labeling

## 十九、项目总结写法

你可以把这个 Kaggle 项目总结成：

基于 Kaggle Facial Keypoints Detection 数据集完成 人脸关键点检测任务。该任务输入 96×96 灰度人脸图像，输出 15 个关键点共 30 个坐标值，属于图像坐标回归问题。项目中首先将图像字符串解析为灰度矩阵，并对关键点坐标进行 0~1 归一化；针对训练标签中存在大量缺失值的问题，设计 Masked SmoothL1Loss，仅对有效标签位置计算损失；同时加入水平翻转增强，并同步完成 x 坐标变换和左右关键点交换。模型方面构建 CNN 回归网络，通过多层卷积提取眼睛、鼻子、嘴角等局部结构特征，最终回归 30 个关键点坐标。最后根据 IdLookupTable 生成 Kaggle 要求的 RowId-Location 提交文件。