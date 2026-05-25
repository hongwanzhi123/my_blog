# Mask2Former

Mask2Former。它是近几年图像分割领域非常重要的 Transformer 分割框架，核心特点是：

用一个统一的 mask classification 框架，同时处理语义分割、实例分割和全景分割。

## 一、Mask2Former 是什么
Mask2Former 全称是：

Masked-attention Mask Transformer

它来自论文 《Masked-attention Mask Transformer for Universal Image Segmentation》，主要目标是统一三类分割任务：

1. Semantic Segmentation：语义分割
2. Instance Segmentation：实例分割
3. Panoptic Segmentation：全景分割

传统分割模型往往是一个任务对应一套架构：

语义分割：DeepLab / U-Net / SegFormer
实例分割：Mask R-CNN / SOLO / YOLACT
全景分割：Panoptic FPN / UPSNet

Mask2Former 的思想是：

不再为不同分割任务设计完全不同的模型，而是把它们统一成“预测一组 mask + 每个 mask 的类别”的问题。

论文摘要中也明确指出，Mask2Former 是一个可以处理 panoptic、instance、semantic segmentation 的统一架构，并通过 masked attention 限制 cross-attention 在预测 mask 区域内提取局部特征。

## 二、先理解三种分割任务
### 1. 语义分割 Semantic Segmentation

语义分割是：

每个像素预测一个类别

例如：

道路 / 天空 / 建筑 / 行人 / 车辆

它不区分同类不同实例。

如果图中有 3 个人，语义分割只会输出：

这些像素都是 person

不会区分：

person_1
person_2
person_3
### 2. 实例分割 Instance Segmentation

实例分割要区分每个独立目标。

例如图中有 3 个人：

person_1 mask
person_2 mask
person_3 mask

它不仅要知道类别，还要知道每个实例的独立轮廓。

### 3. 全景分割 Panoptic Segmentation

全景分割结合了语义分割和实例分割。

它既要分割不可数背景类：

sky
road
grass
wall

也要分割可数目标类：

person_1
person_2
car_1
car_2

通常把类别分为两种：

Stuff：不可数背景区域，如天空、道路、草地
Things：可数目标实例，如人、车、狗

Mask2Former 的优势是：它用同一个模型框架同时支持这三种任务，而不是分别设计三套模型。其官方仓库也说明它是一个支持 panoptic、instance、semantic segmentation 的单一架构，并支持 ADE20K、Cityscapes、COCO、Mapillary Vistas 等主流分割数据集。

## 三、Mask2Former 的核心思想：Mask Classification

传统语义分割通常是 per-pixel classification：

每个像素单独分类

比如输出：

[B, C, H, W]

其中每个像素都有 C 个类别分数。

Mask2Former 采用的是 mask classification：

预测 N 个 mask
每个 mask 再预测一个类别

也就是说，模型不是直接问：

这个像素是什么类别？

而是问：

图中有哪些区域 mask？
每个区域属于什么类别？

输出形式类似：

Query 1 → class = person, mask = 一个人的区域
Query 2 → class = car,    mask = 一辆车的区域
Query 3 → class = road,   mask = 道路区域
Query 4 → class = sky,    mask = 天空区域
...

这就是它能统一语义、实例、全景分割的关键。

## 四、为什么 mask classification 能统一三种任务？

因为三种分割任务本质上都可以看成：

预测若干个区域 mask
再给每个 mask 一个类别

区别只在后处理方式不同。

### 1. 用于语义分割

语义分割不区分实例，所以多个同类别 mask 可以合并。

例如：

mask_1: person
mask_2: person
mask_3: road
mask_4: sky

最终语义图中：

所有 person mask 合并为 person 类
road mask 作为 road 类
sky mask 作为 sky 类
### 2. 用于实例分割

实例分割只关心 things 类，也就是可数目标。

例如：

person_1
person_2
car_1
dog_1

每个 query 输出一个实例 mask。

### 3. 用于全景分割

全景分割同时处理 things 和 stuff：

things：每个实例单独保留
stuff：同类区域可以合并

例如：

person_1 mask
person_2 mask
car_1 mask
road mask
sky mask
building mask

这样就可以得到全景分割结果。

所以 Mask2Former 的统一性来自于：

所有任务都转化为 mask classification

## 五、Mask2Former 和 MaskFormer 的关系

Mask2Former 是在 MaskFormer 基础上的改进。

MaskFormer 已经提出了 mask classification 的统一分割思想。

但是 MaskFormer 还有一些问题：

1. Transformer decoder 的 cross-attention 仍然会关注整张图
2. 计算量和注意力范围较大
3. 小目标和局部区域建模还可以进一步增强

Mask2Former 的核心改进是：

Masked Attention

也就是：

让 query 在 cross-attention 时主要关注自己当前预测 mask 覆盖的区域，而不是整张图。

这使得每个 query 更专注于自己负责的目标或区域。

## 六、Mask2Former 的整体结构

Mask2Former 可以分成四个主要部分：

1. Backbone
2. Pixel Decoder
3. Transformer Decoder
4. Mask Classification Head

整体流程：

输入图像
↓
Backbone 提取多尺度特征
↓
Pixel Decoder 生成高分辨率像素特征
↓
Transformer Decoder 使用 object queries 逐层预测 mask
↓
Masked Attention 限制注意力区域
↓
输出每个 query 的 class 和 mask
↓
根据任务类型生成 semantic / instance / panoptic 结果

可以简化为：

Image
↓
Backbone
↓
Pixel Decoder
↓
Transformer Decoder with Masked Attention
↓
Class Predictions + Mask Predictions

## 七、Backbone：特征提取网络

Backbone 的作用是从原图中提取多尺度视觉特征。

常见 backbone 包括：

ResNet
Swin Transformer
ConvNeXt

Backbone 输出多个尺度的特征图，例如：

C2: H/4  × W/4
C3: H/8  × W/8
C4: H/16 × W/16
C5: H/32 × W/32

不同尺度特征的作用不同：

浅层特征：分辨率高，边界和细节更多
深层特征：语义更强，适合识别类别和整体结构

Mask2Former 会利用这些多尺度特征，为后续 mask 预测提供基础。

## 八、Pixel Decoder：像素解码器

Pixel Decoder 的作用是：

把 backbone 的多尺度特征融合成适合像素级 mask 预测的特征

它需要输出两类东西：

1. 给 Transformer decoder 用的多尺度特征
2. 最终生成 mask 的高分辨率 pixel embedding

可以理解为：

Backbone 提取的是图像特征
Pixel Decoder 把这些特征整理成适合分割的像素表示

Mask2Former 中的 pixel decoder 通常会使用多尺度特征融合，并借鉴 Deformable Attention 的思想来提升效率。

最终会得到一个像素级特征图：

pixel embedding: [B, C, H', W']

后面每个 query 预测出来的 mask embedding 会和这个 pixel embedding 做相似度计算，从而得到 mask。

## 九、Object Query 是什么？

Mask2Former 和 DETR、MaskFormer 类似，也使用一组 queries。

可以把 query 理解为：

模型预留的一组“区域槽位”

例如设置：

num_queries = 100

那么模型会有 100 个 query：

query_1
query_2
query_3
...
query_100

每个 query 最终输出：

1. 一个类别预测
2. 一个 mask 预测

比如：

query_3  → person_1 mask
query_17 → car_1 mask
query_25 → road mask
query_41 → sky mask
其他 query → no object

这和 DETR 中 object query 预测 bbox 很像，只不过 Mask2Former 预测的是 mask。

## 十、Transformer Decoder 的作用

Transformer Decoder 的任务是：

让 queries 从图像特征中提取自己需要的信息，并逐层优化 mask 和类别预测。

每一层 decoder 大致做：

query self-attention
↓
masked cross-attention
↓
feed-forward network
↓
更新 query 表示
↓
预测 class 和 mask

其中最关键的是：

masked cross-attention

这也是 Mask2Former 和普通 Transformer decoder 的重要区别。

## 十一、Masked Attention 是什么？

普通 cross-attention 是：

每个 query 可以关注整张图的所有位置

问题是：

1. 注意力范围太大
2. query 可能关注很多无关区域
3. 对局部目标建模不够专注
4. 计算和学习难度更高

Mask2Former 的 masked attention 是：

每个 query 只在自己预测的 mask 区域内做 cross-attention

也就是说，如果某个 query 当前预测的是一辆车的 mask，那么下一层 decoder 中，这个 query 主要关注这辆车所在区域，而不是整张图。

可以理解为：

第一层：query 粗略找到一个区域
第二层：只在这个区域里继续看
第三层：进一步细化这个区域
第四层：得到更准确的 mask

论文中把 masked attention 描述为通过将 cross-attention 限制在预测 mask 区域内来提取局部特征，这是 Mask2Former 的关键组件之一。

## 十二、Masked Attention 的直观理解

假设图中有：

人、车、道路、天空

普通 attention 中，一个预测“人”的 query 可能还会看道路、天空、车辆等无关区域。

Masked attention 中：

person query 主要看 person 区域
car query 主要看 car 区域
road query 主要看 road 区域
sky query 主要看 sky 区域

这样有几个好处：

1. query 更专注
2. 目标区域特征更纯
3. mask 逐层细化更容易
4. 对小目标和边界更友好
5. 计算更高效

可以把它理解为一种“先粗定位，再局部精修”的注意力机制。

## 十三、Mask Prediction 是怎么做的？

Mask2Former 的 mask 预测通常不是直接输出一张完整 mask，而是通过：

query mask embedding
和
pixel embedding
做点积

假设：

query embedding: [B, N, C]
pixel embedding: [B, C, H, W]

那么每个 query 的 mask 可以通过类似下面的方式得到：

mask_i = query_i · pixel_embedding

输出：

[B, N, H, W]

其中：

N = query 数量
H, W = mask 分辨率

每个 query 对应一张 mask。

再通过 sigmoid 得到每个像素属于该 mask 的概率：

mask_prob = sigmoid(mask_logits)

## 十四、Class Prediction 是怎么做的？

每个 query 还会经过一个分类头：

query embedding
↓
Linear classifier
↓
class logits

输出：

[B, N, num_classes + 1]

其中 +1 是：

no object / no mask

也就是说，每个 query 要么预测一个具体类别，要么预测为空。

## 十五、Mask2Former 的最终输出

Mask2Former 的基础输出是：

1. pred_logits: [B, N, C + 1]
2. pred_masks:  [B, N, H, W]

其中：

N：query 数量
C：类别数
H, W：mask 分辨率

例如：

pred_logits: [1, 100, 134]
pred_masks:  [1, 100, 256, 256]

表示：

100 个 query
每个 query 一个类别预测
每个 query 一张 mask

不同任务只是在这些基础输出上做不同解释。

## 十六、语义分割输出如何生成？

对于语义分割，Mask2Former 会把多个 query 的类别概率和 mask 概率组合起来。

可以理解为：

每个 query 提供一个类别概率
每个 query 提供一个空间 mask
把同类别 query 的 mask 融合
得到每个像素的类别预测

直观公式：

semantic_score[class, h, w]
=
Σ query_class_prob[i, class] × mask_prob[i, h, w]

然后对每个像素取最大类别：

pred[h, w] = argmax_class semantic_score[class, h, w]

这样就得到语义分割图。

## 十七、实例分割输出如何生成？

对于实例分割，重点是 things 类，也就是可数目标。

流程大致是：

1. 选择非 no-object 的 query
2. 过滤低置信度 query
3. 保留 things 类
4. 每个 query 的 mask 就是一个实例 mask
5. 根据类别概率和 mask 质量计算最终得分

例如输出：

query_5:
    class = person
    score = 0.94
    mask = person_1

query_9:
    class = person
    score = 0.87
    mask = person_2

query_21:
    class = car
    score = 0.91
    mask = car_1

## 十八、全景分割输出如何生成？

全景分割需要同时处理 things 和 stuff。

流程可以理解为：

1. 对 things 类 query，保留每个独立实例
2. 对 stuff 类 query，同类区域可以合并
3. 处理重叠区域
4. 给每个像素分配 panoptic id

例如：

person_1
person_2
car_1
road
sky
building

其中：

person_1 和 person_2 是不同实例
road 和 sky 是 stuff 区域

Mask2Former 论文报告其单一架构在 COCO、Cityscapes、ADE20K、Mapillary Vistas 等多个数据集上评估，并在 panoptic、instance、semantic segmentation 上取得很强性能；论文摘要中列出的结果包括 COCO panoptic 57.8 PQ、COCO instance 50.1 AP、ADE20K semantic 57.7 mIoU。

## 十九、Mask2Former 的训练方式

Mask2Former 训练时和 DETR 系列类似，使用集合预测思想。

因为模型输出固定数量的 query：

N 个预测

但真实 mask 数量是不固定的：

一张图可能有 3 个目标
也可能有 20 个目标

所以需要做匹配。

训练流程大致是：

1. 模型输出 N 个 class + mask
2. 使用 Hungarian Matching 匹配预测和真实 mask
3. 匹配上的 query 计算分类损失和 mask 损失
4. 没匹配上的 query 学习 no object
5. 反向传播更新模型
## 二十、Hungarian Matching 在 Mask2Former 中的作用

匈牙利匹配的作用是：

决定哪个 query 负责哪个真实 mask

例如真实标注有：

GT_1: person_1 mask
GT_2: car_1 mask
GT_3: road mask

模型输出 100 个 query：

query_1
query_2
...
query_100

匹配结果可能是：

GT_1 ↔ query_17
GT_2 ↔ query_43
GT_3 ↔ query_5

其他 query 预测 no object。

这样可以避免多个 query 同时预测同一个目标。

## 二十一、Mask2Former 的损失函数

Mask2Former 的损失主要包括：

1. Classification Loss
2. Mask Loss
3. Dice Loss

可以写成：

Loss = L_cls + L_mask + L_dice
### 1. Classification Loss

用于监督每个 query 的类别：

person
car
road
sky
no object
### 2. Mask Loss

用于监督预测 mask 和真实 mask 的像素级差异。

通常是二值交叉熵或类似形式：

预测 mask 和真实 mask 在像素级是否一致
### 3. Dice Loss

Dice Loss 用来衡量预测区域和真实区域的重叠程度。

Dice = 2 × 交集 / (预测区域 + 真实区域)

Dice Loss 对前景面积较小的 mask 比较友好。

## 二十二、为什么 Mask2Former 比普通 per-pixel segmentation 更统一？

传统语义分割模型通常输出：

[B, C, H, W]

本质是：

每个像素独立分类

这种方式天然适合语义分割，但不直接适合实例分割，因为它没有实例维度。

Mask2Former 输出：

[B, N, H, W] masks
+
[B, N, C] classes

本质是：

每个 query 代表一个区域或实例

所以它天然适合：

语义区域
实例目标
全景分割中的 things + stuff

这就是它比普通语义分割模型更统一的原因。

## 二十三、Mask2Former 和 DETR 的关系

DETR 是目标检测模型，输出：

query → class + bbox

Mask2Former 是分割模型，输出：

query → class + mask

可以理解为：

DETR:
    用 query 找目标框

Mask2Former:
    用 query 找目标区域 mask

两者都使用：

object queries
Transformer decoder
Hungarian Matching
set prediction

但是 Mask2Former 的重点是像素级 mask，而不是 bbox。

## 二十四、Mask2Former 和 Mask R-CNN 的区别
对比项	Mask R-CNN	Mask2Former
基本范式	Two-stage instance segmentation	Transformer mask classification
是否依赖 proposal	依赖 RPN proposals	不依赖 RPN proposals
RoI 操作	RoIAlign	不使用 RoIAlign 作为核心流程
输出方式	每个 RoI 一个 mask	每个 query 一个 mask
支持任务	主要实例分割	语义、实例、全景统一
匹配方式	proposal 与 GT 匹配	Hungarian matching
全局建模	CNN/RoI 为主	Transformer query 建模
结构统一性	偏实例分割	通用分割框架

简单理解：

Mask R-CNN：
    先检测候选框，再在框内预测 mask

Mask2Former：
    不走框候选流程，直接用 query 预测一组 mask
## 二十五、Mask2Former 和 U-Net 的区别
对比项	U-Net	Mask2Former
主要任务	语义分割	语义 / 实例 / 全景统一
输出方式	每个像素分类	每个 query 预测 mask + class
是否区分实例	通常不区分	可以区分
核心结构	Encoder-Decoder + Skip Connection	Backbone + Pixel Decoder + Transformer Decoder
小数据表现	通常较稳	更依赖预训练和数据
适合场景	医学、缺陷、小样本	通用分割、大规模数据、多任务

如果你的任务是：

只分割病灶区域 / 裂缝区域 / 缺陷区域

U-Net 可能更简单直接。

如果你的任务是：

既要语义分割，又要实例分割，甚至全景分割

Mask2Former 更合适。

## 二十六、Mask2Former 和 DeepLab 的区别
对比项	DeepLab	Mask2Former
任务	语义分割	通用图像分割
核心思想	Atrous Conv + ASPP	Mask Classification + Transformer
输出	每像素类别	Query mask + class
是否支持实例分割	不直接支持	支持
是否支持全景分割	不直接支持	支持
多尺度建模	ASPP	Pixel decoder + multi-scale features
全局关系	CNN/ASPP 为主	Transformer queries

简单说：

DeepLab：
    强语义分割模型

Mask2Former：
    通用分割框架
## 二十七、Mask2Former 和 SegFormer 的区别

SegFormer 是语义分割模型，主要特点是：

Transformer encoder
轻量 MLP decoder
语义分割输出

Mask2Former 则是：

Transformer decoder + queries
mask classification
统一语义、实例、全景分割

对比：

对比项	SegFormer	Mask2Former
主要任务	语义分割	通用分割
输出方式	per-pixel class	query mask + class
是否区分实例	不直接区分	可以区分
结构复杂度	相对简单	更复杂
统一性	主要语义分割	三类分割统一

## 二十八、Mask2Former 的优点

Mask2Former 的优点主要有：

1. 统一语义分割、实例分割、全景分割
2. 使用 mask classification，任务建模更统一
3. Masked attention 让 query 更关注局部目标区域
4. Transformer decoder 可以建模 query 之间关系
5. 不依赖 RPN / RoIAlign 这类两阶段检测流程
6. 多尺度特征和 pixel decoder 有利于精细 mask 预测
7. 在多个主流分割数据集上表现很强

特别是它的统一性非常适合在面试中讲：

传统方法是 task-specific architecture
Mask2Former 是 universal image segmentation architecture
## 二十九、Mask2Former 的缺点

Mask2Former 也有明显不足：

1. 模型结构比 U-Net / DeepLab 更复杂
2. 训练成本较高
3. 对数据量和预训练依赖更强
4. 部署难度比轻量 CNN 分割模型高
5. 不适合极低算力实时场景
6. 对小数据医学/缺陷分割，未必比 U-Net 更划算

如果只是做一个简单的二分类缺陷分割：

背景 / 裂缝

用 Mask2Former 可能有些“重”。

如果任务是：

多类别
多实例
全景分割
复杂场景理解

Mask2Former 的优势会更明显。

## 三十、Mask2Former 适合什么场景？

Mask2Former 适合：

城市街景全景分割
自动驾驶场景理解
复杂自然场景分割
多实例物体分割
通用图像分割研究
需要统一语义/实例/全景的项目
大规模数据集训练

例如：

COCO 实例/全景分割
ADE20K 语义分割
Cityscapes 街景分割
Mapillary Vistas 道路场景分割

Mask2Former 官方仓库列出的主要特性包括单一架构支持 panoptic、instance、semantic segmentation，并支持 ADE20K、Cityscapes、COCO、Mapillary Vistas 等数据集。

## 三十一、Mask2Former 不太适合什么场景？

如果你的场景是：

移动端实时分割
低算力边缘设备
小样本医学图像二分类分割
简单缺陷区域分割
只需要非常轻量的语义分割

可以优先考虑：

U-Net
U-Net++
DeepLabv3+ MobileNet
BiSeNet
SegFormer-B0
YOLO-Seg 轻量版本

Mask2Former 更适合复杂分割和统一任务框架，不一定适合所有工程场景。

## 三十二、Mask2Former 的训练流程

如果训练 Mask2Former，整体流程大致是：

1. 准备分割数据集
2. 根据任务准备 semantic / instance / panoptic 标注
3. 选择 backbone，例如 ResNet 或 Swin Transformer
4. 构建 pixel decoder
5. 构建 transformer decoder 和 queries
6. 设置 query 数量
7. 使用 Hungarian Matching 匹配预测 mask 和 GT mask
8. 计算 classification loss、mask loss、dice loss
9. 训练模型
10. 在验证集上计算 mIoU / AP / PQ
11. 根据任务做对应后处理
12. 输出 semantic / instance / panoptic 结果

不同任务常用指标不同：

语义分割：mIoU
实例分割：AP / mask AP
全景分割：PQ

## 三十三、Mask2Former 的指标怎么理解？
### 1. mIoU

语义分割常用。

mIoU = 每个类别 IoU 的平均值
### 2. AP / Mask AP

实例分割常用。

衡量实例 mask 的检测和分割质量。

### 3. PQ

全景分割常用。

PQ 全称是：

Panoptic Quality

它同时衡量：

分割质量
识别质量

也就是既看 mask 准不准，也看实例和类别识别对不对。

## 三十五、面试中如何介绍 Mask2Former？

如果面试官问：

你了解 Mask2Former 吗？

可以这样回答：

Mask2Former 是一种基于 Transformer 的通用图像分割框架，它的核心思想是把语义分割、实例分割和全景分割统一成 mask classification 问题。传统语义分割通常是对每个像素直接分类，而 Mask2Former 是通过一组 object queries 预测若干个 mask，并为每个 mask 预测类别。这样同一套输出形式可以适配语义分割、实例分割和全景分割。

它的结构主要包括 backbone、pixel decoder、Transformer decoder 和 mask classification head。Backbone 提取多尺度图像特征，pixel decoder 融合多尺度特征并生成像素级 embedding，Transformer decoder 使用 queries 与图像特征交互，最终每个 query 输出一个类别和一个 mask。Mask2Former 的关键改进是 masked attention，它会限制 query 的 cross-attention 主要发生在当前预测 mask 区域内，使 query 更专注于自己负责的目标或区域，从而提高分割质量和效率。

相比 U-Net、DeepLab 这类主要面向语义分割的模型，Mask2Former 更适合复杂场景下的统一分割任务；相比 Mask R-CNN，它不依赖 RPN 和 RoIAlign，而是直接用 query 预测 mask，结构上更接近 DETR 系列的集合预测思想。

## 三十六、如果面试官追问：Mask2Former 为什么能统一三种分割？

可以回答：

因为它把分割任务统一成“预测一组 mask，并给每个 mask 分类”的问题。对于语义分割，可以把同类别的 mask 合并成最终类别区域；对于实例分割，每个 thing 类 query 的 mask 就对应一个实例；对于全景分割，things 类保留独立实例，stuff 类合并同类区域。因此不同任务只是在同一组 mask classification 输出上采用不同后处理方式，而模型主体可以保持一致。

## 三十七、如果面试官追问：Masked Attention 的作用是什么？

可以回答：

Masked attention 是 Mask2Former 的核心设计。普通 Transformer cross-attention 中，每个 query 会关注整张图像的所有位置，这会带来很多无关背景干扰。Masked attention 会利用上一层预测得到的 mask，把 query 的注意力限制在该 mask 区域内，使 query 更专注于自己负责的目标或区域。这样可以让 mask 逐层细化，同时减少无关区域干扰，提升分割效果。

## 三十八、如果面试官追问：Mask2Former 和 Mask R-CNN 有什么区别？

可以回答：

Mask R-CNN 是两阶段实例分割方法，它先通过 RPN 生成候选框，再用 RoIAlign 提取每个候选区域的特征，最后预测类别、边界框和 mask。Mask2Former 不依赖候选框和 RoIAlign，而是通过 Transformer decoder 中的一组 queries 直接预测一组 mask 和类别。Mask R-CNN 主要面向实例分割，而 Mask2Former 使用 mask classification 形式，可以统一处理语义分割、实例分割和全景分割。