# DETR

## 一、DETR 是什么？

DETR，全称：

DEtection TRansformer

中文可以理解为：

基于 Transformer 的端到端目标检测模型

目标检测任务本来要预测：

类别 class
边界框 bounding box
置信度 score

传统检测器通常会依赖很多人工设计流程，比如：

anchor 设计
候选框 proposal
正负样本分配
NMS 后处理
多尺度特征融合

而 DETR 的核心想法是：

不再把目标检测看成“密集框预测 + 去重”的问题，而是看成“直接预测一个目标集合”的问题。

也就是说，模型直接输出一组预测结果：

prediction 1: person, bbox, score
prediction 2: car, bbox, score
prediction 3: no object
...
prediction N: no object

然后通过一对一匹配，让每个真实目标只对应一个预测结果。

这就是 DETR 最重要的思想：set prediction，集合预测。原论文明确指出，DETR 通过集合式全局损失和二分图匹配，强制模型产生唯一预测，并去除了 NMS 和 anchor generation 这类手工组件。

## 二、DETR 解决了传统目标检测中的什么问题？

在 DETR 之前，主流目标检测器大致分成两类。

### 1. 两阶段检测器

代表算法：

R-CNN
Fast R-CNN
Faster R-CNN
Mask R-CNN

它们的思路是：

先生成候选区域 proposal
再对 proposal 分类和回归

比如 Faster R-CNN：

Backbone 提取特征
↓
RPN 生成 proposals
↓
RoI Pooling / RoI Align
↓
分类 + 边界框回归
↓
NMS

优点是精度高，缺点是流程复杂。

### 2. 单阶段检测器

代表算法：

YOLO
SSD
RetinaNet
FCOS

它们的思路是：

在特征图上密集预测大量候选框
再通过 NMS 去除重复框

优点是速度快，缺点是通常需要复杂的正负样本分配、anchor 设计或者 NMS 后处理。

### 3. DETR 的新思路

DETR 不走这两条传统路线。

它不需要：

anchor
proposal
RPN
RoI Pooling
NMS
复杂的正负样本采样

它直接预测一个目标集合：

输入图像
↓
CNN 提取特征
↓
Transformer 编码全局上下文
↓
Object Queries 查询目标
↓
输出固定数量的预测框和类别
↓
匈牙利匹配训练

所以 DETR 的设计非常简洁。

可以这样理解：

YOLO：密集预测很多框，再用 NMS 去重
Faster R-CNN：先找候选框，再精细分类和回归
DETR：直接预测一组最终目标
## 三、DETR 的整体结构

DETR 的整体结构可以分成五个部分：

1. CNN Backbone
2. 1×1 Conv 降维
3. Transformer Encoder
4. Transformer Decoder + Object Queries
5. Prediction Heads

整体流程如下：

输入图像
↓
CNN Backbone 提取特征图
↓
1×1 Conv 调整通道数
↓
Flatten 成序列 + 加位置编码
↓
Transformer Encoder 编码图像全局特征
↓
Transformer Decoder 使用 object queries 查询目标
↓
FFN 输出类别和边界框
↓
Hungarian Matching 计算训练损失

原始 DETR 使用 CNN backbone 加 Transformer encoder-decoder 架构，并通过固定数量的 learned object queries 并行输出最终预测集合。

## 四、Backbone：图像特征提取

DETR 的第一部分是 CNN backbone。

常见 backbone：

ResNet-50
ResNet-101

输入图像：

[B, 3, H, W]

经过 CNN 后输出特征图：

[B, C, H', W']

例如输入图像是：

800 × 1066

经过 ResNet 后，可能得到：

[B, 2048, H/32, W/32]

Backbone 的作用和普通目标检测模型一样：

提取边缘
提取纹理
提取局部形状
提取高级语义特征

但是和 YOLO / Faster R-CNN 不同，DETR 后面不是接检测头密集预测，而是把这个特征图送入 Transformer。

## 五、为什么要 1×1 Conv？

CNN backbone 输出的通道数可能很大，比如：

2048

而 Transformer 通常使用固定维度，比如原始 DETR 中常用：

d_model = 256

所以需要一个 1×1 Conv 把通道数从 2048 降到 256：

[B, 2048, H', W']
↓
1×1 Conv
↓
[B, 256, H', W']

这个 1×1 Conv 的作用是：

调整通道数
减少计算量
把 CNN 特征映射到 Transformer 需要的 embedding 维度
## 六、图像特征如何输入 Transformer？

Transformer 处理的是序列。

但是 CNN 输出的是二维特征图：

[B, C, H', W']

所以 DETR 需要把二维特征图展平为序列：

[B, C, H', W']
↓
flatten
↓
[B, H' × W', C]

例如：

feature map = 25 × 34

展平后序列长度是：

25 × 34 = 850

也就是有 850 个 image tokens。

每个 token 代表图像特征图上的一个空间位置。

所以 DETR 相当于把一张图像转换成一个视觉 token 序列，然后送入 Transformer。

## 七、为什么 DETR 需要位置编码？

Transformer 自身不具备天然的空间顺序感。

如果只输入一堆 image tokens，它不知道：

哪个 token 在左上角
哪个 token 在右下角
哪个 token 相邻
哪个 token 距离很远

而目标检测强烈依赖空间位置。

所以 DETR 需要给每个图像 token 加上位置编码：

图像 token 特征 + 位置编码

位置编码告诉模型：

这个特征来自图像的哪个空间位置

否则 Transformer 只能看到一组特征集合，而无法理解空间结构。

## 八、Transformer Encoder 的作用

Transformer Encoder 的作用是：

让图像中不同位置的特征进行全局信息交互。

CNN 的卷积更擅长局部特征提取，比如边缘、纹理和局部形状。

Transformer 的 self-attention 更擅长建模全局关系，比如：

目标和背景的关系
目标和目标之间的关系
远距离区域之间的关系
遮挡关系
上下文关系

Encoder 的输入是展平后的图像特征序列：

[B, H'×W', C]

经过多层 Transformer Encoder 后，输出仍然是：

[B, H'×W', C]

但每个位置的特征已经融合了全局上下文。

举个例子：

一辆车被树挡住了一部分

CNN 局部卷积可能只看到车的一部分。

Transformer Encoder 可以让车身不同区域、道路区域、周围上下文互相交互，从而帮助模型理解这是一个完整目标。

## 九、Object Query 是什么？

Object Query 是 DETR 的核心概念之一。

可以把 object query 理解成：

一组可学习的目标查询向量

假设 DETR 设置：

num_queries = 100

那么模型有 100 个 object queries：

query 1
query 2
query 3
...
query 100

每个 query 都尝试去图像特征中“查询”一个目标。

最后每个 query 会输出一个预测结果：

类别
边界框

如果图中只有 5 个目标，那么可能是：

query 3  → person
query 17 → car
query 41 → dog
query 68 → bicycle
query 72 → traffic light
其他 query → no object

所以 object query 可以理解为：

DETR 预留的一组“目标槽位”，每个槽位最终要么负责一个目标，要么预测为 no object。

这和 YOLO 的密集网格预测非常不同。

YOLO 是：

每个空间位置预测目标

DETR 是：

每个 query 预测一个目标槽位
## 十、Transformer Decoder 的作用

Transformer Decoder 的输入是 object queries。

Decoder 会让这些 queries 去关注 Encoder 输出的图像特征。

也就是：

object queries
↓
通过 cross-attention 关注图像特征
↓
每个 query 获取自己关心的目标信息
↓
输出目标表示

Decoder 中有两类注意力：

1. query 之间的 self-attention
2. query 和图像特征之间的 cross-attention
### 1. Query self-attention

让不同 query 之间互相交流。

它的作用是避免多个 query 预测同一个目标。

比如：

query 1 想预测同一辆车
query 2 也想预测同一辆车

通过 query 之间的 self-attention，它们可以协调分工，减少重复预测。

### 2. Cross-attention

让 query 去关注图像特征。

例如：

query 3 关注左下角的人
query 17 关注中间的汽车
query 41 关注右上角的红绿灯

每个 query 会从整张图像特征中提取和自己相关的信息。

最后每个 query 形成一个目标级表示。

## 十一、DETR 的输出是什么？

假设：

num_queries = 100
num_classes = 91

那么 DETR 会输出：

100 个预测结果

每个预测结果包括：

类别概率
边界框坐标

类别输出：

[B, 100, num_classes + 1]

其中 +1 是 no object 类。

边界框输出：

[B, 100, 4]

通常边界框格式是归一化的：

cx, cy, w, h

其中：

cx：框中心点 x 坐标
cy：框中心点 y 坐标
w：框宽度
h：框高度

数值范围通常在：

0 ~ 1

推理时再转换回原图坐标。

## 十二、为什么 DETR 不需要 NMS？

传统检测器会产生很多重复框。

例如一辆车可能被预测成：

box1 score = 0.93
box2 score = 0.89
box3 score = 0.84

所以需要 NMS：

保留最高分框
删除高度重叠框

但 DETR 的训练方式是一对一匹配。

每个真实目标只会匹配一个预测 query：

GT car ↔ query 17
GT person ↔ query 3
GT dog ↔ query 41

其他 query 如果没有匹配到真实目标，就被训练成：

no object

所以 DETR 被训练成：

一个目标只由一个 query 负责预测

因此理论上不会像 YOLO 那样对同一个目标产生大量重复框。

这就是 DETR 不需要 NMS 的根本原因。原始 DETR 论文明确指出，它通过二分图匹配和集合预测方式移除了 NMS 和 anchor generation。

## 十三、Hungarian Matching 匈牙利匹配

DETR 训练中最关键的一步是：

Hungarian Matching

也叫：

匈牙利匹配
二分图匹配

因为 DETR 输出的是固定数量预测，比如 100 个。

但真实目标数量是不固定的，比如一张图里可能有 3 个目标、10 个目标、20 个目标。

所以训练时需要解决一个问题：

哪个预测 query 应该负责哪个真实目标？

假设模型输出：

100 个 predictions

真实框有：

5 个 ground truth boxes

匈牙利匹配会在 100 个预测和 5 个真实框之间找一个最优一对一匹配。

比如：

GT 1 ↔ prediction 8
GT 2 ↔ prediction 21
GT 3 ↔ prediction 37
GT 4 ↔ prediction 66
GT 5 ↔ prediction 91

其余 95 个预测都应该是：

no object
## 十四、匹配代价由什么组成？

匈牙利匹配需要计算每个预测和每个真实目标之间的匹配成本。

匹配成本通常包括：

1. 分类成本
2. L1 bbox 成本
3. GIoU 成本

可以理解为：

cost = class_cost + bbox_L1_cost + giou_cost
### 1. 分类成本

如果真实类别是 car，那么预测越像 car，成本越低。

预测 car 概率高 → 成本低
预测 car 概率低 → 成本高
### 2. L1 bbox 成本

衡量预测框和真实框坐标差异。

预测框越接近真实框 → 成本越低
预测框离真实框越远 → 成本越高
### 3. GIoU 成本

IoU 衡量两个框的重叠程度：

IoU = 交集面积 / 并集面积

GIoU 是 IoU 的改进版本，可以在两个框没有重叠时仍然提供有效梯度。

DETR 使用 L1 loss 和 GIoU loss 共同优化边界框。

## 十五、DETR 的损失函数

DETR 的训练损失可以分成两步：

第一步：Hungarian Matching 找到预测和 GT 的对应关系
第二步：对匹配结果计算 loss

整体 loss 包括：

1. 分类 loss
2. bbox L1 loss
3. GIoU loss

可以写成：

Loss = L_cls + L_bbox + L_giou

对于匹配上的预测：

计算类别损失
计算框回归损失

对于没有匹配上的预测：

学习预测 no object

这就是 DETR 能进行集合预测的关键。

## 十六、DETR 的训练流程

完整训练流程如下：

1. 输入图像和真实标注框
2. CNN backbone 提取图像特征
3. 1×1 Conv 调整通道数
4. Flatten 成图像 token 序列
5. 加入位置编码
6. Transformer Encoder 编码图像全局上下文
7. Transformer Decoder 使用 object queries 查询目标
8. 每个 query 输出类别和 bbox
9. 使用 Hungarian Matching 匹配 predictions 和 GT
10. 计算分类 loss、L1 loss、GIoU loss
11. 反向传播更新模型参数

更直观地说：

模型先预测 100 个可能目标
↓
匈牙利算法找出其中哪些预测对应真实目标
↓
对应上的预测学习类别和框
↓
对应不上的预测学习 no object
## 十七、DETR 的推理流程

DETR 推理时非常简单。

1. 输入图像
2. CNN backbone 提取特征
3. Transformer encoder-decoder 输出 N 个预测
4. 过滤 no object 和低分预测
5. 输出剩余检测框

注意：

不需要 anchor 解码
不需要 proposal
不需要 RoI Pooling
不需要 NMS

这就是 DETR 的“端到端”优势。

官方 DETR 仓库也强调其推理流程非常简洁，并提供 PyTorch 训练代码和预训练模型；其说明中提到 ResNet-50 DETR 在 COCO 上达到约 42 AP，并与 Faster R-CNN 基线相当。

## 十八、DETR 为什么说是端到端？

传统检测器中很多部分不是网络直接学习出来的，比如：

anchor 尺寸设计
proposal 筛选
NMS 阈值
正负样本分配规则

这些都是人为设计组件。

DETR 的端到端体现在：

输入图像
↓
神经网络
↓
直接输出目标集合

训练时用集合损失直接监督最终结果。

它不需要在推理阶段再靠 NMS 去修补重复预测问题。

所以 DETR 的检测流程更像：

输入 → 模型 → 最终检测结果

而不是：

输入 → 模型 → 大量候选框 → 人工后处理 → 最终检测结果

## 二十一、DETR 的优点

DETR 的优点主要有：

1. 结构简洁
2. 不需要 anchor
3. 不需要 NMS
4. 不需要 proposal
5. 端到端训练
6. 可以建模目标之间的全局关系
7. 思想统一，容易扩展到分割等任务

其中最重要的是：

DETR 把目标检测从“密集候选框筛选问题”
变成了“目标集合预测问题”

这个思想影响了后来的大量 DETR 系列模型，比如：

Deformable DETR
Conditional DETR
DAB-DETR
DN-DETR
DINO
RT-DETR
## 二十二、DETR 的缺点

原始 DETR 也有明显缺点。

1. 收敛慢

原始 DETR 通常需要较长训练轮数。

原因包括：

object query 初始不知道该关注哪里
匈牙利匹配早期不稳定
Transformer 需要学习空间关系和检测规则

Deformable DETR 论文明确指出，原始 DETR 存在收敛慢和特征空间分辨率受限的问题，而 Deformable DETR 通过只关注少量采样点来缓解这些问题，并能用少得多的训练轮数取得更好效果。

2. 小目标检测较弱

原始 DETR 通常只使用较低分辨率的特征图。

小目标在低分辨率特征图上可能只占很少位置，容易被忽略。

例如：

原图中一个小目标是 20×20 像素
经过 32 倍下采样后
可能不到 1 个特征点

所以小目标检测能力不如使用 FPN 多尺度特征的检测器。

3. 计算量较大

Transformer encoder 对图像 token 做 self-attention。

如果特征图大小是：

H' × W'

序列长度就是：

N = H' × W'

Self-attention 的复杂度大约是：

O(N²)

图像分辨率越高，token 越多，计算和显存开销越大。

这也是为什么后来的 Deformable DETR、RT-DETR 等都在优化 attention 计算。

## 二十六、DETR 的面试回答版本

如果面试官问：

你了解 DETR 吗？

可以这样回答：

DETR 是一种基于 Transformer 的端到端目标检测算法，全称是 Detection Transformer。它最大的特点是把目标检测建模成集合预测问题，而不是传统的候选框筛选问题。传统检测器通常依赖 anchor、proposal、NMS 等人工设计组件，而 DETR 使用 CNN backbone 提取图像特征，然后将特征展平成序列输入 Transformer encoder，再通过一组可学习的 object queries 在 Transformer decoder 中和图像特征交互，最后每个 query 输出一个类别和边界框。

在训练阶段，DETR 使用 Hungarian Matching 将固定数量的预测和真实目标做一对一匹配，匹配上的预测计算分类损失、L1 边界框损失和 GIoU 损失，未匹配的预测学习 no object。由于每个真实目标只匹配一个 query，模型被训练成直接输出不重复的目标集合，所以推理时不需要 NMS。

DETR 的优点是结构简洁、端到端、不需要 anchor 和 NMS，并且 Transformer 可以建模全局上下文和目标之间的关系。缺点是原始 DETR 收敛较慢，对小目标不够友好，计算量也较大。后续的 Deformable DETR、DINO、RT-DETR 等方法主要就是围绕收敛速度、多尺度特征和实时性进行改进。

## 二十七、如果面试官追问：DETR 为什么不需要 NMS？

可以回答：

因为 DETR 的训练不是让大量候选框分别预测目标，而是通过 Hungarian Matching 做一对一集合匹配。每个真实目标只会匹配一个预测 query，而其他没有匹配到目标的 query 会被训练成 no object。这样模型学习到的是直接输出一组不重复的目标集合，而不是输出大量重叠候选框。因此推理时通常只需要过滤低置信度和 no object 预测，不需要再用 NMS 去重。

## 二十八、如果面试官追问：Object Query 怎么理解？

可以回答：

Object query 可以理解为 DETR 中预设的一组可学习目标槽位。每个 query 会在 decoder 中通过 cross-attention 去图像特征中查询和自己相关的目标信息，最后输出一个类别和边界框。如果某个 query 匹配到了真实目标，它就负责预测这个目标；如果没有匹配到目标，它就预测 no object。所以 object query 不是图像中的某个固定位置，而是一组学习出来的目标查询向量。

## 二十九、如果面试官追问：DETR 和 YOLO 有什么区别？

可以回答：

YOLO 是典型的单阶段检测器，它通常在特征图上进行密集预测，产生大量候选框，然后通过置信度过滤和 NMS 得到最终结果。DETR 则是基于 Transformer 的集合预测检测器，它使用固定数量的 object queries 直接预测最终目标集合，并通过 Hungarian Matching 做一对一监督，因此不需要 anchor 和 NMS。

YOLO 的优势是速度快、部署成熟，适合实时检测；DETR 的优势是结构简洁、端到端、全局建模能力强，但原始 DETR 收敛慢、小目标检测较弱，所以后来出现了 Deformable DETR 和 RT-DETR 等改进版本。