# MaskR-CNN

Mask R-CNN。它是实例分割领域非常经典的算法，可以理解为：

Mask R-CNN = Faster R-CNN + RoIAlign + Mask 分支

它既能做目标检测，也能做实例分割。

## 一、Mask R-CNN 是什么？

Mask R-CNN 是一种经典的实例分割 Instance Segmentation 算法。

目标检测只需要输出：

类别 + 边界框

例如：

person: bbox
car: bbox
dog: bbox

而实例分割不仅要输出目标框，还要输出每个目标的精确轮廓 mask：

person: bbox + mask
car: bbox + mask
dog: bbox + mask

所以 Mask R-CNN 的输出是：

类别 class
边界框 bbox
实例掩码 mask
置信度 score

Mask R-CNN 由 Kaiming He、Georgia Gkioxari、Piotr Dollár、Ross Girshick 提出，它是在 Faster R-CNN 基础上增加一个并行的 mask 预测分支，用于为每个 RoI 预测分割 mask。原论文也强调，Mask R-CNN 是一个概念简单、灵活、通用的实例分割框架

## 二、先区分：语义分割、实例分割、目标检测

理解 Mask R-CNN 前，先区分几个任务。

### 1. 图像分类

只判断整张图是什么：

输入：一张图片
输出：猫

不关心猫在哪里。

### 2. 目标检测

判断图中有什么，并给出框：

person: [x1, y1, x2, y2]
dog: [x1, y1, x2, y2]

它知道目标位置，但只是矩形框。

### 3. 语义分割

对每个像素分类：

像素属于人
像素属于车
像素属于道路
像素属于背景

但语义分割不区分同类不同个体。

例如图里有 3 个人，语义分割只知道这些像素都是“人”，但不知道哪个像素属于第 1 个人、第 2 个人、第 3 个人。

### 4. 实例分割

实例分割既要知道像素类别，又要区分不同实例。

例如：

person_1 mask
person_2 mask
person_3 mask
car_1 mask
car_2 mask

Mask R-CNN 做的就是这个任务。

## 三、Mask R-CNN 和 Faster R-CNN 的关系

Mask R-CNN 是从 Faster R-CNN 扩展来的。

Faster R-CNN 的输出是：

class + bbox

Mask R-CNN 的输出是：

class + bbox + mask

也就是说，Mask R-CNN 在 Faster R-CNN 的基础上额外加了一个 mask 分支。

Faster R-CNN 本身是两阶段目标检测器：第一阶段用 RPN 生成候选区域，第二阶段对候选区域进行分类和边界框回归；RPN 与检测网络共享整图卷积特征，从而高效生成 proposals。

Mask R-CNN 继承这个两阶段结构：

第一阶段：RPN 生成候选框 proposals
第二阶段：对每个 proposal 做分类、边界框回归、mask 预测

所以它本质上是：

Faster R-CNN:
    classification branch
    bbox regression branch

Mask R-CNN:
    classification branch
    bbox regression branch
    mask branch

## 四、Mask R-CNN 的整体结构

Mask R-CNN 主要由五部分组成：

1. Backbone 特征提取网络
2. FPN 特征金字塔，常见但不是唯一选择
3. RPN 区域候选网络
4. RoIAlign
5. Detection Head + Mask Head

整体流程可以写成：

输入图像
↓
Backbone 提取特征
↓
FPN 多尺度特征融合
↓
RPN 生成候选区域 proposals
↓
RoIAlign 对每个 proposal 提取固定尺寸特征
↓
分类分支：预测类别
↓
边界框分支：修正 bbox
↓
Mask 分支：预测实例 mask
↓
输出 class + bbox + mask

原论文中，Mask R-CNN 明确是在 Faster R-CNN 框架上，为每个 RoI 添加一个小型 FCN mask 分支，并与分类和边界框回归分支并行工作。

## 五、Backbone：特征提取网络

Backbone 负责从原始图像中提取视觉特征。

常见 backbone 包括：

ResNet-50
ResNet-101
ResNeXt
Swin Transformer

输入图片：

[B, 3, H, W]

经过 backbone 后得到特征图：

[B, C, H', W']

这些特征图会被后续 RPN 和 RoI Head 使用。

在实际工程中，Mask R-CNN 常常配合 FPN，Feature Pyramid Network 使用。FPN 可以融合不同尺度特征，使模型同时具备浅层细节和深层语义，对小目标、中目标、大目标都更友好。

## 六、FPN：为什么需要特征金字塔？

实例分割中，目标大小差异很大。

例如：

小目标：远处的人、瓶子、小零件
中目标：狗、椅子、行人
大目标：汽车、建筑、动物身体

深层特征语义强，但分辨率低；浅层特征分辨率高，但语义弱。

FPN 的作用是：

融合多尺度特征
让不同大小的目标都能找到合适的特征层

可以理解为：

浅层：适合小目标，边界细节更多
深层：适合大目标，语义信息更强
FPN：把它们融合起来

所以现代 Mask R-CNN 很多实现都会使用：

ResNet + FPN

这也是实际项目中很常见的 baseline。

## 七、RPN：生成候选区域

RPN，全称是：

Region Proposal Network

它的作用是：

找出图像中可能存在目标的区域

RPN 不判断具体类别，只判断：

这里有没有物体？
这个候选框的位置应该怎么修正？

RPN 输出的是一批 proposals：

proposal_1
proposal_2
proposal_3
...

这些 proposal 后面会送入 RoIAlign 和检测头。

RPN 在 Faster R-CNN 中提出，是一个全卷积网络，会在每个位置同时预测 objectness 分数和候选框边界，并与检测网络共享卷积特征。

## 八、RoI Pooling 的问题

在 Faster R-CNN / Fast R-CNN 中，常用 RoI Pooling 把不同大小的 RoI 转换成固定尺寸特征。

例如：

不同大小 proposal
↓
RoI Pooling
↓
7 × 7 特征

但 RoI Pooling 有一个问题：

量化误差

因为 proposal 坐标通常是浮点数，而特征图上的位置也是连续映射来的。

RoI Pooling 会把坐标取整：

3.7 → 4
8.2 → 8

这样对目标检测来说影响可能还可以接受，因为 bbox 是粗粒度输出。

但对 mask 分割来说，像素级对齐非常重要。

如果 RoI 特征和原图位置发生偏移，mask 边界就会不准。

## 九、RoIAlign：Mask R-CNN 的关键改进

Mask R-CNN 用 RoIAlign 替代 RoI Pooling。

RoIAlign 的核心思想是：

不再对 RoI 坐标进行粗暴取整，而是保留浮点坐标，并通过双线性插值获取特征值。

简单理解：

RoI Pooling:
    坐标取整
    容易产生错位

RoIAlign:
    不取整
    用双线性插值采样
    空间对齐更精确

这对 mask 分支非常重要，因为 mask 是像素级预测，空间错位会直接影响分割边界。Mask R-CNN 论文将 RoIAlign 作为关键设计，用于解决 RoI Pooling 的量化导致的 misalignment 问题。

可以这样记：

RoIAlign 是 Mask R-CNN 能够高质量预测 mask 的关键之一。

## 十、Detection Head：分类和边界框回归

经过 RoIAlign 后，每个 proposal 被转换成固定尺寸特征。

例如：

7 × 7 × C

这些特征会进入检测头。

检测头有两个分支：

分类分支 classification branch
边界框回归分支 bbox regression branch
### 1. 分类分支

判断当前 RoI 属于哪个类别：

background
person
car
dog
cat
...

如果有 C 个目标类别，通常输出：

C + 1

其中 +1 是背景类。

### 2. 边界框回归分支

进一步修正 proposal 的位置：

输入：RPN 给出的 proposal
输出：更精确的 bbox

RPN 只是粗定位，检测头会进一步精修。

## 十一、Mask Head：实例分割分支

Mask Head 是 Mask R-CNN 相比 Faster R-CNN 新增的部分。

它的作用是：

对每个 RoI 预测一个像素级 mask

通常结构是一个小型全卷积网络 FCN：

RoIAlign 特征
↓
多个 Conv 层
↓
Deconv / Upsample
↓
输出 mask

比如输出：

28 × 28 × K

其中：

28 × 28：mask 分辨率
K：类别数

如果有 80 个类别，则每个 RoI 可以预测：

80 个类别对应的 mask

但训练时通常只监督真实类别对应的那个 mask。

原论文说明，mask 分支是一个应用在每个 RoI 上的小型 FCN，以像素到像素的方式预测 segmentation mask。

## 十二、为什么 Mask 分支不用 softmax？

这是 Mask R-CNN 里一个很重要的细节。

Mask R-CNN 的 mask 分支通常对每个类别的 mask 使用 sigmoid，而不是在类别之间做 softmax。

原因是：

分类分支已经负责判断这个 RoI 是什么类别
mask 分支只需要为该类别预测二值 mask

例如一个 RoI 被分类为：

dog

那么最终只取 dog 类对应的 mask：

mask_dog

其他类别的 mask 不参与最终输出。

这样做的好处是：

分类任务和 mask 分割任务解耦
mask 分支只关注前景形状
不需要在 mask 像素上重复做类别竞争

也就是说：

分类分支：这个 RoI 是什么？
mask 分支：这个 RoI 内哪些像素属于这个实例？

这就是 Mask R-CNN 很重要的设计思想。

## 十三、Mask R-CNN 的三个输出分支

Mask R-CNN 对每个 RoI 有三个输出：

1. class prediction
2. bbox regression
3. mask prediction

可以理解为：

分类分支：
    这个 proposal 是人、车、狗，还是背景？

bbox 分支：
    这个 proposal 的边界框应该怎么修正？

mask 分支：
    在这个 proposal 内，哪些像素属于该目标？

最终输出：

目标类别
目标置信度
目标边界框
目标实例 mask

## 十四、Mask R-CNN 的损失函数

Mask R-CNN 的总损失由多个部分组成。

通常可以写成：

L = L_cls + L_box + L_mask

如果考虑 RPN，则完整一些是：

L = L_rpn_cls + L_rpn_box + L_cls + L_box + L_mask

其中：

L_rpn_cls：RPN 前景/背景分类损失
L_rpn_box：RPN 边界框回归损失
L_cls：RoI 分类损失
L_box：RoI 边界框回归损失
L_mask：mask 分割损失
### 1. RPN 分类损失

判断 anchor 是前景还是背景：

foreground / background
### 2. RPN 回归损失

修正 anchor 到 proposal：

dx, dy, dw, dh
### 3. RoI 分类损失

判断 RoI 属于哪个类别：

person / car / dog / background

通常使用交叉熵损失。

### 4. RoI bbox 回归损失

进一步修正 proposal 的位置。

常见使用 Smooth L1 Loss 或其变体。

### 5. Mask Loss

Mask 分支通常使用像素级二分类损失。

例如对每个 RoI 的真实类别 mask 计算：

Binary Cross Entropy

也就是说，对于一个被标注为 dog 的 RoI，只计算 dog 类 mask 的损失。

## 十五、训练时 Mask 标签怎么来？

训练 Mask R-CNN 需要实例级标注。

每个实例需要有：

类别 class
边界框 bbox
实例 mask

例如一张图中有两个人：

person_1:
    bbox
    mask_1

person_2:
    bbox
    mask_2

这两个 mask 是不同的。

这和语义分割不同。

语义分割只需要：

所有 person 像素 = person 类

实例分割必须知道：

哪些像素属于 person_1
哪些像素属于 person_2

所以 Mask R-CNN 的标注成本通常比目标检测和语义分割都更高。

## 十六、Mask R-CNN 的训练流程

训练流程可以概括为：

1. 输入图像和标注：bbox + class + instance mask
2. Backbone 提取特征
3. FPN 融合多尺度特征
4. RPN 生成 proposals
5. 根据 proposal 和 GT 匹配正负样本
6. RoIAlign 提取固定大小 RoI 特征
7. 分类分支预测类别
8. bbox 分支预测边界框修正
9. mask 分支预测实例 mask
10. 计算分类、回归、mask 损失
11. 反向传播更新参数

可以直观理解为：

先学会找目标
再学会判断类别和框
最后学会在框内画出目标轮廓

## 十七、Mask R-CNN 的推理流程

推理时没有真实标签。

流程是：

1. 输入图片
2. Backbone + FPN 提取特征
3. RPN 生成候选框 proposals
4. 对 proposals 做 NMS 和筛选
5. RoIAlign 提取 RoI 特征
6. 分类分支输出类别和置信度
7. bbox 分支修正边界框
8. mask 分支输出每个 RoI 的 mask
9. 根据预测类别选择对应类别 mask
10. 将 mask 映射回原图坐标
11. 输出最终实例分割结果

最终结果类似：

instance_1:
    class = person
    score = 0.96
    bbox = [x1, y1, x2, y2]
    mask = 二值图

instance_2:
    class = dog
    score = 0.91
    bbox = [x1, y1, x2, y2]
    mask = 二值图

## 二十、Mask R-CNN 和 YOLO 分割的区别

现代 YOLO 也支持实例分割，比如 YOLOv8-seg、YOLO11-seg。

它们和 Mask R-CNN 的区别大致是：

对比项	Mask R-CNN	YOLO-Seg
检测范式	两阶段	单阶段
速度	相对较慢	通常更快
精度	边界质量较稳	取决于版本和数据
结构	RPN + RoIAlign + mask head	一次前向输出检测和 mask
部署	较重	更适合实时部署
适合场景	高精度实例分割	实时实例分割

简单说：

Mask R-CNN：
    更像“先找候选区域，再精细分割”

YOLO-Seg：
    更像“一次性快速预测目标和 mask”

如果是工业高精度、医学、科研 baseline，可以考虑 Mask R-CNN。

如果是实时视频、移动端、边缘设备，更常用 YOLO-Seg。

## 二十一、Mask R-CNN 的优点

Mask R-CNN 的优点主要有：

1. 结构清晰，基于 Faster R-CNN 扩展
2. 同时完成检测和实例分割
3. RoIAlign 提升 mask 空间对齐精度
4. 对每个实例单独预测 mask
5. 分割质量通常较稳定
6. 可扩展性强
7. 可扩展到人体关键点检测等任务

原论文也指出，Mask R-CNN 不仅可以用于实例分割，还可以较自然地扩展到人体关键点估计等任务。

## 二十二、Mask R-CNN 的缺点

Mask R-CNN 的缺点也明显：

1. 两阶段结构，推理速度相对慢
2. 训练和部署比 U-Net / YOLO 更复杂
3. 需要实例级 mask 标注，标注成本高
4. 对密集小目标可能比较吃力
5. RoI 级别处理带来额外计算开销
6. 不太适合低算力实时场景

尤其是数据标注方面，Mask R-CNN 需要每个目标实例的独立 mask，比普通 bbox 检测数据更贵。

## 二十三、Mask R-CNN 适合什么场景？

Mask R-CNN 适合：

多目标实例分割
目标之间存在重叠
需要区分同类不同个体
对 mask 质量要求较高
实时性要求不是极端严格

典型场景：

行人实例分割
车辆实例分割
细胞实例分割
医学病灶实例分割
工业零件分割
遥感建筑物实例分割
农作物/果实计数与分割
机器人抓取中的物体分割

例如：

图中有多个苹果，需要分别分割每个苹果
图中有多个人，需要分别得到每个人的轮廓
图中有多个细胞，需要分开每个细胞实例

这些都很适合 Mask R-CNN。

## 二十四、Mask R-CNN 不适合什么场景？

如果任务是：

只需要判断每个像素属于哪个类别
不需要区分同类不同实例

例如：

道路 / 天空 / 建筑 / 草地
裂缝区域 / 背景
病灶区域 / 正常组织

那么 U-Net、DeepLab、SegFormer 可能更直接。

如果任务是：

实时视频检测
移动端部署
边缘设备推理

那么 YOLO-Seg、RT-DETR、轻量化分割模型可能更合适。

## 二十五、Mask R-CNN 的评价指标

实例分割常用指标包括：

AP
AP50
AP75
mask AP
bbox AP
mAP

其中：

bbox AP：边界框检测效果
mask AP：实例 mask 分割效果

COCO 实例分割任务通常重点看：

mask AP

如果是业务项目，也可以补充：

Precision
Recall
IoU
Dice
实例级召回率
平均 mask IoU

## 二十六、Mask R-CNN 项目训练流程

如果需要做一个 Mask R-CNN 项目，大致流程是：

1. 明确实例分割类别
2. 收集图像数据
3. 使用 CVAT / LabelMe / Label Studio 等工具标注实例 mask
4. 转换为 COCO instance segmentation 格式
5. 划分 train / val / test
6. 选择 Mask R-CNN + ResNet-FPN backbone
7. 加载 COCO 预训练权重
8. 修改类别数
9. 设置输入尺寸、batch size、学习率、epoch
10. 训练模型
11. 在验证集上观察 bbox AP 和 mask AP
12. 分析误检、漏检、mask 边界错误
13. 调整数据增强、学习率、阈值和类别采样
14. 在测试集上最终评估
15. 导出部署或做可视化 demo

## 二十七、实际项目中如何改进 Mask R-CNN？

可以从几个方向改进。

### 1. 更换 Backbone

例如：

ResNet50-FPN
ResNet101-FPN
ResNeXt-FPN
Swin Transformer-FPN
ConvNeXt-FPN

更强 backbone 通常能提升特征提取能力，但计算量也会增加。

### 2. 改进 FPN

可以尝试：

PANet
BiFPN
NAS-FPN
HRFPN

增强多尺度特征融合能力。

### 3. 改进 Mask Head

可以加深 mask 分支：

更多卷积层
更高 mask 分辨率
attention 模块
边界细化模块

如果目标边界复杂，mask head 的设计会影响分割质量。

### 4. 数据增强

常用增强包括：

随机翻转
随机缩放
随机裁剪
颜色扰动
Copy-Paste
Mosaic，视任务而定
旋转
模糊
噪声

实例分割增强要确保：

image、bbox、mask 同步变换
### 5. 类别不均衡处理

如果某些类别实例很少，可以使用：

类别重采样
少数类增强
Copy-Paste
class-balanced sampling
loss weighting

## 二十八、Mask R-CNN 的核心公式理解

虽然 Mask R-CNN 不像 DETR 那样有复杂匹配公式，但它的核心可以总结为：

每个 RoI:
    class = 分类头(RoI feature)
    bbox = 回归头(RoI feature)
    mask = mask head(RoI feature)

损失函数：

L = L_cls + L_box + L_mask

如果加上 RPN：

L_total =
    L_rpn_cls
  + L_rpn_box
  + L_roi_cls
  + L_roi_box
  + L_mask

训练目标：

RPN 学会提出候选区域
分类头学会识别类别
bbox 头学会精修边界框
mask 头学会预测实例轮廓

## 二十九、Mask R-CNN 核心总结

核心说明：

Mask R-CNN 是一种经典的实例分割算法，可以看作是在 Faster R-CNN 的基础上增加了一个并行的 mask 分支。它首先使用 backbone 和 FPN 提取图像多尺度特征，然后通过 RPN 生成候选区域 proposals。对于每个 proposal，模型使用 RoIAlign 提取固定尺寸的 RoI 特征，之后分别送入分类分支、边界框回归分支和 mask 分支。分类分支判断目标类别，bbox 分支修正目标框，mask 分支为每个实例预测像素级掩码。

Mask R-CNN 的关键改进之一是 RoIAlign。相比 RoI Pooling，RoIAlign 不会对 RoI 坐标进行取整，而是使用双线性插值保留更精确的空间对齐信息，这对像素级 mask 预测非常重要。它的整体损失包括分类损失、边界框回归损失和 mask 损失。由于它能同时输出类别、边界框和实例 mask，所以非常适合需要区分同类不同目标的实例分割任务。

## 三十、延伸知识：Mask R-CNN 为什么要用 RoIAlign？

核心说明：

因为 mask 预测是像素级任务，对空间位置非常敏感。传统 RoI Pooling 会把 RoI 坐标映射到特征图时进行取整，这会产生量化误差，导致 RoI 特征和原图目标位置不完全对齐。RoIAlign 取消取整操作，保留浮点坐标，并通过双线性插值采样特征，从而减少空间错位，提高 mask 边界预测精度。

## 三十一、延伸知识：Mask R-CNN 和 Faster R-CNN 有什么区别？

核心说明：

Faster R-CNN 主要做目标检测，输出目标类别和边界框。Mask R-CNN 在 Faster R-CNN 的基础上增加了一个 mask 分支，用来为每个 RoI 预测实例级分割 mask。同时 Mask R-CNN 使用 RoIAlign 替代 RoI Pooling，改善 RoI 特征和原图之间的空间对齐问题。因此 Mask R-CNN 可以同时完成目标检测和实例分割。

## 三十二、延伸知识：Mask R-CNN 和 U-Net 有什么区别？

核心说明：

U-Net 通常用于语义分割，它对整张图像做像素级分类，但一般不区分同一类别中的不同实例。Mask R-CNN 是实例分割算法，它会先检测出每个目标实例，再为每个实例预测独立 mask。比如图中有三个人，U-Net 通常输出所有人的整体 person 区域，而 Mask R-CNN 会分别输出 person_1、person_2、person_3 的 mask。所以如果任务需要区分同类不同目标，Mask R-CNN 更合适；如果只需要分割类别区域，U-Net 更直接。

