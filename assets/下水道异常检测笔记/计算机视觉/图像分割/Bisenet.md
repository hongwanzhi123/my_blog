# Bisenet

下面详细介绍 BiSeNet。它是实时语义分割领域非常经典的一类网络，核心目标是：

在尽量保持高 FPS 的情况下，完成像素级语义分割，并尽量减少因为轻量化带来的边界细节损失。

## 一、BiSeNet 是什么？

BiSeNet 全称是：

Bilateral Segmentation Network

中文可以理解为：

双边分割网络

它主要用于 实时语义分割 Real-time Semantic Segmentation。

语义分割任务是：

输入：一张图像
输出：每个像素所属的类别

例如自动驾驶场景中，需要把每个像素分成：

道路 / 车辆 / 行人 / 建筑 / 天空 / 树木 / 标志牌

BiSeNet 的重点不是单纯追求最高精度，而是在速度和精度之间取得较好平衡。BiSeNet v1 论文指出，语义分割既需要丰富的空间信息，也需要足够大的感受野，但很多实时方法为了速度会牺牲空间分辨率，从而导致精度下降；BiSeNet 就是为了解决这个矛盾而提出的双路径结构。

## 二、BiSeNet 解决的核心矛盾

语义分割有两个关键需求：

1. 空间细节
2. 语义上下文
### 1. 空间细节

分割任务需要知道目标边界在哪里。

例如：

道路边缘
行人轮廓
车辆边界
裂缝边缘
建筑物轮廓

如果模型下采样太多，边界就会模糊，小目标也容易丢失。

### 2. 语义上下文

模型还需要知道每个区域到底是什么。

例如某个灰色区域到底是：

道路
墙面
建筑物
天空
车辆

这需要较大的感受野和高层语义信息。

### 3. 速度要求

实时语义分割还要求模型非常快。

例如自动驾驶、视频分析、机器人视觉中，模型可能需要：

30 FPS
60 FPS
甚至 100 FPS 以上

如果用很重的 DeepLab、HRNet、Mask2Former 这类模型，精度可能不错，但实时部署压力很大。

BiSeNet 的核心思想就是：

不用一条网络同时兼顾细节和语义，而是把二者拆开：一条路径保留空间细节，一条路径提取语义上下文，最后再融合。

## 三、BiSeNet v1 的整体结构

BiSeNet v1 主要由四个部分组成：

1. Spatial Path
2. Context Path
3. Attention Refinement Module, ARM
4. Feature Fusion Module, FFM

整体结构可以理解为：

输入图像
 ├── Spatial Path：保留空间细节
 └── Context Path：提取语义上下文
          ↓
     Attention Refinement
          ↓
   Feature Fusion Module
          ↓
   Segmentation Head
          ↓
   输出分割结果

BiSeNet v1 论文明确提出了 Spatial Path 和 Context Path：Spatial Path 用小步幅保留空间信息并生成高分辨率特征，Context Path 用快速下采样策略获得足够感受野，然后用 Feature Fusion Module 高效融合两路特征。

## 四、Spatial Path：空间路径

Spatial Path 的作用是：

保留图像的空间细节信息

它主要关注：

边缘
轮廓
纹理
目标位置
局部细节

BiSeNet v1 中，Spatial Path 通常比较浅，只使用几层卷积，并且不会过度下采样。论文实现中 Spatial Path 使用 3 个卷积层，用于生成相对高分辨率的空间特征。

可以理解为：

输入图像：H × W
↓
少量卷积和下采样
↓
输出较高分辨率特征，比如 H/8 × W/8

它的特点是：

网络浅
计算较轻
保留细节
语义能力较弱

也就是说，Spatial Path 能比较好地知道：

边界在哪里
轮廓在哪里
纹理变化在哪里

但它不一定能准确判断：

这个区域到底是车还是墙
这个区域到底是道路还是人行道

## 五、Context Path：上下文路径

Context Path 的作用是：

提取高级语义信息和大感受野上下文

它主要关注：

类别语义
全局上下文
大范围区域关系
目标整体结构

Context Path 通常使用轻量化分类 backbone，例如：

Xception39
ResNet18
其他轻量 CNN

BiSeNet v1 论文中使用 Xception39 作为 Context Path 的一种配置，并通过快速下采样来获得较大的感受野。

Context Path 的特点是：

下采样快
感受野大
语义信息强
空间细节弱

可以理解为：

Spatial Path：
    看得细，但理解浅

Context Path：
    看得远，理解深，但细节粗

这两个路径刚好互补。

## 六、为什么 BiSeNet 要用双路径？

如果只用一条很深的网络，会出现问题：

下采样多 → 速度快、语义强，但细节丢失
下采样少 → 细节好，但计算量大、速度慢

BiSeNet 的设计是把这两个目标拆开：

Spatial Path：
    少量下采样，保留空间细节

Context Path：
    快速下采样，获得语义上下文

融合模块：
    把细节和语义结合起来

这就是 “Bilateral” 的含义。

可以简单记成：

BiSeNet = 空间细节路径 + 语义上下文路径 + 特征融合

## 七、Attention Refinement Module：注意力细化模块

BiSeNet v1 在 Context Path 中使用了 Attention Refinement Module，ARM。

ARM 的作用是：

对上下文特征进行通道注意力增强

它大致流程是：

输入特征
↓
Global Average Pooling
↓
1×1 Conv
↓
BatchNorm
↓
Sigmoid
↓
得到通道注意力权重
↓
和原特征相乘

可以理解为：

哪些通道重要，就增强哪些通道
哪些通道不重要，就抑制哪些通道

比如在道路场景中，某些通道可能更关注车辆，某些通道更关注道路，某些通道更关注天空。ARM 可以帮助模型根据全局上下文调整特征通道的重要性。

它解决的问题是：

Context Path 虽然语义强，但直接拿来融合可能包含冗余信息
ARM 先对上下文特征做筛选和增强

## 八、Feature Fusion Module：特征融合模块

Feature Fusion Module，FFM 是 BiSeNet v1 的另一个关键模块。

它的作用是：

融合 Spatial Path 的空间细节特征
和 Context Path 的语义上下文特征

通常流程是：

Spatial Feature
Context Feature
↓
Concat
↓
Conv + BN + ReLU
↓
Attention / Reweight
↓
输出融合特征

为什么不能简单相加？

因为两路特征性质不同：

Spatial Path：低层细节强
Context Path：高层语义强

如果直接相加，可能会造成信息混杂。

FFM 通过卷积和注意力重新调整融合后的特征，让模型学习：

哪些空间细节需要保留
哪些语义信息更重要
如何平衡边界和类别判断

BiSeNet v1 论文中也明确提出 Feature Fusion Module 用来高效结合两条路径的特征。

## 九、BiSeNet v1 的输出

BiSeNet 是语义分割模型，输出通常是：

[B, num_classes, H, W]

其中：

B：batch size
num_classes：类别数
H, W：图像高度和宽度

例如 Cityscapes 有 19 个常用语义类别，那么输出可以是：

[B, 19, H, W]

每个像素位置都有 19 个类别分数。

推理时：

pred = output.argmax(dim=1)

得到：

[B, H, W]

每个像素的值就是类别 ID。

## 十、BiSeNet v1 的辅助损失

BiSeNet v1 训练时通常会在中间特征层加辅助监督。

原因是：

实时模型比较轻
深层语义路径可能比较短
中间监督可以帮助梯度传播
提升训练稳定性

常见做法是：

主输出 loss
+
Context Path 中间输出 auxiliary loss

总损失可以写成：

Loss = main_loss + aux_loss_1 + aux_loss_2

辅助损失只在训练时使用，推理时一般丢弃，不增加推理开销。

## 十一、BiSeNet v1 的性能特点

BiSeNet v1 的主要特点是：

速度快
结构清晰
空间路径和上下文路径分工明确
适合实时语义分割

BiSeNet v1 论文报告，在 Cityscapes 2048×1024 输入上，模型达到 68.4% mIoU，并在单张 NVIDIA Titan XP 上达到 105 FPS，体现了其速度和精度平衡。

不过 v1 也有一些不足：

双路径带来额外计算
Spatial Path 可能存在冗余
依赖分类 backbone 的 Context Path 未必是最适合分割的结构

这些问题后来在 BiSeNet v2 和 STDC 系列中被进一步改进。

## 十二、BiSeNet v2：进一步强化实时性

BiSeNet v2 是 BiSeNet 的重要改进版本。

它仍然保留“双边网络”的思想，但把 v1 中的 Spatial Path 和 Context Path 改成了：

Detail Branch
Semantic Branch

整体结构为：

输入图像
 ├── Detail Branch：浅层宽通道，保留低级细节
 └── Semantic Branch：深层窄通道，快速下采样提取语义
        ↓
 Guided Aggregation Layer
        ↓
 输出分割结果

BiSeNet v2 论文指出，低级细节和高级语义对语义分割都很重要，因此设计了 Detail Branch 和 Semantic Branch：前者用宽通道和浅层结构捕获低级细节，后者用窄通道和深层结构获取高级语义上下文，并通过 Guided Aggregation Layer 融合二者。

## 十三、Detail Branch：细节分支

Detail Branch 对应 v1 中 Spatial Path 的思想。

它的作用是：

捕获低级细节
保留高分辨率特征
帮助恢复边界

它的特点是：

层数浅
通道较宽
下采样较少
保留空间信息

为什么通道要宽？

因为 Detail Branch 层数浅，如果通道太窄，表达能力不足；用较宽通道可以让它在浅层就捕获丰富的边缘、纹理和细节信息。

## 十四、Semantic Branch：语义分支

Semantic Branch 的作用是：

获取高级语义信息
提供大感受野
理解图像全局上下文

它的特点是：

层数较深
通道较窄
快速下采样
计算量较小

和 Detail Branch 相比：

Detail Branch：
    高分辨率，细节强，语义弱

Semantic Branch：
    低分辨率，细节弱，语义强

两者仍然是互补关系。

## 十五、Guided Aggregation Layer：引导聚合层

BiSeNet v2 的关键模块是：

Guided Aggregation Layer, GAL

它的作用是：

让语义分支引导细节分支
让细节分支补充空间边界

可以理解为：

Semantic Branch 告诉模型：
    哪些区域语义上重要

Detail Branch 告诉模型：
    边界和细节在哪里

GAL 不是简单 concat，而是通过引导机制增强两路特征的有效融合。

BiSeNet v2 论文明确提出 Guided Aggregation Layer 用于增强两类特征之间的相互连接和融合，同时还设计了 booster training strategy，在不增加推理成本的情况下提升分割性能。

## 十六、BiSeNet v2 的 Booster Training Strategy

BiSeNet v2 还提出了 booster training strategy。

它的核心思想是：

训练时增加额外监督或增强模块
推理时不增加额外计算

这类策略有点类似 YOLO 系列中的：

Bag of Freebies

也就是：

训练阶段帮助模型学得更好
推理阶段保持轻量和高速

BiSeNet v2 论文中提到 booster training strategy 可以提升分割性能，而且不会带来额外推理成本。

## 十七、BiSeNet v2 的性能特点

BiSeNet v2 相比 v1 更工程化，更强调实时语义分割中的速度-精度平衡。

论文报告中，BiSeNet v2 在 Cityscapes 2048×1024 输入上达到 72.6% mIoU，并在单张 NVIDIA GTX 1080 Ti 上达到 156 FPS。

这说明它相比 v1 在精度和速度上都有明显提升。

## 十八、Rethinking BiSeNet / STDC：对双路径结构的反思

后续有一篇很重要的工作叫：

Rethinking BiSeNet For Real-time Semantic Segmentation

它提出了 STDC Network。

这篇论文认为，BiSeNet 的额外空间路径虽然能保留细节，但也会带来时间开销；而直接借用分类任务 backbone 也不一定最适合语义分割。于是它提出了更高效的 STDC 结构，并通过 Detail Aggregation Module 在单路径结构中学习细节信息。

STDC 的核心思路是：

不再额外增加完整 Spatial Path
而是在主干网络中更高效地聚合短程和细节特征

它的设计目标是：

减少冗余
提升速度
保持较好分割精度

论文报告中，STDC 方法在 Cityscapes 上达到 71.9% mIoU / 250.4 FPS，或者在更高分辨率推理下达到 76.8% mIoU / 97.0 FPS，体现了更强的速度-精度权衡。

需要注意：很多资料会把 STDC 和 BiSeNet 后续改进放在一起讨论，但它的论文标题是 “Rethinking BiSeNet”，并不完全等同于原始 BiSeNet v1/v2 的直接版本号延续。

## 十九、BiSeNet V3 相关说明

也有论文提出 BiSeNet V3，继续围绕实时语义分割中的空间信息和感受野问题进行改进。相关论文将其描述为基于 BiSeNet 的新架构，目标仍然是在实时推理速度下尽量保留空间细节和提升分割精度。

不过在工程和论文引用中，更常见的主线通常是：

BiSeNet v1
BiSeNet v2
STDC / Rethinking BiSeNet

如果你是面试或项目介绍，重点掌握这三者就已经足够。

## 二十、BiSeNet 和 DeepLab 的区别
对比项	BiSeNet	DeepLab
任务	语义分割	语义分割
目标	实时分割	高质量语义分割
核心思想	双路径：细节 + 语义	空洞卷积 + ASPP
多尺度上下文	Context / Semantic Branch	ASPP 明确建模
边界细节	Spatial / Detail Branch 保留	DeepLabv3+ 用 Decoder 改善
速度	更偏实时	通常更重
典型场景	自动驾驶、实时视频、边缘设备	城市场景、高精度语义分割

简单说：

DeepLab：
    更重视多尺度上下文和分割精度

BiSeNet：
    更重视实时速度和空间细节保留

如果你要做实时道路分割，BiSeNet 很合适。

如果你更关注高精度 benchmark，DeepLabv3+ 可能更强。

## 二十一、BiSeNet 和 U-Net 的区别
对比项	BiSeNet	U-Net
主要场景	实时语义分割	医学/缺陷/小数据分割
结构	双路径实时结构	Encoder-Decoder + Skip Connection
速度	通常更快	取决于 backbone，标准 U-Net 不一定极致实时
细节恢复	Spatial/Detail Branch	Skip Connection
语义上下文	Context/Semantic Branch	Bottleneck + Encoder 深层特征
适合数据	城市场景、视频、实时任务	医学图像、工业缺陷、小样本

简单理解：

U-Net：
    通过编码器-解码器和跳跃连接恢复细节

BiSeNet：
    直接分出一条细节分支和一条语义分支，再高效融合

如果你的项目是裂缝、病灶、小缺陷，U-Net 往往更直接。

如果你的项目是自动驾驶实时分割、机器人实时视觉，BiSeNet 更有针对性。

## 二十二、BiSeNet 和 FCN 的区别

FCN 是早期全卷积分割网络，核心思路是：

把分类网络改成全卷积网络
再上采样恢复分割结果

BiSeNet 相比 FCN 的改进在于：

明确拆分空间细节和语义上下文
专门为实时分割设计
使用更高效的双路径或双分支结构
加入特征融合和注意力细化模块

FCN 更像是语义分割基础框架。

BiSeNet 更像是面向实时场景优化的语义分割网络。

## 二十三、BiSeNet 和 Mask R-CNN 的区别

BiSeNet 是语义分割模型。

Mask R-CNN 是实例分割模型。

对比项	BiSeNet	Mask R-CNN
任务	语义分割	实例分割
输出	每个像素类别	每个实例的 bbox + mask
是否区分同类实例	不区分	区分
速度	更适合实时	通常较慢
结构	双路径分割网络	Faster R-CNN + Mask Head
应用	道路、天空、建筑、车道区域	人、车、细胞、物体实例

例如图中有 3 个人：

BiSeNet：
    把所有人像素都预测成 person

Mask R-CNN：
    分别输出 person_1、person_2、person_3 的 mask

所以如果你要区分每一个独立目标，用 Mask R-CNN；如果只需要每个像素的类别，用 BiSeNet。

## 二十四、BiSeNet 的训练流程

训练 BiSeNet 和普通语义分割模型类似。

流程是：

1. 准备 image 和 mask
2. 划分 train / val / test
3. 对 image 和 mask 做同步增强
4. 输入图像
5. 模型输出 segmentation logits
6. 计算主损失和辅助损失
7. 反向传播
8. 验证集计算 mIoU / Pixel Acc
9. 保存最优模型

常见数据增强：

随机缩放
随机裁剪
随机水平翻转
颜色扰动
模糊
归一化

注意：

image 和 mask 的几何变换必须同步
mask resize 必须使用最近邻插值
颜色增强只作用于 image

## 二十五、BiSeNet 常用损失函数

BiSeNet 是语义分割模型，常用损失包括：

CrossEntropy Loss
OHEM CrossEntropy Loss
Dice Loss
Focal Loss
CE + Dice
Auxiliary Loss
### 1. CrossEntropy Loss

多类别语义分割最常用：

criterion = nn.CrossEntropyLoss()

输入：

outputs: [B, C, H, W]
masks:   [B, H, W]

标签像素值是：

0 ~ C-1
### 2. OHEM CrossEntropy

实时分割和场景分割中常用 OHEM：

Online Hard Example Mining

它会更关注难分类像素。

例如：

道路和人行道边界
远处小行人
交通标志
细小杆状物

这些像素比大片天空、道路更难，OHEM 可以让训练更关注它们。

### 3. Auxiliary Loss

BiSeNet 中经常会有辅助输出。

例如：

主分割输出：main loss
中间语义输出：aux loss

总损失：

Loss = L_main + λ1 * L_aux1 + λ2 * L_aux2

辅助 loss 可以帮助中间层学习更有效的语义特征。

## 二十六、BiSeNet 的评价指标

语义分割常用指标包括：

mIoU
Pixel Accuracy
Mean Pixel Accuracy
FPS
Latency
Params
FLOPs

实时分割特别关注：

mIoU + FPS

因为只看 mIoU 不够。

例如：

模型 A：80 mIoU，10 FPS
模型 B：75 mIoU，120 FPS

如果是自动驾驶实时系统，模型 B 可能更实用。

BiSeNet v1/v2 论文都在 Cityscapes 等数据集上同时报告分割精度和推理速度，以体现实时语义分割中的速度-精度平衡。

## 二十七、BiSeNet 适合什么场景？

BiSeNet 适合：

实时语义分割
自动驾驶道路分割
视频流场景解析
机器人视觉
移动端/边缘端分割
城市街景分割
交通场景分割
工业实时区域分割

例如：

道路 / 车辆 / 行人 / 天空 / 建筑物分割
机器人地面可通行区域分割
实时背景区域分割
实时缺陷区域粗分割

它的优势是：

速度快
结构相对轻
空间细节和语义上下文分工明确
适合实时部署
二十八、BiSeNet 不太适合什么场景？

如果任务非常重视边界极致精度，比如：

医学病灶精细边界
微小裂缝
细胞边界
高精度工业缺陷轮廓

BiSeNet 不一定是首选，U-Net++、Attention U-Net、DeepLabv3+、SegFormer、Mask2Former 等可能更合适。

如果任务需要区分不同实例：

每个人单独分割
每辆车单独分割
每个细胞单独分割

BiSeNet 也不适合，因为它是语义分割，不是实例分割。

## 二十九、实际项目中如何改进 BiSeNet？

可以从几个方向改进。

### 1. 换轻量 backbone

例如：

MobileNetV2
MobileNetV3
ShuffleNet
STDCNet
EfficientNet-lite

目标是提升速度，降低参数量。

### 2. 加注意力机制

可以加入：

SE
CBAM
Coordinate Attention
ECA
Spatial Attention

用于增强重要通道和空间区域。

### 3. 改进特征融合模块

FFM / GAL 是 BiSeNet 的关键。

可以尝试：

更轻量的融合模块
门控融合
注意力融合
多尺度融合
边界引导融合
### 4. 边界辅助监督

如果任务边界很重要，可以加入：

Boundary Loss
Edge Loss
Detail Aggregation
边缘分支

STDC 的 Rethinking BiSeNet 工作就通过 Detail Aggregation 思路，把细节学习融入更高效的结构中。

### 5. 蒸馏

实时模型常用知识蒸馏：

Teacher：大模型，比如 DeepLabv3+ / SegFormer
Student：BiSeNet

让 BiSeNet 学习大模型的 soft logits 或中间特征。

这样可以在不明显增加推理成本的情况下提高精度。

## 三十一、面试中如何介绍 BiSeNet？

如果面试官问：

你了解 BiSeNet 吗？

可以这样回答：

BiSeNet 是一种面向实时语义分割的双路径网络。它的核心思想是将空间细节信息和语义上下文信息分开建模，再进行融合。BiSeNet v1 中有 Spatial Path 和 Context Path，Spatial Path 使用较浅的卷积结构保留高分辨率空间细节，Context Path 使用快速下采样的轻量 backbone 获取大感受野和高级语义信息。之后通过 Attention Refinement Module 对上下文特征进行细化，再用 Feature Fusion Module 融合空间特征和语义特征，最终输出像素级分割结果。

BiSeNet v2 进一步将这种思想改进为 Detail Branch 和 Semantic Branch，前者用浅层宽通道结构捕获低级细节，后者用窄通道深层结构快速提取语义上下文，并通过 Guided Aggregation Layer 融合两路特征。整体来看，BiSeNet 的优势是速度快、结构清晰，适合自动驾驶、视频场景解析、机器人视觉等实时语义分割任务。

## 三十二、如果面试官追问：BiSeNet 为什么适合实时分割？

可以回答：

因为 BiSeNet 没有用一个很重的网络同时承担细节保留和语义理解，而是把任务拆成两个轻量分支。空间或细节分支负责保留高分辨率边界信息，语义或上下文分支通过快速下采样获得大感受野和高级语义，然后用轻量融合模块结合两者。这样既避免了保持高分辨率带来的巨大计算量，也避免了过度下采样导致的边界丢失，所以能在速度和精度之间取得较好平衡。

## 三十三、如果面试官追问：BiSeNet v1 和 v2 有什么区别？

可以回答：

BiSeNet v1 使用 Spatial Path 和 Context Path。Spatial Path 通过几层卷积保留空间细节，Context Path 使用轻量 backbone 快速下采样获取语义上下文，并通过 ARM 和 FFM 融合特征。

BiSeNet v2 则进一步重新设计了双边结构，使用 Detail Branch 和 Semantic Branch。Detail Branch 是浅层、宽通道结构，专门捕获低级细节；Semantic Branch 是深层、窄通道结构，通过快速下采样获得语义上下文。两者通过 Guided Aggregation Layer 融合。相比 v1，v2 更强调高效性和速度-精度平衡。

## 三十四、如果面试官追问：BiSeNet 和 DeepLab 怎么选？

可以回答：

如果任务更重视实时性，比如自动驾驶视频流、机器人视觉或边缘设备部署，我会优先考虑 BiSeNet，因为它通过双路径结构在速度和分割效果之间做了比较好的权衡。

如果任务更重视语义分割精度，尤其是自然场景或城市街景分割，且对速度要求没那么极端，可以考虑 DeepLabv3+。DeepLab 的空洞卷积和 ASPP 对多尺度上下文建模更强，而 BiSeNet 更强调实时性和轻量化。