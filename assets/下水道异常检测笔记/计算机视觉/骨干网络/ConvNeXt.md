# ConvNeXt

ConvNeXt 是一种“现代化 CNN”架构，它用纯卷积网络吸收 Vision Transformer / Swin Transformer 中有效的设计思想，让 CNN 在分类、检测、分割任务中重新具备很强竞争力。

它来自 2022 年论文 《A ConvNet for the 2020s》。论文的核心不是提出一个完全新奇的算子，而是系统地研究：如果把 ResNet 按照现代视觉模型的经验重新设计，纯 CNN 能不能追上甚至超过 Transformer？ 结果就是 ConvNeXt。论文报告 ConvNeXt 作为纯 ConvNet，在 ImageNet 上最高达到 87.8% Top-1，并在 COCO 检测和 ADE20K 分割上表现优于 Swin Transformer，同时保持标准卷积网络的简洁性和效率。

## 一、ConvNeXt 是什么？

ConvNeXt 可以拆开理解：

Conv = Convolution，卷积
NeXt = Next generation，下一代

所以它可以理解为：

新一代卷积网络

它本质上仍然是 CNN，但不是传统 ResNet 那种经典 CNN，而是经过现代化改造的 CNN。

它的目标是：

保留 CNN 的优势：
    局部建模
    平移等变性
    结构简单
    推理友好
    工程部署成熟

吸收 Transformer 的优势：
    更合理的层级设计
    更大的感受野
    更强的可扩展性
    更好的训练策略

一句话总结：

ConvNeXt = 用 Transformer 时代的设计经验重新设计的 ResNet 风格纯卷积网络

## 二、ConvNeXt 解决了什么问题？

在 ViT、Swin Transformer 出现后，视觉任务中 Transformer 变得非常流行。

很多人认为：

Transformer 比 CNN 更先进
CNN 已经过时

但 ConvNeXt 的问题意识是：

Transformer 的强大，真的全部来自 self-attention 吗？还是很多性能提升来自训练策略、网络尺度设计、归一化方式、下采样结构、宏观设计等“工程设计”？

于是 ConvNeXt 作者从一个标准 ResNet 出发，逐步引入现代视觉模型中的设计习惯，最后发现：

即使不使用 self-attention
只用卷积
也能达到非常强的性能

这说明 CNN 并没有过时，而是传统 ResNet 的设计有些地方已经不适合现代训练规模和任务需求。

## 三、ConvNeXt 和 ResNet 的关系

ConvNeXt 可以看成是：

ResNet 的现代化版本

ResNet 的核心是：

残差连接
卷积块
分阶段下采样
Global Average Pooling
分类头

ConvNeXt 仍然保留这些大方向：

分 stage 的层级结构
卷积特征提取
残差连接
下采样形成多尺度特征

但 ConvNeXt 对 ResNet 做了很多关键改造：

1. 改变 stage 堆叠比例
2. 使用 patchify stem
3. 使用 depthwise convolution
4. 使用更大卷积核，例如 7×7
5. 使用 LayerNorm 替代 BatchNorm
6. 使用 GELU 替代 ReLU
7. 使用 inverted bottleneck
8. 减少激活函数和归一化层数量
9. 使用 Layer Scale
10. 使用更现代的训练策略

所以它既不像传统 ResNet，也不是 Transformer，而是介于二者设计思想之间的现代 CNN。

## 四、ConvNeXt 的整体结构

ConvNeXt 的整体结构仍然是四个 stage。

以输入图像 224×224 为例：

输入图像：224 × 224 × 3
↓
Patchify Stem，4×4 Conv stride=4
↓
Stage 1：56 × 56
↓
Downsample
↓
Stage 2：28 × 28
↓
Downsample
↓
Stage 3：14 × 14
↓
Downsample
↓
Stage 4：7 × 7
↓
Global Average Pooling
↓
Linear Classifier
↓
输出类别

这和 Swin Transformer 的层级结构非常接近：

56×56 → 28×28 → 14×14 → 7×7

而传统 ResNet 的开头通常是：

7×7 Conv stride=2
↓
MaxPool stride=2
↓
56×56

ConvNeXt 则把开头改成更像 ViT/Swin 的 patch embedding：

4×4 Conv stride=4

这一步相当于直接把图像划分成 4×4 patch，并映射到特征通道。

## 、ConvNeXt 的核心模块

ConvNeXt block 是它最核心的部分。

一个典型 ConvNeXt block 可以写成：

输入 x
↓
7×7 Depthwise Conv
↓
LayerNorm
↓
1×1 Conv，通道扩展 4 倍
↓
GELU
↓
1×1 Conv，通道还原
↓
Layer Scale
↓
DropPath
↓
残差连接
↓
输出

可以看到，它仍然是卷积网络，但结构已经很像 Transformer block：

Transformer:
    Token mixing
    LayerNorm
    MLP
    Residual

ConvNeXt:
    Depthwise Conv 做空间混合
    LayerNorm
    1×1 Conv MLP 做通道混合
    Residual

也就是说：

ConvNeXt 用 depthwise convolution 替代 self-attention 来做空间信息交互
用 1×1 convolution 替代 MLP 做通道变换

## 六、Depthwise Convolution 是什么？

ConvNeXt 大量使用 Depthwise Convolution。

普通卷积会同时做两件事：

1. 空间维度上的特征提取
2. 通道之间的信息融合

Depthwise Convolution 则只做空间维度上的卷积：

每个通道单独卷积
不同通道之间不混合

假设输入有 96 个通道，普通 7×7 Conv 会在所有输入通道和输出通道之间做卷积，计算量很大。

Depthwise 7×7 Conv 是：

第 1 个通道用自己的 7×7 卷积核
第 2 个通道用自己的 7×7 卷积核
...
第 96 个通道用自己的 7×7 卷积核

它不负责通道融合。

通道融合交给后面的：

1×1 Conv

这种设计类似 MobileNet 中的深度可分离卷积思想，但 ConvNeXt 用它来构建强大的大规模 backbone。

## 七、为什么 ConvNeXt 使用 7×7 大卷积核？

传统 ResNet 中最常见的是：

3×3 Conv

ConvNeXt 使用：

7×7 Depthwise Conv

原因是：

Transformer 的 self-attention 能建模较大范围的信息
传统 3×3 卷积局部性太强
大卷积核可以扩大感受野

但如果直接用普通 7×7 Conv，计算量太大。

所以 ConvNeXt 用：

7×7 Depthwise Conv

这样既扩大了空间感受野，又控制了计算量。

可以理解为：

3×3 卷积：
    看得近

7×7 depthwise 卷积：
    看得更远，但计算仍然可控

这也是 ConvNeXt 向 Transformer 学习的一个关键点：

增强空间建模范围

## 八、Patchify Stem 是什么？

传统 ResNet 的 stem 是：

7×7 Conv stride=2
↓
MaxPool stride=2

ConvNeXt 改成：

4×4 Conv stride=4

这叫：

Patchify Stem

它类似 ViT/Swin 的 patch embedding。

输入：

224 × 224 × 3

经过 4×4 Conv stride=4 后：

56 × 56 × C

这一步相当于：

把图像切成 4×4 patch
并投影成 C 维特征

为什么这样做？

因为现代视觉 Transformer 通常从 patch embedding 开始，而不是传统 CNN 那种大卷积 + 池化的开头。

ConvNeXt 发现这种 stem 更简洁，也更适合现代架构

## 九、Stage 堆叠比例的改变

传统 ResNet-50 的 block 分布是：

3, 4, 6, 3

也就是第三个 stage 最深，但整体比例比较传统。

Swin Transformer 的 stage 分布通常类似：

2, 2, 6, 2

或者大模型中第三 stage 更重。

ConvNeXt 采用类似的设计，把更多计算放在中后期 stage，尤其是第三 stage。

例如 ConvNeXt-Tiny 的 block 分布是：

3, 3, 9, 3

ConvNeXt-Small / Base 等也会在第三 stage 堆更多 block。

这样做的直觉是：

早期特征图分辨率高，计算成本大
后期特征图分辨率低，堆更多层更划算
中后期语义特征更强，增加深度收益更高

## 十、Inverted Bottleneck 是什么？

ResNet bottleneck 是：

1×1 降维
3×3 卷积
1×1 升维

例如：

256 → 64 → 64 → 256

它先压缩通道，再做空间卷积。

而 ConvNeXt 使用的是类似 MobileNetV2 / Transformer MLP 的 inverted bottleneck：

先扩展通道
再压回通道

例如：

96 → 384 → 96

结构是：

Depthwise Conv
↓
1×1 Conv 扩展 4 倍通道
↓
GELU
↓
1×1 Conv 压回原通道

这很像 Transformer 中的 FFN：

d_model → 4d_model → d_model

所以 ConvNeXt block 的通道 MLP 部分可以理解为：

CNN 版 Transformer FFN

## 十一、为什么 ConvNeXt 使用 LayerNorm？

传统 CNN 常用：

BatchNorm

Transformer 常用：

LayerNorm

ConvNeXt 把 BatchNorm 替换为 LayerNorm。

BatchNorm 的特点是：

依赖 batch 统计量
在大 batch 训练时效果好
小 batch 时可能不稳定

LayerNorm 的特点是：

不依赖 batch 维度
对每个样本自身做归一化
和 Transformer 架构习惯一致

ConvNeXt 使用 LayerNorm 后，网络风格更接近 Transformer。

不过这也带来一些工程实现细节：

CNN 特征常用 [B, C, H, W]
LayerNorm 常在最后一维做归一化
所以有时会转换成 [B, H, W, C]

在代码里你会看到 ConvNeXt block 中经常有：

x = x.permute(0, 2, 3, 1)
LayerNorm
x = x.permute(0, 3, 1, 2)

或者使用支持 channels_first 的 LayerNorm 实现。

## 十二、为什么 ConvNeXt 使用 GELU？

传统 CNN 常用：

ReLU

Transformer 常用：

GELU

ConvNeXt 使用 GELU，原因是它继承了 Transformer 的 MLP 设计习惯。

GELU 比 ReLU 更平滑：

ReLU:
    x > 0 保留
    x <= 0 置 0

GELU:
    根据输入大小平滑地控制通过比例

GELU 在 Transformer 中非常常见，ConvNeXt 使用它也是现代化设计的一部分。

## 十三、ConvNeXt 为什么减少激活和归一化？

传统 ResNet block 中有多个：

Conv-BN-ReLU
Conv-BN-ReLU
Conv-BN

ConvNeXt 则更像 Transformer block：

Depthwise Conv
LayerNorm
MLP
GELU
MLP
Residual

它减少了中间不必要的激活和归一化层。

这样做的意义是：

简化结构
降低额外开销
更接近 Transformer block 的设计
可能改善特征流动

也就是说，ConvNeXt 不是简单把 ResNet 加宽加深，而是重新审视了每个 block 内部的微观结构。

## 十四、Layer Scale 是什么？

Layer Scale 是一种给残差分支输出乘上一个可学习的小系数的技巧。

形式大致是：

y = x + γ * F(x)

其中：

γ 是可学习参数
初始值很小，比如 1e-6

这样做的作用是：

训练初期让残差分支影响较小
让网络更稳定
随着训练逐渐学习合适的残差幅度

尤其是深层网络中，Layer Scale 有助于稳定训练。

## 十五、DropPath 是什么？

DropPath 也叫：

Stochastic Depth

它是一种正则化方法。

训练时随机丢弃某些残差分支：

y = x + F(x)

有时变成：

y = x

也就是让这个 block 暂时只走 shortcut。

它的作用是：

增强模型泛化能力
降低过拟合
训练更深网络时更稳定

推理时不会随机丢弃，而是使用完整网络。

DropPath 在 Transformer 和现代 CNN 中都很常见。

## 十六、ConvNeXt 的不同规模

ConvNeXt 有多个模型规模。

常见包括：

ConvNeXt-Tiny
ConvNeXt-Small
ConvNeXt-Base
ConvNeXt-Large
ConvNeXt-XLarge

可以简单理解为：

Tiny：轻量，速度快
Small：中等规模
Base：标准较大模型
Large：更高精度
XLarge：更大规模，训练和推理成本更高

它们主要区别在：

通道数不同
每个 stage 的 block 数量不同
模型参数量不同
计算量不同

常见 stage block 设计类似：

Tiny:  3, 3, 9, 3
Small: 3, 3, 27, 3
Base:  3, 3, 27, 3
Large: 3, 3, 27, 3

Small、Base、Large 的差异更多在通道宽度。

## 十七、ConvNeXt 和 ResNet 的区别
对比项	ResNet	ConvNeXt
基础思想	残差 CNN	现代化残差 CNN
Stem	7×7 Conv + MaxPool	4×4 Conv stride=4 patchify stem
核心卷积	3×3 Conv	7×7 Depthwise Conv
Block	BasicBlock / Bottleneck	Depthwise Conv + MLP-like 1×1 Conv
归一化	BatchNorm	LayerNorm
激活	ReLU	GELU
通道结构	Bottleneck 常先降维	Inverted bottleneck 先升维
设计风格	经典 CNN	借鉴 Transformer 的现代 CNN
性能	稳定经典	通常更强，更适合现代任务

一句话：

ResNet 是经典残差 CNN
ConvNeXt 是吸收 Transformer 设计经验后的现代残差 CNN

## 十八、ConvNeXt 和 Swin Transformer 的关系

ConvNeXt 的很多设计都是对照 Swin Transformer 来做的。

Swin Transformer 的特点：

Patch embedding
层级结构
窗口注意力
MLP
LayerNorm
GELU
DropPath

ConvNeXt 借鉴了其中很多非 attention 的设计：

Patchify stem
层级 stage
更合理的 block 分布
LayerNorm
GELU
MLP-like inverted bottleneck
DropPath
大感受野

但 ConvNeXt 不使用 self-attention，而是使用：

7×7 depthwise convolution

来做空间信息交互。

所以二者可以对比为：

Swin：
    Window Attention 做空间混合

ConvNeXt：
    Large-kernel Depthwise Conv 做空间混合

ConvNeXt 的意义在于说明：

Transformer 的成功不一定全部来自 attention
现代化网络设计和训练策略同样非常重要

## 十九、ConvNeXt 和 Vision Transformer 的区别
对比项	ConvNeXt	ViT
核心操作	卷积	Self-Attention
结构	层级 CNN	Patch token Transformer
空间归纳偏置	强	相对弱
局部建模	天然擅长	依赖 attention 学习
全局建模	通过大卷积核和深层堆叠	直接全局 attention
小数据表现	通常更稳	更依赖大规模预训练
部署	CNN 生态成熟	依赖 attention 优化
检测/分割适配	很自然	原始 ViT 需改造，Swin 更适合

简单说：

ConvNeXt 是 Transformer 时代的强 CNN
ViT 是把图像当 token 序列处理的 Transformer

## 二十、ConvNeXt 和 EfficientNet 的区别

EfficientNet 的核心是：

复合缩放：
    depth
    width
    resolution

并大量使用：

MBConv
SE
Swish/SiLU

ConvNeXt 的核心是：

重新设计 ResNet，使其接近现代 Transformer 风格

对比：

对比项	EfficientNet	ConvNeXt
主要思想	复合缩放搜索	手工现代化 ResNet
核心块	MBConv + SE	7×7 DWConv + 1×1 MLP
设计来源	NAS / MobileNet 系列	ResNet + Transformer 设计经验
优势	参数效率高	通用 backbone 能力强
典型用途	分类、轻量模型	分类、检测、分割 backbone

如果你想做轻量分类，EfficientNet 很合适。

如果你想做强 backbone，用于分类、检测、分割，ConvNeXt 很有竞争力。

## 二十一、ConvNeXt 在图像分类中的使用

图像分类中，ConvNeXt 结构是：

输入图像
↓
ConvNeXt backbone
↓
Global Average Pooling
↓
LayerNorm
↓
Linear classifier
↓
类别 logits

输出：

[B, num_classes]

如果你有一个 5 分类任务，可以把最后分类头改成：

model.classifier[-1] = nn.Linear(in_features, 5)

或者根据具体库实现修改 head。

ConvNeXt 很适合作为分类任务的强 baseline：

ConvNeXt-Tiny：较轻量
ConvNeXt-Base：更强
ConvNeXt-Large：高精度但更重

## 二十二、ConvNeXt 在目标检测中的使用

目标检测中，ConvNeXt 通常作为 backbone。

例如：

Mask R-CNN + ConvNeXt
Cascade Mask R-CNN + ConvNeXt
RetinaNet + ConvNeXt
DETR / DINO + ConvNeXt

它输出多尺度特征：

C1 / C2 / C3 / C4

再接 FPN：

P2 / P3 / P4 / P5

用于检测不同大小的目标。

相比 ResNet，ConvNeXt 的特征表达能力通常更强，所以在检测任务中可以作为更强 backbone。

原始 ConvNeXt 论文也报告其不仅在 ImageNet 上表现强，在 COCO 目标检测和 ADE20K 语义分割等下游任务中也能取得优于 Swin Transformer 的结果。

## 二十三、ConvNeXt 在语义分割中的使用

语义分割中，ConvNeXt 可以作为 encoder/backbone。

常见组合：

UPerNet + ConvNeXt
DeepLab + ConvNeXt
Semantic FPN + ConvNeXt
Mask2Former + ConvNeXt

流程：

输入图像
↓
ConvNeXt 提取多尺度特征
↓
分割 decoder 融合特征
↓
输出像素级 mask

ConvNeXt 的优势是：

层级特征天然适合分割
大卷积核有助于更大感受野
CNN 对局部边界和纹理建模较稳
特征表达比传统 ResNet 更强

## 二十四、ConvNeXt V2 是什么？

ConvNeXt V2 是 ConvNeXt 的后续改进版本，论文名是 《ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders》。它主要解决的问题是：ConvNeXt 原本是为有监督训练设计的，如果直接结合 MAE 这类自监督预训练，效果并不理想。因此 ConvNeXt V2 提出了 FCMAE 和 GRN。

ConvNeXt V2 的两个关键词：

1. FCMAE：Fully Convolutional Masked Autoencoder
2. GRN：Global Response Normalization

它的目标是：

让纯 ConvNet 更好地适配 masked autoencoder 自监督训练
提升分类、检测、分割等任务表现

论文报告 ConvNeXt V2 在 ImageNet、COCO detection、ADE20K segmentation 等多个 benchmark 上提升了纯 ConvNet 的性能，并提供从 Atto 到 Huge 的多个规模模型。

## 二十五、GRN 是什么？

GRN 全称：

Global Response Normalization

中文可以理解为：

全局响应归一化

它是 ConvNeXt V2 中加入的新层。

它的核心作用是：

增强通道之间的竞争和响应校准

为什么需要它？

在 masked autoencoder 预训练中，模型需要从部分可见区域恢复被遮挡区域。作者发现 ConvNeXt 在这种训练方式下可能出现特征表达不够充分的问题，于是引入 GRN 来增强不同通道之间的响应对比。

直观理解：

普通特征：
    各通道可能学习得比较平均或冗余

GRN：
    根据全局响应强度重新调整通道
    让重要通道更突出
    让通道之间形成更有效的竞争

这有点类似一种轻量的通道响应调制。

## 二十六、ConvNeXt 的优点

ConvNeXt 的优点主要有：

1. 保留 CNN 的简洁和高效
2. 纯卷积结构，工程部署友好
3. 性能接近或超过很多 Transformer backbone
4. 层级结构天然适合检测和分割
5. 比传统 ResNet 表达能力更强
6. 适合分类、检测、语义分割、实例分割等任务
7. 可作为 ResNet 的强替代 backbone

尤其是在你做视觉项目时，ConvNeXt 是一个非常好的升级选项：

ResNet baseline
↓
ConvNeXt stronger backbone

## 二十七、ConvNeXt 的缺点

ConvNeXt 也不是万能的。

1. 没有显式全局 attention

虽然大卷积核扩大了感受野，但它本质上还是 CNN。

如果任务非常依赖远距离全局关系，ViT、Swin、Mask2Former 等 Transformer 结构可能仍有优势。

2. 大模型计算成本高

ConvNeXt-Large、XLarge 很强，但也很重。

如果是移动端、低算力设备，可能不如：

MobileNet
ShuffleNet
EfficientNet-lite
BiSeNet
PP-LiteSeg

这些轻量模型合适。

3. 对小数据任务未必一定优于专用结构

如果你的任务是医学图像小样本分割，标准 U-Net / Attention U-Net / nnU-Net 可能更直接。

ConvNeXt 作为 encoder 可以尝试，但不一定天然最优。

4. LayerNorm 在 CNN 部署中可能有额外适配成本

传统 CNN 部署通常对 Conv + BN 融合优化很好。

ConvNeXt 使用 LayerNorm，在某些推理框架或硬件上可能不像 Conv-BN-ReLU 那样容易融合优化。

不过主流框架已经普遍支持。

## 二十八、实际项目中什么时候选 ConvNeXt？
适合选择 ConvNeXt 的场景
图像分类想要比 ResNet 更强的 baseline
目标检测想换更强 backbone
语义分割想提升 encoder 表达能力
有较好 GPU 资源
不追求极限轻量化
希望兼顾 CNN 工程友好和 Transformer 时代性能

例如：

工业缺陷分类
医学图像分类
遥感图像分类
目标检测 backbone
分割 encoder
多类别图像识别
不一定首选 ConvNeXt 的场景
移动端极限实时推理
非常小的数据集且模型容易过拟合
需要极强开放世界泛化
需要文本-图像多模态理解

这些场景可能优先考虑：

MobileNet / EfficientNet-lite
U-Net / nnU-Net
CLIP / SAM / GroundingDINO
ViT / Swin / SegFormer

## 三十、面试中如何介绍 ConvNeXt？

如果面试官问：

你了解 ConvNeXt 吗？

可以这样回答：

ConvNeXt 是一种现代化的卷积神经网络，可以看作是对 ResNet 的重新设计。它的核心思想是吸收 Vision Transformer 和 Swin Transformer 中有效的架构设计，但仍然保持纯卷积结构。ConvNeXt 从 ResNet 出发，逐步引入 patchify stem、层级 stage 设计、depthwise convolution、大卷积核、LayerNorm、GELU、inverted bottleneck、Layer Scale 和 DropPath 等现代设计，使 CNN 在分类、检测和分割任务中重新达到非常强的性能。

ConvNeXt 的基本 block 通常由 7×7 depthwise convolution、LayerNorm、两个 1×1 convolution 构成的 MLP-like 模块、GELU、Layer Scale 和残差连接组成。其中 depthwise convolution 负责空间信息混合，1×1 convolution 负责通道信息融合。相比传统 ResNet 的 3×3 卷积和 BatchNorm-ReLU 结构，ConvNeXt 更接近 Transformer block 的设计风格，但没有使用 self-attention。

它的优势是保留了 CNN 的局部归纳偏置和工程部署友好性，同时具备接近 Transformer 的精度和可扩展性，因此常被用作图像分类、目标检测和语义分割任务中的强 backbone。

## 三十一、如果面试官追问：ConvNeXt 和 ResNet 有什么区别？

可以回答：

ResNet 是经典残差 CNN，主要由 3×3 卷积、BatchNorm、ReLU 和残差连接组成；ConvNeXt 则是在 ResNet 基础上吸收了 Transformer 时代的一些设计。比如它用 4×4 stride=4 的 patchify stem 替代 ResNet 的 7×7 卷积加 MaxPool；用 7×7 depthwise convolution 扩大感受野；用 LayerNorm 替代 BatchNorm；用 GELU 替代 ReLU；并采用类似 Transformer FFN 的 inverted bottleneck 结构。整体来说，ConvNeXt 还是 CNN，但它的设计风格更现代，性能通常强于传统 ResNet。

## 三十二、如果面试官追问：ConvNeXt 为什么用 Depthwise Conv？

可以回答：

Depthwise convolution 可以把空间特征提取和通道融合解耦。ConvNeXt 使用 7×7 depthwise convolution 来做空间信息混合，这样可以扩大感受野，但计算量不会像普通 7×7 卷积那样大。随后再用 1×1 convolution 做通道融合和通道扩展。这种设计既保留了卷积的局部建模能力，又能以较低成本获得更大的空间感受野。

## 三十三、如果面试官追问：ConvNeXt 为什么说是 CNN 版 Transformer？

可以回答：

因为 ConvNeXt 的 block 结构和 Transformer block 很像。Transformer block 通常是 token mixing，也就是 self-attention，加上 LayerNorm 和 MLP；ConvNeXt 则用 depthwise convolution 做空间 mixing，用 1×1 convolution 构成 MLP-like 通道变换，并同样使用 LayerNorm、GELU、残差连接和 DropPath。所以它在整体设计习惯上接近 Transformer，但核心空间建模操作仍然是卷积，而不是 self-attention。

## 三十四、如果面试官追问：ConvNeXt V2 改进了什么？

可以回答：

ConvNeXt V2 主要面向自监督预训练场景进行改进。原始 ConvNeXt 是在有监督训练下设计的，如果直接和 masked autoencoder 结合，效果并不理想。ConvNeXt V2 提出了 Fully Convolutional Masked Autoencoder，也就是 FCMAE，并加入 Global Response Normalization，简称 GRN，用来增强通道之间的特征竞争和全局响应建模。这样可以让 ConvNeXt 更好地适配 MAE 式自监督预训练，并提升分类、检测和分割等下游任务表现。