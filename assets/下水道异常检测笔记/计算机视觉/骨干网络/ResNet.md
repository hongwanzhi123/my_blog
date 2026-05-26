# ResNet
它是深度学习和计算机视觉里非常重要的经典网络，很多后续模型都直接或间接受到它影响。可以把它理解为：

ResNet 是一种通过“残差连接”解决深层神经网络难训练问题的卷积神经网络结构。

## 一、ResNet 是什么？

ResNet 全称是：

Residual Network

中文通常叫：

残差网络

它是一个经典 CNN 网络，主要用于图像分类，也常作为目标检测、语义分割、实例分割等任务的 backbone。

例如很多模型会用：

Faster R-CNN + ResNet
Mask R-CNN + ResNet-FPN
DeepLabv3+ + ResNet
U-Net + ResNet Encoder
RT-DETR + ResNet

ResNet 最核心的创新是：

Residual Connection / Skip Connection

也就是：

残差连接 / 跳跃连接

它让深层网络更容易训练，使得网络可以从几十层扩展到上百层甚至更深。

## 二、ResNet 解决了什么问题？

在 ResNet 之前，人们发现一个现象：

神经网络变深后，理论上表达能力应该更强，但实际训练效果反而可能变差。

这不是简单的过拟合问题，而是退化问题 degradation problem。

比如一个 20 层网络和一个 56 层网络：

理论上：
56 层网络至少应该不比 20 层差

实际中：
56 层网络训练误差反而更高

这说明问题不是测试集泛化差，而是：

深层网络本身训练不动
优化变困难
训练误差都降不下来

常见原因包括：

梯度传播困难
深层网络优化难度大
信息在多层传递中逐渐衰减
网络难以学习恒等映射

ResNet 就是为了解决这个问题提出的。

## 三、普通网络在学什么？

普通神经网络的一层或一个模块可以理解为学习一个映射：

H(x)

其中：

x 是输入
H(x) 是网络希望学到的目标映射

也就是说，普通网络直接学习：

输入 x → 输出 H(x)

如果网络很深，每一层都要直接学习一个复杂映射，优化会变得困难。

## 四、ResNet 的核心思想：学习残差

ResNet 不让网络直接学习：

H(x)

而是让网络学习：

F(x) = H(x) - x

那么：

H(x) = F(x) + x

这就是残差学习。

也就是说，ResNet 的一个残差块输出是：

y = F(x) + x

其中：

x：输入
F(x)：卷积层学到的残差
+ x：把原始输入直接加回来

结构大致是：
```
输入 x
  │
  ├───────────────┐
  │               │
  ↓               │
Conv → BN → ReLU  │
  ↓               │
Conv → BN         │
  ↓               │
  + ←─────────────┘
  ↓
ReLU
  ↓
输出 y

这条绕过卷积层直接传递输入的路径，就叫：

shortcut connection
skip connection
identity connection
```
## 五、为什么学习残差更容易？

假设某几层网络其实不需要做复杂变换，只需要保持输入不变，也就是：

H(x) = x

普通网络要直接学出这个恒等映射：

H(x) ≈ x

这并不一定容易。

但在 ResNet 中：

H(x) = F(x) + x

如果希望：

H(x) = x

那只需要让：

F(x) = 0

就可以了。

也就是说，残差块只需要学习：

在输入基础上需要改多少

而不是从零学习完整输出。

这就大大降低了深层网络的优化难度。

可以这样理解：

普通网络：
    每一层都要重新学习一个完整表示

ResNet：
    每个残差块只需要学习对输入的修正量

所以 ResNet 更容易训练得很深。

## 六、残差连接带来的直观效果

残差连接有几个重要作用。

1. 保留原始信息

输入 x 可以通过 shortcut 直接传到后面。

这样即使卷积路径暂时学得不好，信息也不会完全丢失。

2. 改善梯度传播

反向传播时，梯度可以沿 shortcut 更直接地传回前面层。

这缓解了深层网络中的梯度衰减问题。

3. 让深层网络更容易学习恒等映射

如果某些层没有必要改变特征，残差块可以接近恒等映射。

4. 支持构建非常深的网络

ResNet 可以稳定训练：

ResNet-18
ResNet-34
ResNet-50
ResNet-101
ResNet-152

相比早期 CNN，深度大幅增加，但训练更稳定。

## 七、ResNet 的基本结构

一个典型 ResNet 可以分成几部分：

1. Stem 初始卷积层
2. 多个残差 stage
3. Global Average Pooling
4. Fully Connected 分类层

整体结构如下：

输入图像
↓
7×7 Conv, stride=2
↓
BatchNorm + ReLU
↓
3×3 MaxPool, stride=2
↓
Stage 1: residual blocks
↓
Stage 2: residual blocks
↓
Stage 3: residual blocks
↓
Stage 4: residual blocks
↓
Global Average Pooling
↓
Fully Connected
↓
输出类别

以输入 224×224 图像为例，特征图尺寸大致变化是：

输入：224 × 224 × 3
↓
Conv7×7 stride=2：112 × 112
↓
MaxPool stride=2：56 × 56
↓
Stage 1：56 × 56
↓
Stage 2：28 × 28
↓
Stage 3：14 × 14
↓
Stage 4：7 × 7
↓
Global Average Pooling：1 × 1
↓
FC 分类

## 八、ResNet 中的两个基本残差块

ResNet 主要有两种 block：

BasicBlock
Bottleneck
```
1. BasicBlock

BasicBlock 常用于较浅的 ResNet：

ResNet-18
ResNet-34

结构是：

3×3 Conv
BN
ReLU
3×3 Conv
BN
Add shortcut
ReLU

可以写成：

F(x) = Conv3×3 → BN → ReLU → Conv3×3 → BN

y = F(x) + x

图示：

x
│
├─────────────────────┐
│                     │
↓                     │
3×3 Conv              │
BN                    │
ReLU                  │
3×3 Conv              │
BN                    │
↓                     │
+ ←───────────────────┘
↓
ReLU

BasicBlock 比较简单，计算量适中。

2. Bottleneck

Bottleneck 常用于较深的 ResNet：

ResNet-50
ResNet-101
ResNet-152

结构是：

1×1 Conv
3×3 Conv
1×1 Conv

具体来说：

1×1 Conv：降维
3×3 Conv：提取空间特征
1×1 Conv：升维

例如输入通道是 256，可以变成：

256 → 64 → 64 → 256

这样做的好处是：

减少 3×3 卷积的计算量
在更低维空间中做空间卷积
允许网络做得更深

Bottleneck 结构：

x
│
├────────────────────────────┐
│                            │
↓                            │
1×1 Conv 降维                 │
BN + ReLU                    │
3×3 Conv                     │
BN + ReLU                    │
1×1 Conv 升维                 │
BN                           │
↓                            │
+ ←──────────────────────────┘
↓
ReLU

Bottleneck 是 ResNet-50 及以上版本的核心。
```

## 九、为什么 Bottleneck 叫“瓶颈”？

因为它中间会先把通道数压缩，再恢复。

例如：

输入：256 通道
↓
1×1 Conv 降到 64 通道
↓
3×3 Conv 在 64 通道上计算
↓
1×1 Conv 升回 256 通道

中间的 64 通道就像一个“瓶颈”。

这样做可以减少计算量。

如果直接在 256 通道上做 3×3 卷积，计算量会很大。

Bottleneck 通过 1×1 → 3×3 → 1×1，既能保持表达能力，又能降低计算成本。

## 十、Shortcut 有两种情况

残差连接要求：

F(x) 和 x 的 shape 一样

这样才能相加。

但有时候特征图尺寸或通道数会变化，比如：

56×56×64 → 28×28×128

这时候 F(x) 和 x 不能直接相加。

所以 ResNet 中 shortcut 有两种形式。

1. Identity Shortcut

如果输入和输出 shape 一样，直接相加：

y = F(x) + x

这叫 identity shortcut。

例如：

56×56×64 → 56×56×64

可以直接加。

2. Projection Shortcut

如果 shape 不一样，需要用 1×1 Conv 调整 x：

y = F(x) + W_s x

其中 W_s 通常是一个 1×1 Conv。

例如：

x:    56×56×64
F(x): 28×28×128

shortcut 分支会做：

1×1 Conv, stride=2

把 x 变成：

28×28×128

然后再相加。

结构是：
```
x
│
├── 1×1 Conv stride=2 ──────┐
│                           │
↓                           │
Conv path                   │
↓                           │
+ ←─────────────────────────┘
```

## 十一、ResNet-18、34、50、101、152 的区别

这些数字表示网络大致的层数。

常见配置如下：

模型	Block 类型	每个 stage block 数量	特点
ResNet-18	BasicBlock	2, 2, 2, 2	轻量，速度快
ResNet-34	BasicBlock	3, 4, 6, 3	中等深度
ResNet-50	Bottleneck	3, 4, 6, 3	最常用，精度和速度平衡
ResNet-101	Bottleneck	3, 4, 23, 3	更深，精度更高但更慢
ResNet-152	Bottleneck	3, 8, 36, 3	很深，计算更重

注意：

ResNet-34 和 ResNet-50 的 stage block 数量看起来一样：
3, 4, 6, 3

但 ResNet-34 用 BasicBlock
ResNet-50 用 Bottleneck

所以 ResNet-50 实际层数更多。

## 十二、ResNet-18 的结构

ResNet-18 使用 BasicBlock。

结构大致是：

Conv1: 7×7 Conv, 64, stride=2
MaxPool
Layer1: BasicBlock × 2, 64 channels
Layer2: BasicBlock × 2, 128 channels
Layer3: BasicBlock × 2, 256 channels
Layer4: BasicBlock × 2, 512 channels
Global Average Pooling
FC

特征图尺寸变化：

224×224
↓
112×112
↓
56×56
↓
56×56
↓
28×28
↓
14×14
↓
7×7
↓
1×1

ResNet-18 的特点：

速度快
参数量较少
适合小数据集和轻量项目
常用作 baseline
## 十三、ResNet-50 的结构

ResNet-50 使用 Bottleneck。

结构大致是：

Conv1: 7×7 Conv, 64, stride=2
MaxPool
Layer1: Bottleneck × 3
Layer2: Bottleneck × 4
Layer3: Bottleneck × 6
Layer4: Bottleneck × 3
Global Average Pooling
FC

每个 Bottleneck 包含 3 个卷积层：

1×1
3×3
1×1

所以层数比 ResNet-34 更深。

ResNet-50 的特点：

精度和速度比较平衡
工程中非常常用
检测、分割中常作为 backbone
迁移学习常用

如果项目里不知道选哪个 ResNet，通常可以从：

ResNet-50

开始。

## 十四、ResNet 中的 BatchNorm 和 ReLU

ResNet 中常见结构是：

Conv → BatchNorm → ReLU

BatchNorm 的作用：

稳定训练
加快收敛
缓解梯度分布变化
允许使用较大学习率

ReLU 的作用：

引入非线性
提高表达能力
缓解梯度消失

在原始 ResNet block 中，通常是：

Conv → BN → ReLU → Conv → BN → Add → ReLU

后面还出现了 Pre-activation ResNet，结构有所变化。

## 十五、Pre-activation ResNet 是什么？

原始 ResNet 是：

Conv → BN → ReLU

而 Pre-activation ResNet 是：

BN → ReLU → Conv

残差块中变成：

BN → ReLU → Conv
BN → ReLU → Conv
Add

也就是说，BN 和 ReLU 放在卷积前面。

这种结构的好处是：

让 identity shortcut 更加干净
梯度传播更顺畅
更适合训练非常深的网络

可以简单理解：

原始 ResNet：
    激活在加法后

Pre-activation ResNet：
    先归一化和激活，再卷积
    shortcut 分支更接近真正的恒等映射

很多后续网络都受到 Pre-activation 思想影响。

## 十六、ResNet 为什么能缓解梯度消失？

假设残差块输出：

y = F(x) + x

反向传播时，损失函数对 x 的梯度可以分成两部分：

一部分经过 F(x)
一部分直接经过 shortcut

也就是说，梯度不必完全穿过很多卷积层，它可以沿 shortcut 直接传递。

直观理解：

普通深层网络：
    梯度必须一层一层穿过所有卷积
    容易变小或不稳定

ResNet：
    梯度有一条近路可以走
    更容易传回前面层

这就是 ResNet 能训练更深网络的重要原因。

## 十七、ResNet 和 VGG 的区别

VGG 的核心思想是：

不断堆叠 3×3 卷积

ResNet 的核心思想是：

堆叠残差块，并加入 shortcut connection

对比如下：

对比项	VGG	ResNet
核心结构	3×3 卷积堆叠	残差块
网络深度	VGG16 / VGG19	可到 50 / 101 / 152 层
训练难度	深了容易难训练	残差连接让深层更容易训练
参数量	较大	更高效
特征复用	较弱	shortcut 保留信息
常用性	早期常用	现代更常用

简单说：

VGG 是“直接堆深”
ResNet 是“带残差连接地堆深”

## 十八、ResNet 和 DenseNet 的区别

DenseNet 也是受残差连接影响的网络。

ResNet 是：

y = F(x) + x

也就是特征相加。

DenseNet 是：

x_l = H_l([x_0, x_1, ..., x_{l-1}])

也就是把前面所有层的特征 concat 起来。

区别：

对比项	ResNet	DenseNet
连接方式	相加	拼接
特征复用	较强	更强
显存占用	相对较低	concat 可能占显存
结构复杂度	简洁	更复杂
使用广泛度	非常广	也常见，但不如 ResNet 常用

简单说：

ResNet：学习残差，直接相加
DenseNet：密集连接，特征拼接复用

## 十九、ResNet 和 Transformer 的区别

ResNet 是 CNN 网络，Transformer 是基于 attention 的架构。

对比项	ResNet	Transformer / ViT
核心操作	卷积	Self-Attention
擅长	局部特征提取	全局关系建模
图像归纳偏置	强	相对弱
小数据表现	通常较稳	更依赖预训练
计算特点	高效、部署成熟	高分辨率下 attention 成本较高
常见用途	分类、检测、分割 backbone	大规模视觉预训练、全局建模

可以理解为：

ResNet 更擅长局部纹理、边缘、形状的层级提取
Transformer 更擅长建模远距离区域之间的关系

很多现代模型会结合二者思想。

## 二十、ResNet 在图像分类中的作用

在图像分类中，ResNet 通常完整使用：

输入图像
↓
ResNet backbone
↓
Global Average Pooling
↓
Fully Connected
↓
类别 logits

输出是：

[B, num_classes]

例如 ImageNet 是 1000 类：

[B, 1000]

如果目标任务是 5 类，需要把最后 FC 层改成：

model.fc = nn.Linear(model.fc.in_features, 5)

训练时常用：

CrossEntropyLoss

## 二十一、ResNet 在目标检测中的作用

目标检测模型经常用 ResNet 作为 backbone。

例如 Faster R-CNN：

输入图像
↓
ResNet 提取特征
↓
RPN 生成 proposals
↓
RoI Head 分类和回归

Mask R-CNN：

输入图像
↓
ResNet-FPN
↓
RPN
↓
RoIAlign
↓
class + bbox + mask

很多检测模型会使用：

ResNet-50-FPN
ResNet-101-FPN

其中 FPN 用于多尺度特征融合。

## 二十二、ResNet 在语义分割中的作用

语义分割中，ResNet 通常作为 encoder / backbone。

例如 DeepLabv3+：

输入图像
↓
ResNet backbone
↓
空洞卷积调整 output stride
↓
ASPP 多尺度上下文
↓
Decoder
↓
分割 mask

U-Net 也可以使用 ResNet encoder：

ResNet 编码器
+
U-Net 解码器

也就是：

ResNet-UNet

这种做法的优势是：

可以利用 ImageNet 预训练权重
特征提取能力强
小数据集更容易收敛

## 二十三、ResNet + FPN 是什么？

FPN 全称：

Feature Pyramid Network

它用于多尺度特征融合。

ResNet 本身有多个 stage：

C2: H/4
C3: H/8
C4: H/16
C5: H/32

浅层特征：

分辨率高
细节多
语义弱

深层特征：

分辨率低
语义强
细节少

FPN 会把它们融合成：

P2, P3, P4, P5

用于检测不同大小的目标。

所以 ResNet-FPN 是检测和实例分割中非常经典的组合。

例如：

Faster R-CNN ResNet50-FPN
Mask R-CNN ResNet50-FPN

## 二十四、ResNet 的迁移学习怎么做？

实际项目中，一般不会从零训练 ResNet，而是使用预训练模型。

流程是：

1. 加载 ImageNet 预训练 ResNet
2. 替换最后分类层
3. 冻结或微调部分 backbone
4. 在目标数据集上训练

例如 PyTorch：

import torch.nn as nn
from torchvision import models

num_classes = 5

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

如果数据量很小，可以先冻结 backbone：

for param in model.parameters():
    param.requires_grad = False

model.fc.requires_grad_(True)

只训练最后分类层。

如果数据量较多，可以微调整个网络：

for param in model.parameters():
    param.requires_grad = True

## 二十五、什么时候选 ResNet-18、ResNet-50、ResNet-101？

可以按任务复杂度和计算资源选择。

1. ResNet-18

适合：

小数据集
快速实验
轻量部署
教学和 baseline
算力有限

优点：

速度快
参数少
训练快

缺点：

表达能力不如更深模型
2. ResNet-50

适合：

大多数实际项目
分类 baseline
检测 backbone
分割 backbone
迁移学习

优点：

精度和速度平衡好
生态成熟
使用广泛
3. ResNet-101

适合：

更高精度需求
数据量较大
算力充足
检测/分割 benchmark

优点：

特征表达更强

缺点：

计算更重
训练更慢
部署成本更高

## 二十六、ResNet 的优点

ResNet 的优点主要有：

1. 解决深层网络退化问题
2. 残差连接改善梯度传播
3. 可以训练很深的 CNN
4. 结构清晰，容易理解
5. 迁移学习效果好
6. 可作为分类、检测、分割 backbone
7. 工程生态成熟
8. 和 FPN、UNet、DeepLab 等结构容易结合

一句话：

ResNet 是 CNN backbone 中最经典、最稳定、最常用的基础结构之一。
## 二十七、ResNet 的缺点

ResNet 也不是完美的。

1. 全局建模能力有限

ResNet 本质上还是 CNN，卷积主要关注局部区域。

虽然深层感受野会变大，但对远距离依赖的直接建模不如 Transformer。

2. 深层模型计算量较大

ResNet-101、ResNet-152 计算成本明显更高。

如果部署到移动端，通常不如 MobileNet、ShuffleNet 这类轻量网络合适。

3. 对高分辨率分割边界不一定最优

ResNet 分类 backbone 会不断下采样，细节会丢失。

所以做分割时通常需要：

FPN
Decoder
Skip Connection
Atrous Convolution

来恢复空间信息。

4. 结构已经比较经典

在很多高精度任务中，现代模型可能使用：

ConvNeXt
Swin Transformer
ViT
EfficientNet
InternImage

但 ResNet 依然是非常重要的 baseline。

## 三十一、ResNet 核心总结

核心说明：

ResNet 是一种经典的卷积神经网络，它的核心是残差连接。普通深层网络随着层数增加可能出现退化问题，也就是网络更深后训练误差反而变高。ResNet 通过引入 shortcut connection，让网络不直接学习目标映射 H(x)，而是学习残差 F(x)=H(x)-x，最终输出为 F(x)+x。这样如果某些层只需要保持恒等映射，残差分支只需要学习接近 0 的映射即可，优化难度会降低。

ResNet 的基本模块包括 BasicBlock 和 Bottleneck。ResNet-18、ResNet-34 通常使用 BasicBlock，由两个 3×3 卷积组成；ResNet-50、101、152 使用 Bottleneck，由 1×1、3×3、1×1 三个卷积组成，其中 1×1 卷积用于降维和升维，减少计算量。ResNet 因为训练稳定、迁移学习效果好，所以常被用作图像分类、目标检测和语义分割的 backbone。

## 三十二、延伸知识：ResNet 为什么能训练更深？

核心说明：

因为 ResNet 引入了残差连接，使得输入可以通过 shortcut 直接传到后面的层。这样一方面保留了原始信息，另一方面反向传播时梯度也可以沿 shortcut 更顺畅地传回前面的层，缓解深层网络中的梯度传播困难。同时，残差学习让网络只需要学习相对于输入的变化量，而不是完整映射。如果某些层没有必要改变输入，只需要让残差分支输出接近 0，就可以实现恒等映射，因此优化更容易。

## 三十三、延伸知识：BasicBlock 和 Bottleneck 有什么区别？

核心说明：

BasicBlock 通常用于 ResNet-18 和 ResNet-34，由两个 3×3 卷积组成，结构简单，适合较浅网络。Bottleneck 通常用于 ResNet-50、101、152，由 1×1、3×3、1×1 三个卷积组成。第一个 1×1 卷积用于降维，中间 3×3 卷积负责空间特征提取，最后一个 1×1 卷积用于升维。Bottleneck 可以在保持表达能力的同时减少 3×3 卷积的计算量，因此适合构建更深的网络。

## 三十四、延伸知识：ResNet 和 VGG 有什么区别？

核心说明：

VGG 主要是通过堆叠大量 3×3 卷积来加深网络，结构简单但参数量较大，而且网络太深后训练会变困难。ResNet 在加深网络的同时引入了残差连接，让输入可以直接跨层传递，缓解深层网络的退化问题和梯度传播问题。因此 ResNet 能够稳定训练更深的网络，比如 ResNet-50、101、152，并且在分类、检测、分割任务中都更常用。

## 三十五、延伸知识：ResNet 中 shortcut 什么时候需要 1×1 卷积？

核心说明：

当残差分支输出 F(x) 和 shortcut 分支的 x 形状一致时，可以直接相加，这叫 identity shortcut。但如果特征图尺寸或通道数发生变化，比如 stride=2 下采样，或者通道数从 64 变成 128，那么 x 和 F(x) 不能直接相加。这时需要在 shortcut 分支上使用 1×1 卷积，有时还带 stride，用来调整 x 的空间尺寸和通道数，使它和残差分支输出形状一致。