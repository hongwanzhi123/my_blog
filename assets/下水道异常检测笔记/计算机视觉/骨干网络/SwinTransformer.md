# SwimTransformer

Swin Transformer 是一种把 Transformer 改造成适合图像任务的层级视觉骨干网络。它通过窗口注意力降低计算量，通过移动窗口实现跨窗口信息交互，因此既能处理高分辨率图像，又适合分类、检测、分割等视觉任务。

## 一、Swin Transformer 是什么？

Swin Transformer 全称是：

Shifted Window Transformer

中文可以理解为：

移动窗口 Transformer

它是微软提出的一种视觉 Transformer 架构，主要用于计算机视觉任务，例如：

图像分类
目标检测
语义分割
实例分割
全景分割

传统 ViT 主要用于图像分类，而 Swin Transformer 更像 ResNet、ConvNeXt 那样，可以作为通用视觉 backbone。

很多模型可以这样搭配：

Mask R-CNN + Swin Transformer
Cascade Mask R-CNN + Swin Transformer
UPerNet + Swin Transformer
Mask2Former + Swin Transformer

一句话概括：

Swin Transformer 是一种层级式视觉 Transformer，它把图像划分为局部窗口，在窗口内做 self-attention，并通过 shifted window 让不同窗口之间交换信息，从而兼顾计算效率和建模能力。

## 二、为什么需要 Swin Transformer？

在 Swin Transformer 之前，Vision Transformer，也就是 ViT，已经证明 Transformer 可以用于图像分类。

但是原始 ViT 有几个问题：

1. 需要大量数据预训练
2. 对高分辨率图像计算量很大
3. 输出是单尺度特征，不太适合检测和分割
4. 缺少类似 CNN 的层级特征结构
### 1. ViT 计算量太大

ViT 会把图像切成 patch，然后把所有 patch 当成 token。

例如输入图像：

224 × 224

patch size = 16：

224 / 16 = 14

所以 token 数是：

14 × 14 = 196

这还可以接受。

但如果是检测、分割任务，输入可能是：

1024 × 1024

如果 patch size 还是 16：

64 × 64 = 4096 tokens

Self-Attention 的复杂度大约是：

O(N²)

N 是 token 数量。

所以 token 从 196 增加到 4096，attention 计算量会急剧上升。

### 2. ViT 缺少层级特征

CNN 通常有层级特征：

浅层：分辨率高，细节多
深层：分辨率低，语义强

例如 ResNet：

C2: H/4
C3: H/8
C4: H/16
C5: H/32

这对检测和分割非常重要。

目标检测需要多尺度特征：

小目标依赖高分辨率特征
大目标依赖高级语义特征

但原始 ViT 通常输出一个固定分辨率的 token 序列，不像 CNN 那样天然有多层级特征。

Swin Transformer 就是为了解决这些问题提出的。

## 三、Swin Transformer 的核心思想

Swin Transformer 的核心设计有三个：

1. Hierarchical Architecture：层级结构
2. Window-based Self-Attention：窗口自注意力
3. Shifted Window：移动窗口机制

可以这样理解：

层级结构：
    像 CNN 一样逐步降低分辨率、增加通道数

窗口注意力：
    不对整张图所有 token 做 attention，而是在局部窗口内做

移动窗口：
    下一层把窗口位置移动，让不同窗口之间也能交流信息

这三个设计共同解决了：

计算量问题
高分辨率输入问题
跨区域建模问题
多尺度特征问题

## 四、Swin Transformer 的整体结构

Swin Transformer 的整体结构和 CNN backbone 很像，通常分为 4 个 stage。

以输入图像 224×224 为例：

输入图像
↓
Patch Partition
↓
Linear Embedding
↓
Stage 1: 56 × 56
↓
Patch Merging
↓
Stage 2: 28 × 28
↓
Patch Merging
↓
Stage 3: 14 × 14
↓
Patch Merging
↓
Stage 4: 7 × 7
↓
Global Average Pooling
↓
分类头

可以看到，它也是逐层下采样：

56×56 → 28×28 → 14×14 → 7×7

这和 ResNet / ConvNeXt 很像。

所以 Swin Transformer 不只是一个分类模型，而是一个通用 backbone。

## 五、Patch Partition：图像切块

Swin Transformer 首先把输入图像切成不重叠的小 patch。

例如输入图像：

224 × 224 × 3

patch size = 4：

每个 patch 大小是 4 × 4

那么图像会被切成：

56 × 56 个 patch

因为：

224 / 4 = 56

每个 patch 包含：

4 × 4 × 3 = 48 个数

所以最开始的 token 数是：

56 × 56 = 3136

每个 token 的原始维度是 48。

## 六、Linear Embedding：线性映射

Patch Partition 后，每个 patch 是一个向量。

例如：

4 × 4 × 3 = 48

但是 Transformer 需要统一的 embedding 维度，比如：

C = 96

所以需要一个线性层把每个 patch 映射到 96 维：

48 → 96

这个过程叫：

Linear Embedding

也可以理解成 CNN 里的：

4×4 Conv, stride=4

经过这一步，特征变成：

56 × 56 × 96

## 七、Swin Transformer Block

Swin Transformer 的基本模块叫：

Swin Transformer Block

它和普通 Transformer block 很像，但注意力部分不是全局 attention，而是窗口 attention。

一个 Swin Transformer block 大致结构是：

输入 x
↓
LayerNorm
↓
Window Multi-Head Self-Attention
↓
Residual Connection
↓
LayerNorm
↓
MLP
↓
Residual Connection
↓
输出

写成公式大概是：

x = x + W-MSA(LN(x))
x = x + MLP(LN(x))

其中：

W-MSA = Window Multi-head Self-Attention

如果是移动窗口版本，则是：

SW-MSA = Shifted Window Multi-head Self-Attention

## 八、什么是 Window-based Self-Attention？

原始 Transformer 的 self-attention 是全局的。

也就是说，每个 token 都要和所有 token 计算注意力。

如果特征图大小是：

56 × 56

token 数是：

3136

全局 attention 要计算：

3136 × 3136

非常大。

Swin Transformer 改成：

把特征图划分成多个小窗口，每个窗口内部做 self-attention，窗口之间暂时不交互。

例如窗口大小是：

7 × 7

每个窗口有：

49 个 token

那么 attention 只在这 49 个 token 内计算。

这样计算量会小很多。

## 九、窗口注意力的直观例子

假设特征图是：

56 × 56

窗口大小是：

7 × 7

那么可以分成：

8 × 8 = 64 个窗口

每个窗口里有：

49 个 token

普通全局 attention：

所有 3136 个 token 两两计算注意力

窗口 attention：

每个窗口内部 49 个 token 两两计算注意力
一共有 64 个窗口

计算复杂度从：

3136²

变成大约：

64 × 49²

降低非常明显。

## 十、窗口注意力的问题

窗口注意力虽然降低了计算量，但有一个问题：

如果每一层都只在固定窗口内做 attention，那么不同窗口之间无法交流信息。

例如左上角窗口里的 token，永远看不到右边窗口里的 token。

这会限制模型的全局建模能力。

这时就需要 Swin Transformer 最核心的设计：

Shifted Window

也就是移动窗口。

## 十一、什么是 Shifted Window？

Shifted Window 的思想是：

第一层使用正常窗口划分，第二层把窗口整体移动半个窗口大小，再做窗口 attention。

例如窗口大小是：

7 × 7

移动距离通常是：

3

也就是：

window_size / 2

第一层窗口划分：

| A | B | C |
| D | E | F |
| G | H | I |

每个窗口内部做 attention。

第二层窗口整体移动后，新的窗口会跨越原来的窗口边界。

这样原本属于不同窗口的 token，就会在新窗口里相遇，从而交换信息。

## 十二、Shifted Window 的作用

Shifted Window 有两个重要作用：

1. 实现跨窗口信息交互
2. 仍然保持局部窗口 attention 的低计算量

如果没有 shifted window：

每个窗口像孤岛
窗口之间缺少交流

有了 shifted window：

第 1 层：窗口内建模
第 2 层：移动窗口，跨窗口建模
第 3 层：继续窗口内建模
第 4 层：继续移动窗口交互

这样多层堆叠后，信息可以逐渐从局部传播到更大区域。

可以理解为：

W-MSA 负责局部窗口内部建模
SW-MSA 负责跨窗口信息交互

## 十三、W-MSA 和 SW-MSA

Swin Transformer Block 通常成对出现：

第一个 block：W-MSA
第二个 block：SW-MSA

也就是：

Block 1:
    Window Multi-head Self-Attention

Block 2:
    Shifted Window Multi-head Self-Attention

这样设计可以让模型既高效，又能跨窗口交流。

结构可以写成：

Block 1:
    LN → W-MSA → Residual
    LN → MLP → Residual

Block 2:
    LN → SW-MSA → Residual
    LN → MLP → Residual

## 十四、为什么 Swin Transformer 适合高分辨率图像？

因为它没有做全局 attention，而是在局部窗口内做 attention。

如果图像分辨率提高，token 数增加，Swin 的计算量增长相对更可控。

普通 ViT：

Self-Attention 复杂度 ≈ O(N²)

Swin Transformer：

Window Attention 复杂度 ≈ O(N × M²)

其中：

N = 总 token 数
M = 每个窗口中的 token 数

如果窗口大小固定，比如 7×7，那么 M 固定为 49。

因此随着图像变大，计算量大致随 token 数线性增长，而不是平方增长。

这就是 Swin 比原始 ViT 更适合检测、分割、高分辨率输入的重要原因。

## 十五、Patch Merging：类似 CNN 下采样

Swin Transformer 中，每个 stage 之间会做：

Patch Merging

它的作用类似 CNN 中的下采样。

例如 Stage 1 输出：

56 × 56 × 96

Patch Merging 会把相邻的 2×2 patch 合并。

空间尺寸变成：

28 × 28

通道数通常增加：

96 → 192

也就是：

H × W × C
↓
H/2 × W/2 × 2C

这和 CNN 很像：

空间尺寸减半
通道数增加
语义信息增强

## 十六、Patch Merging 怎么做？

假设输入特征是：

H × W × C

Patch Merging 会取每个 2×2 邻域：

左上 token
右上 token
左下 token
右下 token

把它们在通道维度拼接：

C + C + C + C = 4C

然后通过线性层映射到：

2C

所以过程是：

H × W × C
↓
H/2 × W/2 × 4C
↓
Linear
↓
H/2 × W/2 × 2C

这就完成了下采样和通道扩展。

## 十七、Swin Transformer 的四个 Stage

以 Swin-Tiny 为例，常见结构如下：

Stage 1:
    resolution = 56 × 56
    channels = 96
    blocks = 2

Stage 2:
    resolution = 28 × 28
    channels = 192
    blocks = 2

Stage 3:
    resolution = 14 × 14
    channels = 384
    blocks = 6

Stage 4:
    resolution = 7 × 7
    channels = 768
    blocks = 2

可以看到，它和 CNN backbone 很像：

浅层分辨率高
深层通道数大

这就是为什么 Swin Transformer 可以很自然地接 FPN、UPerNet、Mask R-CNN 等检测分割框架。

## 十八、Swin Transformer 的模型规模

常见版本有：

Swin-T
Swin-S
Swin-B
Swin-L

分别是：

Swin-Tiny
Swin-Small
Swin-Base
Swin-Large

一般来说：

Tiny：速度快，适合轻量实验
Small：中等规模
Base：更强，计算更大
Large：高精度，训练成本高

它们主要区别在：

通道数
block 数量
参数量
计算量
预训练数据规模

如果你做项目，通常可以从：

Swin-Tiny

开始。

如果算力足够，再尝试：

Swin-Small
Swin-Base

## 二十二、Swin Transformer 在图像分类中的使用

用于分类时，流程是：

输入图像
↓
Patch Partition + Linear Embedding
↓
Stage 1
↓
Stage 2
↓
Stage 3
↓
Stage 4
↓
Global Average Pooling
↓
Linear 分类头
↓
类别 logits

输出形状：

[B, num_classes]

如果是 ImageNet：

num_classes = 1000

如果你做自己的分类任务，比如 5 类病害分类，就把最后分类头改成：

num_classes = 5

## 二十三、Swin Transformer 在目标检测中的使用

目标检测需要多尺度特征。

Swin Transformer 的四个 stage 刚好可以输出：

C2: H/4
C3: H/8
C4: H/16
C5: H/32

然后接 FPN：

P2, P3, P4, P5

常见组合：

Mask R-CNN + Swin
Cascade Mask R-CNN + Swin
RetinaNet + Swin
DINO + Swin

相比 ResNet backbone，Swin 具有更强的上下文建模能力，在检测任务中通常表现很好。

## 二十四、Swin Transformer 在语义分割中的使用

语义分割也需要多尺度特征。

常见组合：

UPerNet + Swin Transformer
Semantic FPN + Swin Transformer
Mask2Former + Swin Transformer

流程：

输入图像
↓
Swin backbone 提取多尺度特征
↓
分割解码器融合特征
↓
上采样
↓
输出像素级类别图

Swin 的优势是：

窗口 attention 有较强局部区域建模能力
shifted window 可以跨区域传播信息
层级结构适合多尺度分割

## 二十五、Swin Transformer 在实例分割中的使用

实例分割中常见：

Mask R-CNN + Swin
Mask2Former + Swin

例如 Mask R-CNN：

输入图像
↓
Swin backbone
↓
FPN
↓
RPN
↓
RoIAlign
↓
class + bbox + mask

Swin 替代 ResNet 作为 backbone，提供更强特征。

## 二十二、Swin Transformer 在图像分类中的使用

用于分类时，流程是：

输入图像
↓
Patch Partition + Linear Embedding
↓
Stage 1
↓
Stage 2
↓
Stage 3
↓
Stage 4
↓
Global Average Pooling
↓
Linear 分类头
↓
类别 logits

输出形状：

[B, num_classes]

如果是 ImageNet：

num_classes = 1000

如果你做自己的分类任务，比如 5 类病害分类，就把最后分类头改成：

num_classes = 5

## 二十三、Swin Transformer 在目标检测中的使用

目标检测需要多尺度特征。

Swin Transformer 的四个 stage 刚好可以输出：

C2: H/4
C3: H/8
C4: H/16
C5: H/32

然后接 FPN：

P2, P3, P4, P5

常见组合：

Mask R-CNN + Swin
Cascade Mask R-CNN + Swin
RetinaNet + Swin
DINO + Swin

相比 ResNet backbone，Swin 具有更强的上下文建模能力，在检测任务中通常表现很好。

## 二十四、Swin Transformer 在语义分割中的使用

语义分割也需要多尺度特征。

常见组合：

UPerNet + Swin Transformer
Semantic FPN + Swin Transformer
Mask2Former + Swin Transformer

流程：

输入图像
↓
Swin backbone 提取多尺度特征
↓
分割解码器融合特征
↓
上采样
↓
输出像素级类别图

Swin 的优势是：

窗口 attention 有较强局部区域建模能力
shifted window 可以跨区域传播信息
层级结构适合多尺度分割

## 二十五、Swin Transformer 在实例分割中的使用

实例分割中常见：

Mask R-CNN + Swin
Mask2Former + Swin

例如 Mask R-CNN：

输入图像
↓
Swin backbone
↓
FPN
↓
RPN
↓
RoIAlign
↓
class + bbox + mask

Swin 替代 ResNet 作为 backbone，提供更强特征。

## 二十六、Swin Transformer 的优点

Swin Transformer 的优点主要有：

1. 计算效率比原始 ViT 更适合高分辨率图像
2. 层级结构天然适合检测和分割
3. shifted window 实现跨窗口信息交互
4. 可以作为通用视觉 backbone
5. 在分类、检测、分割任务中表现都很强
6. 比原始 ViT 更接近 CNN 的多尺度特征形式

尤其是：

窗口注意力降低计算量
移动窗口补充跨区域信息
Patch Merging 形成层级特征

这三点是 Swin 的核心。

## 二十七、Swin Transformer 的缺点

Swin Transformer 也有不足。

### 1. 实现比 CNN 复杂

ResNet、ConvNeXt 的结构相对简单。

Swin 需要处理：

窗口划分
窗口移动
attention mask
relative position bias
patch merging

工程实现更复杂。

### 2. 对预训练依赖较强

Transformer 类模型通常更依赖大规模预训练。

如果你的数据很小，从零训练 Swin 不一定比 ResNet 好。

实际项目中一般用预训练权重。

### 3. 部署比 CNN 更麻烦

虽然 Swin 比 ViT 高效，但它仍然有 attention 操作。

在某些端侧设备上，attention 不一定像卷积那样高度优化。

### 4. 窗口机制仍然不是完全全局 attention

Swin 通过 shifted window 逐步跨窗口传播信息，但它不是一层就全局交互。

如果任务特别依赖长距离全局关系，仍然可能需要更强全局建模模块。

## 二十八、Swin Transformer 的关键细节：Relative Position Bias

在图像中，空间位置很重要。

Transformer 原本需要位置编码。

Swin Transformer 使用：

Relative Position Bias

也就是相对位置偏置。

在窗口 attention 中，每个 token 和另一个 token 的相对位置会影响注意力分数。

例如在同一个 7×7 窗口内：

左边相邻
右边相邻
上方相邻
对角线位置

这些不同相对位置会有不同 bias。

Attention 计算可以理解成：

Attention(Q, K, V) = Softmax(QK^T / sqrt(d) + B) V

其中：

B = relative position bias

它告诉模型：

两个 token 的相对空间位置关系

这对图像任务很重要。

## 二十九、Swin Transformer 的面试回答版本

如果面试官问：

你了解 Swin Transformer 吗？

可以这样回答：

Swin Transformer 是一种层级式视觉 Transformer，它的核心设计是 shifted window attention。原始 ViT 对所有 patch 做全局 self-attention，计算量会随着 token 数平方增长，不太适合高分辨率的检测和分割任务。Swin Transformer 把特征图划分成固定大小窗口，只在窗口内部做 self-attention，从而显著降低计算量。

但是固定窗口会导致窗口之间缺少信息交互，所以 Swin 又引入 shifted window，也就是在相邻 Transformer block 中移动窗口划分，让原本处于不同窗口的 token 可以进入同一个窗口进行 attention。这样既保持了窗口 attention 的高效性，又实现了跨窗口的信息传播。

另外，Swin Transformer 通过 patch merging 构建类似 CNN 的层级结构，逐步降低空间分辨率、增加通道数，因此可以输出多尺度特征，非常适合作为目标检测、语义分割和实例分割任务的 backbone。

## 三十、如果面试官追问：Swin 为什么比 ViT 更适合检测和分割？

可以回答：

主要有两点。第一，Swin 使用窗口 attention，计算复杂度比全局 attention 更低，更适合高分辨率输入；检测和分割通常需要比分类更高的图像分辨率。第二，Swin 通过 patch merging 构建层级特征，可以输出不同尺度的 feature maps，类似 ResNet 的 C2、C3、C4、C5，这样可以方便地接 FPN、UPerNet、Mask R-CNN 等检测分割头。而原始 ViT 通常是单尺度 token 输出，直接用于检测分割不如 Swin 自然。

## 三十一、如果面试官追问：Shifted Window 有什么作用？

可以回答：

如果只使用普通 window attention，每个 token 只能和同一窗口内的 token 交互，不同窗口之间信息隔离。Shifted window 通过在相邻 block 中把窗口划分移动半个窗口大小，让原来位于不同窗口的 token 被划分到同一个新窗口中，从而实现跨窗口信息交互。这样既不需要做全局 attention，也能逐层扩大信息交流范围。

## 三十二、如果面试官追问：Patch Merging 是什么？

可以回答：

Patch Merging 是 Swin Transformer 中的下采样模块，作用类似 CNN 中的 stride convolution 或 pooling。它会把相邻 2×2 的 token 在通道维度拼接起来，然后通过一个线性层降维。这样空间分辨率减半，通道数增加，形成层级特征。例如输入是 H×W×C，经过 Patch Merging 后变成 H/2×W/2×2C。

## 三十三、如果面试官追问：Swin 和 CNN 有什么相似点？

可以回答：

Swin 和 CNN 都有层级结构，都会随着网络加深逐渐降低空间分辨率、增加通道数，并形成多尺度特征。Swin 的窗口 attention 也类似卷积的局部建模，只不过卷积使用固定卷积核，而 Swin 在局部窗口内通过 self-attention 自适应建模 token 之间关系。因此 Swin 可以看作一种更适合视觉任务的层级 Transformer backbone。