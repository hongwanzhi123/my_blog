## VIT

ViT 是把图像切成一个个 patch，然后像处理文本 token 一样送入 Transformer Encoder 做图像分类的模型。

它的核心意义是：证明 Transformer 不只适合 NLP，也可以直接用于计算机视觉任务。

## 一、ViT 是什么？

ViT 全称是：

Vision Transformer

中文可以理解为：

视觉 Transformer

传统 CNN 处理图像时，核心操作是卷积：

图像 → 卷积提取局部特征 → 池化 / 下采样 → 分类

而 ViT 的思路是：

图像 → 切成 patch → patch 当成 token → Transformer Encoder → 分类

也就是说，ViT 把一张图像变成一个 token 序列，然后用 Transformer 处理这个序列。

这和 NLP 很像：

NLP:
    句子 → 单词 token → Transformer

ViT:
    图像 → patch token → Transformer

## 二、ViT 为什么重要？

在 ViT 之前，计算机视觉主流模型基本是 CNN，比如：

AlexNet
VGG
GoogLeNet
ResNet
DenseNet
EfficientNet

CNN 的优势是很明显的：

局部感受野
参数共享
平移等变性
对图像有很强的归纳偏置
小数据上比较稳定

但是 CNN 也有局限：

长距离依赖建模不够直接
感受野需要通过多层卷积逐渐扩大
全局关系建模不如 self-attention 灵活

ViT 的出现说明：

在大规模数据预训练下，纯 Transformer 也可以在图像分类任务上取得非常强的效果。

它打开了后续大量视觉 Transformer 模型的发展，比如：

DeiT
Swin Transformer
PVT
SegFormer
MAE
BEiT
DINO
CLIP Vision Encoder
SAM Image Encoder

## 三、ViT 的核心思想

ViT 的核心思想很简单：

把图像切成固定大小的小块 patch，每个 patch 看作一个 token，然后把这些 token 输入标准 Transformer Encoder。

例如输入图像是：

224 × 224 × 3

如果 patch size 是：

16 × 16

那么图像会被切成：

14 × 14 = 196 个 patch

因为：

224 / 16 = 14

每个 patch 都变成一个 token。

然后加上：

位置编码
CLS token

送进 Transformer Encoder，最后用 CLS token 做分类。

整体流程：

输入图像
↓
切成 patch
↓
Patch Embedding
↓
加入 Position Embedding
↓
加入 CLS Token
↓
Transformer Encoder
↓
MLP Head
↓
输出类别

## 四、ViT 的整体结构

ViT 主要由五部分组成：

1. Patch Partition
2. Patch Embedding
3. Position Embedding
4. Transformer Encoder
5. Classification Head

结构可以写成：

Image
↓
Split into patches
↓
Linear Projection
↓
Add Position Embedding
↓
Add CLS Token
↓
Transformer Encoder × L
↓
CLS Token
↓
MLP Head
↓
Class Prediction

## 五、Patch Partition：图像切块

ViT 不像 CNN 那样直接用卷积核滑动处理图像，而是先把图像切成很多不重叠的 patch。

假设输入图像是：

224 × 224 × 3

patch size 是：

16 × 16

那么每个 patch 的大小是：

16 × 16 × 3

一个 patch 展平后维度是：

16 × 16 × 3 = 768

图像一共切成：

14 × 14 = 196 个 patch

所以原图变成了：

196 个 patch token

可以理解成：

一张图像被切成 196 个“小单词”

## 六、Patch Embedding：把 patch 变成 token 向量

每个 patch 展平后是一个向量。

例如：

16 × 16 × 3 = 768 维

但是 Transformer 通常需要统一的 hidden dimension，比如：

D = 768

所以 ViT 会用一个线性层把每个 patch 映射成 embedding：

patch vector → patch embedding

如果 patch 展平后本来就是 768 维，ViT-Base 的 embedding dim 也是 768，但本质上仍然会经过一个线性映射。

形式可以写成：

x_patch ∈ R^(P×P×C)
flatten
↓
Linear Projection
↓
z_patch ∈ R^D

在代码实现里，Patch Embedding 通常可以用两种方式实现：

## 1. Linear Projection

先把 patch 手动 flatten，再用 Linear：

nn.Linear(patch_size * patch_size * in_channels, embed_dim)
## 2. Conv2d 实现

更常见、更高效：

nn.Conv2d(
    in_channels=3,
    out_channels=embed_dim,
    kernel_size=patch_size,
    stride=patch_size
)

例如：

nn.Conv2d(3, 768, kernel_size=16, stride=16)

输入：

[B, 3, 224, 224]

输出：

[B, 768, 14, 14]

再 flatten 成：

[B, 196, 768]

这个 Conv2d 本质上就是在做 patch embedding。

## 七、为什么 ViT 需要位置编码？

Transformer 本身不天然知道 token 的空间位置。

如果只给它 196 个 patch embedding，它不知道：

哪个 patch 在左上角
哪个 patch 在右下角
哪些 patch 相邻
哪些 patch 距离很远

而图像任务非常依赖空间结构。

所以 ViT 需要给每个 patch token 加上位置编码：

patch embedding + position embedding

例如：

第 1 个 patch：左上角
第 2 个 patch：第一行第二个
...
第 196 个 patch：右下角

位置编码让 Transformer 知道每个 token 在图像中的大致位置。

ViT 中常用的是：

learnable position embedding

也就是位置编码本身是可学习参数。

## 八、CLS Token 是什么？

ViT 借鉴了 BERT 的做法，引入了一个特殊 token：

[CLS]

也叫：

class token

它不是来自图像 patch，而是一个可学习向量。

假设图像有 196 个 patch token，加入 CLS token 后序列长度变成：

196 + 1 = 197

输入 Transformer 的序列是：

[CLS], patch_1, patch_2, ..., patch_196

经过 Transformer Encoder 后，CLS token 会和所有 patch token 进行 self-attention 交互。

最后分类时，不是取所有 patch，而是取 CLS token 的输出：

CLS output → MLP Head → class logits

可以理解为：

CLS token 是一个全局信息汇聚器，它通过 self-attention 从所有 patch 中收集整张图像的信息。

## 九、ViT 的输入序列形状

以 ViT-Base 为例：

输入图像：224 × 224 × 3
patch size：16 × 16
patch 数量：14 × 14 = 196
embedding dim：768

Patch embedding 后：

[B, 196, 768]

加入 CLS token 后：

[B, 197, 768]

加入 position embedding 后：

[B, 197, 768]

送入 Transformer Encoder。

## 十、Transformer Encoder 在 ViT 中做什么？

ViT 使用的是标准 Transformer Encoder。

每一层 Encoder 包括：

1. LayerNorm
2. Multi-Head Self-Attention
3. Residual Connection
4. LayerNorm
5. MLP / Feed Forward Network
6. Residual Connection

结构：

输入 x
↓
LayerNorm
↓
Multi-Head Self-Attention
↓
Residual Add
↓
LayerNorm
↓
MLP
↓
Residual Add
↓
输出

公式可以写成：

x = x + MSA(LN(x))
x = x + MLP(LN(x))

这里用的是 Pre-LN 结构，也就是先 LayerNorm，再进入 Attention 或 MLP。

## 十一、Self-Attention 在 ViT 中的作用

CNN 的卷积通常只看局部区域。

例如 3×3 卷积一次只能看附近 9 个像素位置。

而 Self-Attention 可以让每个 patch token 和所有 patch token 交互。

例如一个左上角的 patch，可以直接关注右下角的 patch。

这使 ViT 天然具备全局建模能力。

在图像中，这意味着模型可以学习：

物体不同部位之间的关系
前景和背景之间的关系
远距离区域之间的关系
整体形状和布局
上下文信息

比如识别一只狗，CNN 可能逐层组合局部纹理和形状；ViT 可以直接让狗头、狗身、狗腿等不同区域通过 attention 建立联系。

## 十二、Multi-Head Self-Attention 是什么？

Self-Attention 的基本思想是：

每个 token 根据 Query、Key、Value 去决定自身应该关注哪些 token。

对输入 token 表示 X，会通过线性层得到：

Q = XWq
K = XWk
V = XWv

然后计算：

Attention(Q, K, V) = Softmax(QK^T / sqrt(d)) V

其中：

Q：查询，表示想找什么信息
K：键，表示有什么信息可被匹配
V：值，表示真正要传递的信息

Multi-Head 的意思是：

把注意力分成多个头
每个头学习不同的关系
最后拼接起来

例如 ViT-Base 通常有：

12 个 attention heads

每个 head 可以关注不同模式：

有的关注边缘
有的关注物体部件
有的关注背景关系
有的关注全局轮廓

## 十三、ViT 中的 MLP 是什么？

每个 Transformer Encoder 里，Attention 后面会接一个 MLP，也叫 FFN：

Linear
GELU
Dropout
Linear
Dropout

通常隐藏层维度是 embedding dim 的 4 倍。

例如 ViT-Base：

embedding dim = 768
MLP hidden dim = 3072

所以 MLP 是：

768 → 3072 → 768

作用是：

增强每个 token 的非线性表达能力
进行通道维度的信息变换

如果说 Attention 主要做：

token 之间的信息交互

那么 MLP 主要做：

每个 token 内部特征变换

## 十四、ViT 的分类头

Transformer Encoder 输出：

[B, 197, 768]

取出 CLS token：

[B, 768]

送入分类头：

Linear(768, num_classes)

输出：

[B, num_classes]

例如 ImageNet：

[B, 1000]

如果是二分类：

[B, 2]

如果是 5 分类：

[B, 5]

## 十五、ViT 的不同规模

常见 ViT 模型有：

ViT-Base
ViT-Large
ViT-Huge

也可以按 patch size 区分：

ViT-B/16
ViT-B/32
ViT-L/16
ViT-L/32
ViT-H/14

其中：

B = Base
L = Large
H = Huge
/16 = patch size 16
/32 = patch size 32
/14 = patch size 14

例如：

ViT-B/16
patch size = 16
embedding dim = 768
encoder layers = 12
attention heads = 12
MLP dim = 3072
ViT-L/16
patch size = 16
embedding dim = 1024
encoder layers = 24
attention heads = 16
MLP dim = 4096
ViT-H/14
patch size = 14
embedding dim 更大
层数更多
计算量更高

模型越大，通常性能越强，但训练和推理成本也越高。

## 十六、Patch Size 对 ViT 有什么影响？

Patch size 是 ViT 里非常重要的超参数。

假设输入是：

224 × 224

如果 patch size = 16：

14 × 14 = 196 tokens

如果 patch size = 32：

7 × 7 = 49 tokens

如果 patch size = 8：

28 × 28 = 784 tokens

Self-Attention 的复杂度是：

O(N²)

N 是 token 数量。

所以：

patch 越小 → token 越多 → 细节更多 → 计算量更大
patch 越大 → token 越少 → 计算更快 → 细节损失更多

例如：

ViT-B/16 比 ViT-B/32 更细
但计算量也更大

在图像分类中，16×16 是很常见的选择。

## 十七、ViT 和 CNN 的核心区别
对比项	CNN	ViT
基本单位	像素局部区域	patch token
核心操作	卷积	Self-Attention
信息交互	局部到全局逐层扩大	全局 token 直接交互
归纳偏置	强	弱
数据需求	小数据也较稳	更依赖大规模预训练
空间结构	天然保留	需要位置编码
多尺度特征	天然有	原始 ViT 不明显
适合任务	分类、检测、分割都成熟	分类强，检测分割需改造

简单说：

CNN 更像“从局部纹理逐步组合成整体”
ViT 更像“把图像切成 token 后做全局关系建模”

## 十八、ViT 为什么需要大数据？

ViT 的图像归纳偏置比 CNN 弱。

CNN 天然有：

局部连接
权重共享
平移等变性

这些设计本身就很适合图像。

而 ViT 更通用，它没有那么强的图像先验。

这意味着：

CNN 在小数据上更容易学到合理图像特征
ViT 需要更多数据来学习这些结构规律

所以原始 ViT 如果只在中等规模数据集上从零训练，效果可能不如 ResNet。

但如果在超大规模数据集上预训练，ViT 的性能可以非常强。

这也是为什么 ViT 经常和大规模预训练联系在一起。

## 十九、ViT 的优点

ViT 的优点主要有：

1. 全局建模能力强
2. 结构简洁，直接使用标准 Transformer Encoder
3. 可扩展性好，模型变大后性能提升明显
4. 适合大规模预训练
5. 能和 NLP、多模态模型共享 Transformer 架构思想
6. 是 CLIP、DINO、MAE、SAM 等模型的重要基础

尤其是大模型时代，ViT 的统一架构优势非常明显。

例如：

CLIP 用 ViT 作为视觉编码器
MAE 用 ViT 做自监督重建
SAM 用 ViT 做图像编码器
DINO 用 ViT 学习自监督视觉特征
二十、ViT 的缺点

ViT 也有明显不足。

### 1. 小数据集上不如 CNN 稳

如果没有足够数据或预训练权重，ViT 可能不如 ResNet、ConvNeXt 稳定。

### 2. 计算量随 token 数平方增长

Self-Attention 复杂度是：

O(N²)

如果输入分辨率提高，token 数增加，计算量会快速变大。

这也是为什么原始 ViT 不太适合直接处理高分辨率检测和分割任务。

### 3. 原始 ViT 缺少层级多尺度特征

检测和分割需要多尺度特征：

小目标需要高分辨率浅层特征
大目标需要深层语义特征

CNN 天然有：

C2, C3, C4, C5

而原始 ViT 输出通常是单尺度 token。

所以后续才有 Swin Transformer、PVT、SegFormer 等更适合密集预测任务的视觉 Transformer。

### 4. 对位置编码和输入尺寸比较敏感

ViT 使用固定长度 position embedding。

如果 fine-tune 时输入尺寸改变，patch 数量改变，position embedding 需要插值调整。

## 二十一、ViT 和 Swin Transformer 的区别
对比项	ViT	Swin Transformer
Attention 范围	全局 attention	窗口 attention
计算复杂度	token 数平方级	更适合高分辨率
特征结构	原始 ViT 单尺度	层级多尺度
下采样	不明显	Patch Merging
位置关系	位置编码	相对位置偏置
检测分割适配	需要改造	更自然
代表用途	分类、自监督、多模态	分类、检测、分割 backbone

简单理解：

ViT 是最基础、最直接的视觉 Transformer
Swin 是为了视觉任务工程化改造后的层级 Transformer
## 二十二、ViT 和 ConvNeXt 的区别

ConvNeXt 是现代 CNN，ViT 是 Transformer。

对比项	ViT	ConvNeXt
核心操作	Self-Attention	Depthwise Conv
建模方式	token 全局交互	局部卷积堆叠
图像归纳偏置	弱	强
数据需求	更依赖预训练	相对更稳
结构来源	Transformer	CNN + Transformer 设计经验
高分辨率成本	较高	CNN 更友好
多尺度特征	原始 ViT 不明显	天然层级结构

简单说：

ViT 是把图像当序列处理
ConvNeXt 是把 CNN 现代化
## 二十三、ViT 和 ResNet 的区别
对比项	ResNet	ViT
架构	CNN	Transformer
核心模块	残差卷积块	Transformer Encoder
基本输入	像素局部区域	patch token
局部性	强	需要学习
全局关系	多层卷积逐渐获得	self-attention 直接建模
训练数据	小数据较稳	大数据预训练更强
下游任务	非常成熟	需要对应改造或预训练

ResNet 更适合稳定 baseline。

ViT 更适合大规模预训练和全局建模。

## 二十四、ViT 在图像分类中的使用

ViT 最经典的任务是图像分类。

流程：

输入图像
↓
Resize 到固定尺寸
↓
Patch Embedding
↓
Transformer Encoder
↓
CLS token
↓
分类头
↓
输出类别

训练损失通常是：

CrossEntropyLoss

如果使用 PyTorch：

criterion = nn.CrossEntropyLoss()

输出：

[B, num_classes]

标签：

[B]

## 二十五、ViT 在目标检测中的使用

原始 ViT 不太直接适合检测，因为它缺少多尺度特征。

但可以作为 backbone 用于检测框架。

常见做法：

ViT backbone
+
FPN / feature pyramid
+
检测头

或者用于 DETR 系列：

ViT / Transformer backbone
+
Transformer decoder
+
object queries

不过在检测任务中，更常见的是使用：

Swin Transformer
PVT
ConvNeXt
ResNet-FPN

因为它们更自然地输出多尺度特征。

## 二十六、ViT 在语义分割中的使用

语义分割需要像素级输出，而原始 ViT 输出 token 序列，所以需要 decoder。

常见结构：

ViT Encoder
+
Decoder / Upsampling Head
+
Pixel-wise Prediction

代表模型：

SETR
Segmenter
DPT
Mask2Former with ViT/Swin

其中 SETR 的思想就是把语义分割看成序列到序列预测，用 Transformer Encoder 提取全局特征，再用 decoder 恢复像素级输出。

## 二十七、ViT 在自监督学习中的作用

ViT 非常适合自监督预训练。

代表方法：

MAE
DINO
BEiT
iBOT
MoCo v3
MAE

MAE 的思路是：

随机遮住大部分 patch
只看少量可见 patch
让模型重建被遮住的图像 patch

例如遮住：

75% patch

模型根据剩下的 25% 去重建整张图。

ViT 很适合 MAE，因为它本来就是 patch token 结构。

DINO

DINO 是自监督表征学习方法，可以让 ViT 学到很强的视觉语义表示。

DINO 训练后的 ViT attention map 经常能自动关注物体区域，这也是 ViT 很有代表性的现象。

## 二十八、DeiT：小数据训练 ViT 的改进

原始 ViT 很依赖大数据。

DeiT 的目标是：

在 ImageNet 这样相对较小的数据集上训练出强 ViT

它引入了更强的数据增强、训练策略和知识蒸馏。

DeiT 中还有一个特殊 token：

distillation token

它通过教师模型监督，让学生 ViT 学得更好。

可以理解为：

ViT：大数据预训练效果强
DeiT：让 ViT 在较小数据上也能训练得不错

## 三十一、ViT 核心总结

核心说明：

ViT，也就是 Vision Transformer，是一种把 Transformer 应用于图像分类的模型。它的核心思想是把输入图像划分成固定大小的 patch，每个 patch 展平成向量后经过线性映射变成 patch embedding，然后像 NLP 中的 token 一样输入 Transformer Encoder。为了保留空间位置信息，ViT 会给每个 patch token 加上 position embedding，并额外加入一个可学习的 CLS token。经过多层 Transformer Encoder 后，取 CLS token 的输出接分类头进行图像分类。

ViT 和 CNN 最大的区别在于，CNN 通过卷积逐层提取局部特征并扩大感受野，而 ViT 通过 self-attention 让所有 patch token 之间直接建立关系，因此全局建模能力更强。但 ViT 的图像归纳偏置比 CNN 弱，所以通常更依赖大规模数据预训练。在分类任务上 ViT 很强，但原始 ViT 缺少层级多尺度特征，所以在检测和分割任务中通常会使用 Swin Transformer、PVT 或其他层级视觉 Transformer 变体。

## 三十二、延伸知识：ViT 为什么要切 patch？

核心说明：

因为 Transformer 处理的是序列，而图像是二维网格结构。ViT 通过把图像切成固定大小的 patch，把每个 patch 展平并映射成一个 token，这样就可以把图像转换成 token 序列输入 Transformer。patch size 决定了 token 数量，patch 越小，保留的细节越多，但 self-attention 计算量也越大；patch 越大，计算量更小，但会损失更多局部细节。

## 三十三、延伸知识：ViT 为什么需要位置编码？

核心说明：

Transformer 的 self-attention 本身不包含空间顺序信息。如果只输入 patch embedding，模型不知道每个 patch 来自图像的哪个位置，也不知道 patch 之间的空间关系。图像任务非常依赖空间结构，所以 ViT 需要给每个 patch token 加上 position embedding，让模型知道不同 patch 在图像中的位置。

## 三十四、延伸知识：ViT 和 CNN 有什么区别？

核心说明：

CNN 依靠卷积操作提取局部特征，通过多层堆叠逐渐扩大感受野，具有很强的图像归纳偏置，比如局部连接和权重共享，因此在小数据场景下比较稳定。ViT 则把图像切成 patch token，用 self-attention 建模 token 之间的关系，可以直接捕获长距离依赖和全局上下文，但它的图像先验较弱，更依赖大规模预训练。简单说，CNN 更擅长局部模式建模，ViT 更擅长全局关系建模。

## 三十五、延伸知识：ViT 为什么不如 Swin 适合检测分割？

核心说明：

原始 ViT 通常输出单尺度 token 表示，而检测和分割需要多尺度特征来处理不同大小的目标，尤其是小目标和边界细节。并且 ViT 使用全局 self-attention，输入分辨率高时计算量会随着 token 数平方增长，不适合直接处理高分辨率图像。Swin Transformer 通过窗口 attention 降低计算量，通过 patch merging 形成层级特征，所以更自然地适合检测和分割任务。