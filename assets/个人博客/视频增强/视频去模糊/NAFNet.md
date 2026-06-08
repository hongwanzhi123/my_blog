# NAFNet

NAFNet 全名是 Nonlinear Activation Free Network，中文可以理解成 “无非线性激活函数的图像复原网络”。它来自论文 Simple Baselines for Image Restoration，主要用于 图像去模糊、图像去噪、图像复原 等任务。它最有意思的地方是：作者发现图像复原网络里很多常用的复杂模块和非线性激活函数不一定必要，甚至可以用更简单的乘法门控替代，因此提出了一个结构很简单但效果很强的网络。论文报告 NAFNet 在 GoPro 图像去模糊上达到 33.69 dB PSNR，在 SIDD 图像去噪上达到 40.30 dB PSNR，同时计算成本明显低于当时一些更复杂的方法。

## 1. NAFNet 解决的是什么问题

NAFNet 属于 image restoration 图像复原模型。它主要处理这类问题：

退化图像 x
    ↓
复原网络
    ↓
干净图像 y

常见任务包括：

图像去模糊：blurred image → sharp image
图像去噪：noisy image → clean image
JPEG 去伪影：compressed image → clean image
图像增强：degraded image → restored image

在工程上，NAFNet 最常被用在：

图片去模糊
图片去噪
视频逐帧去模糊
视频逐帧去噪
低照度增强后的细节恢复
作为视频增强 pipeline 的单帧 baseline

注意一点：NAFNet 本身不是视频模型。它不显式利用前后帧，不做光流、不做时序传播、不做多帧对齐。如果你把它用于视频，就是逐帧处理：

第 1 帧 → NAFNet
第 2 帧 → NAFNet
第 3 帧 → NAFNet
...

这样部署简单，但可能出现时间闪烁。

## 2. NAFNet 的核心思想

NAFNet 的核心思想可以概括成一句话：

图像复原不一定需要复杂的非线性激活函数，很多非线性表达可以通过简单的乘法门控实现。

传统 CNN / Transformer 网络里经常有：

ReLU
GELU
Sigmoid
Softmax
Attention
复杂门控
复杂归一化

NAFNet 的作者做了大量简化实验后发现：对于图像复原任务，像 ReLU、GELU、Sigmoid、Softmax 这类常见非线性激活函数并不是不可替代的；它们可以被乘法或更简单的结构替换，甚至直接移除。论文摘要中明确指出，作者把这些非线性激活函数替换为乘法或移除，由此得到 NAFNet。

这就是名字里的 Nonlinear Activation Free：

Nonlinear Activation Free
= 不使用传统非线性激活函数

但这里不要误解为“整个网络完全没有非线性能力”。
NAFNet 没有 ReLU/GELU/Sigmoid/Softmax 这类显式激活函数，但它通过 SimpleGate 乘法门控 引入了非线性表达能力。

## 3. 为什么图像复原可以不用复杂激活函数

分类任务、检测任务里，强非线性很重要，因为模型要做复杂的语义判别。但图像复原任务有一些特殊性：

输入和输出高度对齐
输出图像和输入图像结构相似
任务更偏像素级映射
局部细节、纹理、边缘很重要
过强非线性可能破坏低层信息

比如图像去模糊：

模糊图像和清晰图像的主体内容一样
只是边缘、纹理、高频细节需要恢复

图像去噪：

带噪图和干净图大部分结构一样
主要区别是噪声残差

所以图像复原网络不一定需要非常复杂的语义建模结构，反而需要：

稳定的低层特征传递
较大的感受野
通道间信息交互
局部细节恢复能力
轻量高效的结构

NAFNet 正是围绕这些需求设计的。

## 4. NAFNet 的整体架构

NAFNet 整体上是一个 U-Net 风格的 encoder-decoder 架构。

可以理解成：

输入退化图像
    ↓
浅层卷积提取特征
    ↓
Encoder：逐级下采样，扩大感受野
    ↓
Middle Blocks：瓶颈层特征处理
    ↓
Decoder：逐级上采样，恢复分辨率
    ↓
Skip Connection：融合 encoder 的细节特征
    ↓
输出复原图像

更具体一点：

Input Image
    ↓
Intro Conv
    ↓
Encoder Stage 1: NAFBlocks
    ↓ Downsample
Encoder Stage 2: NAFBlocks
    ↓ Downsample
Encoder Stage 3: NAFBlocks
    ↓ Downsample
Encoder Stage 4: NAFBlocks
    ↓
Middle NAFBlocks
    ↓
Decoder Stage 4: Upsample + Skip + NAFBlocks
    ↓
Decoder Stage 3: Upsample + Skip + NAFBlocks
    ↓
Decoder Stage 2: Upsample + Skip + NAFBlocks
    ↓
Decoder Stage 1: Upsample + Skip + NAFBlocks
    ↓
Ending Conv
    ↓
Output Image

官方代码仓库也把 NAFNet 定位为 “without nonlinear activation functions” 的图像复原模型，并提供了代码和预训练模型。

## 5. NAFNet 的核心模块：NAFBlock

NAFNet 最核心的结构是 NAFBlock。

一个 NAFBlock 可以拆成两部分：

第一部分：局部空间建模 + 通道注意力
第二部分：前馈网络 FFN 风格的通道混合

它有点像 Transformer block 的思想：

Transformer Block:
    Attention
    FFN

NAFBlock:
    简化空间/通道混合模块
    简化 FFN 模块

但 NAFBlock 不使用 Self-Attention，也不使用 ReLU/GELU，而是用更轻量的卷积和 SimpleGate。

## 6. NAFBlock 第一部分：空间建模 + SimpleGate + 简化通道注意力

第一部分大致流程是：

输入特征 x
    ↓
LayerNorm
    ↓
1×1 Conv：通道扩张
    ↓
3×3 Depthwise Conv：局部空间建模
    ↓
SimpleGate：通道一分为二，相乘
    ↓
Simplified Channel Attention
    ↓
1×1 Conv：通道压回
    ↓
残差连接

逐个解释。

6.1 LayerNorm

NAFNet 使用 LayerNorm 稳定训练。

在图像复原任务中，BatchNorm 有时会带来问题，因为 BatchNorm 会依赖 batch 统计信息，可能影响图像细节和颜色稳定性。很多图像复原模型都倾向于不用 BatchNorm，或者使用 LayerNorm / InstanceNorm 这类方式。

NAFNet 里 LayerNorm 的作用是：

稳定特征分布
让训练更容易
减少数值不稳定
6.2 1×1 Conv：通道扩张

1×1 卷积用于通道混合。

假设输入特征是：

H × W × C

经过 1×1 Conv 后，通道数可以扩张，例如变成：

H × W × 2C

作用是：

增加通道表达能力
让不同通道的信息先混合
为后面的 SimpleGate 做准备
6.3 3×3 Depthwise Conv：局部空间建模

Depthwise Conv 是深度可分离卷积中的一部分。它和普通卷积不同：

普通卷积：

每个输出通道同时看所有输入通道和空间邻域

Depthwise Conv：

每个通道单独做空间卷积
不同通道之间不混合

它的优点是计算量低。

在 NAFNet 里，3×3 Depthwise Conv 负责：

提取局部空间纹理
感知边缘、模糊、噪声模式
增强局部恢复能力

这对图像去模糊和去噪非常关键。

6.4 SimpleGate：NAFNet 最重要的操作

SimpleGate 是 NAFNet 的灵魂。

它做的事情非常简单：

把特征在通道维度一分为二
x1, x2 = split(x)

然后逐元素相乘：
y = x1 * x2

也就是：

SimpleGate(x) = x1 ⊙ x2

这里 ⊙ 表示逐元素乘法。

为什么这个操作重要？

因为虽然 NAFNet 没有 ReLU/GELU 这种激活函数，但乘法本身会引入非线性。

比如：

y = a * b

这不是线性操作。它可以表达通道之间的交互关系。

你可以把 SimpleGate 理解成：

一半特征作为内容
另一半特征作为门控
二者相乘后，网络自动决定哪些信息通过、哪些信息抑制

这有点像：

GLU / Gated Linear Unit

但比 GLU 更简单，因为它没有 Sigmoid。

普通门控可能是：

y = x1 * sigmoid(x2)

SimpleGate 直接变成：

y = x1 * x2

去掉了 Sigmoid。

这就是 NAFNet 的简洁思想。

6.5 Simplified Channel Attention：简化通道注意力

很多图像复原模型会使用通道注意力，比如 SE Block：

Global Average Pooling
    ↓
MLP
    ↓
Sigmoid
    ↓
通道加权

NAFNet 进一步简化，不使用 Sigmoid，也不使用复杂 MLP。

它使用类似：

Global Average Pooling
    ↓
1×1 Conv
    ↓
通道加权

作用是：

让网络知道哪些通道更重要
增强有用特征
抑制无用特征

比如在去模糊任务中，有些通道可能更关注边缘，有些通道更关注纹理，有些通道更关注低频颜色。通道注意力可以动态调节这些信息。

6.6 残差连接 + 可学习缩放参数

NAFBlock 中有残差连接：

输出 = 输入 + β * 分支输出

其中 β 是可学习参数，通常初始化为 0 或很小。

这样做的好处是：

训练初期网络接近恒等映射
避免深层网络一开始破坏输入
训练更稳定

图像复原任务里，输入图和输出图通常差异不是特别大，所以残差学习非常合理。

## 7. NAFBlock 第二部分：简化 FFN

NAFBlock 的第二部分类似 Transformer 中的 FFN，但也做了简化。

大致流程：

输入特征
    ↓
LayerNorm
    ↓
1×1 Conv：通道扩张
    ↓
SimpleGate
    ↓
1×1 Conv：通道压回
    ↓
残差连接

它主要负责通道维度的信息交互。

和 Transformer FFN 对比：

Transformer FFN:
Linear → GELU → Linear

NAFBlock FFN:
1×1 Conv → SimpleGate → 1×1 Conv

也就是说，NAFNet 用 SimpleGate 取代了 GELU。

## 8. NAFBlock 可以这样整体理解

一个 NAFBlock 的作用是：

输入特征
    ↓
局部空间建模：Depthwise Conv
    ↓
非线性门控：SimpleGate
    ↓
通道重标定：Simplified Channel Attention
    ↓
通道混合：1×1 Conv FFN
    ↓
残差输出

它的设计目标是：

保留图像复原需要的局部细节建模
保留通道交互能力
减少不必要的复杂激活函数
降低计算量
提升工程可用性

所以 NAFBlock 可以看成一种：

极简但有效的图像复原基础模块。

## 9. NAFNet 为什么适合去模糊

图像去模糊需要恢复：

边缘
纹理
高频细节
清晰结构

模糊的本质是高频损失和边缘扩散。

NAFNet 对去模糊有效，主要因为：

1. U-Net 架构有大感受野，可以感知模糊范围
2. Encoder-decoder 能同时建模低频结构和高频细节
3. Depthwise Conv 提取局部边缘和纹理
4. SimpleGate 增强非线性表达
5. Skip Connection 保留浅层细节
6. 残差学习让网络重点恢复退化部分

所以它可以较好地从模糊图中恢复清晰边缘。

论文报告 NAFNet 在 GoPro 去模糊 benchmark 上取得 33.69 dB PSNR，并且相对当时方法用更低计算成本达到更高结果。

## 10. NAFNet 为什么适合去噪

图像去噪的关键是区分：

真实细节
随机噪声

NAFNet 对去噪有效，原因包括：

1. 多尺度结构可以区分局部噪声和全局结构
2. Depthwise Conv 建模局部纹理
3. Channel Attention 可以抑制噪声相关通道
4. SimpleGate 可以根据内容自适应筛选特征
5. 残差结构适合学习噪声残差

论文报告 NAFNet 在 SIDD 图像去噪 benchmark 上达到 40.30 dB PSNR，同时计算成本低于一些复杂模型。

## 11. NAFNet 和 U-Net 的关系

NAFNet 的整体架构像 U-Net，但核心 block 更强。

普通 U-Net：

Conv + ReLU
Downsample
Upsample
Skip Connection

NAFNet：

NAFBlock
Downsample
Upsample
Skip Connection

也就是说：

NAFNet = U-Net 框架 + NAFBlock

U-Net 负责多尺度结构，NAFBlock 负责每个尺度上的特征复原能力。

## 12. NAFNet 和 Restormer 的区别

Restormer 是 Transformer 图像复原模型，NAFNet 是简化 CNN 风格模型。

对比	NAFNet	Restormer
主体结构	CNN / U-Net / NAFBlock	Transformer
核心操作	Depthwise Conv + SimpleGate	Multi-Dconv Head Transposed Attention
是否用激活函数	去掉传统激活	使用更复杂模块
复杂度	更简单、工程友好	表达能力强但更复杂
适合	高效图像复原 baseline	高质量图像复原
部署	相对容易	相对复杂

如果你做工程项目，NAFNet 的优势是：

结构简单
推理稳定
代码清楚
容易改成轻量版
适合做 baseline

Restormer 更适合追求高质量和研究指标。

## 13. NAFNet 和 MPRNet 的区别

MPRNet 是多阶段图像复原网络。它强调：

Stage 1 粗恢复
Stage 2 中间恢复
Stage 3 精修
跨阶段特征融合

NAFNet 则强调：

不用复杂多阶段
用一个简单高效 backbone
通过 NAFBlock 达到强性能
对比	NAFNet	MPRNet
思路	简单 baseline	多阶段渐进恢复
结构复杂度	较低	较高
核心贡献	无激活函数 NAFBlock	多阶段特征交互
工程部署	更方便	更复杂
面试关键词	SimpleGate, activation-free	multi-stage, progressive restoration

## 14. NAFNet 和 DeblurGAN-v2 的区别

DeblurGAN-v2 偏感知质量和生成式去模糊，NAFNet 偏保真复原。

对比	NAFNet	DeblurGAN-v2
训练目标	通常偏 L1/PSNR	GAN + 感知质量
输出风格	稳定、保真	更锐利，但可能假细节
适合	去模糊、去噪 benchmark / 工程稳定	主观锐化、实时视觉增强
风险	可能偏平滑	可能 hallucination
视频逐帧	相对稳	更容易纹理跳动

如果场景要求内容真实性，比如工业检测、监控增强、文字恢复，NAFNet 这类保真模型通常比 GAN 模型更安全。

## 15. NAFNet 用于视频增强时怎么理解

NAFNet 不是视频模型，但工程中可以作为视频增强 pipeline 的一个模块。

例如视频去模糊：

输入视频
    ↓
ffmpeg 拆帧
    ↓
NAFNet 逐帧去模糊
    ↓
ffmpeg 合成视频

视频去噪：

输入视频
    ↓
NAFNet 逐帧去噪
    ↓
合成视频

优点：

部署简单
不需要光流
不需要多帧缓存
延迟低
适合快速 demo

缺点：

不能利用多帧信息
可能出现时间闪烁
运动区域无法从邻近帧补细节
每帧独立恢复，纹理可能不一致

所以在视频增强面试中，你可以这样定位 NAFNet：

NAFNet 是很好的逐帧图像复原 baseline，但如果要做真正的视频增强，还需要引入时间一致性、多帧融合或视频模型，比如 EDVR、BasicVSR++、RVRT。

## 16. 工程部署中的优点

NAFNet 在工程上比较受欢迎，主要因为：

16.1 结构简单

没有复杂 attention，没有光流，没有 DCN，没有递归传播。

主要组件是：

1×1 Conv
Depthwise Conv
LayerNorm
SimpleGate
Channel Attention
PixelShuffle / Upsample
Residual Connection

这些模块大多容易部署。

16.2 速度和效果平衡好

论文强调 NAFNet 不仅效果强，而且计算效率高。比如论文摘要中提到在 GoPro 去模糊上超过之前 SOTA，同时只使用其一小部分计算成本。

16.3 适合改轻量版

你可以通过调整这些参数控制复杂度：

width：基础通道数
enc_blk_nums：encoder 每层 block 数
dec_blk_nums：decoder 每层 block 数
middle_blk_num：瓶颈 block 数

如果要部署到移动端，可以减少 width 和 block 数。

16.4 适合做项目 baseline

如果你要做视频增强项目，NAFNet 很适合做：

逐帧去模糊 baseline
逐帧去噪 baseline
和 EDVR / RVRT / BasicVSR++ 对比
低照度增强后的后处理模块

## 17. NAFNet 的局限
17.1 不是视频模型

它没有时间建模，所以用于视频会有闪烁风险。

如果视频任务要求时间一致性，NAFNet 不如：

EDVR
BasicVSR++
RealBasicVSR
VRT
RVRT
FastDVDnet
EMVD

这些真正的视频模型。

17.2 对真实复杂退化需要重新训练

如果训练数据是标准去模糊或去噪数据，直接用于真实低清视频可能不稳。

真实视频常见退化是混合的：

噪声
压缩伪影
运动模糊
低照度
过锐化
色偏
超分退化

NAFNet 如果没有针对这些退化训练，可能只能解决其中一部分。

17.3 不擅长生成不存在的细节

NAFNet 偏保真，不像 GAN 或扩散模型那样强生成。对于极度模糊、极低分辨率、强压缩图像，它可能恢复得比较保守。

这在很多工程场景反而是优点，因为不容易生成假细节。

## 18. 面试里怎么讲 NAFNet


NAFNet 是一个用于图像复原的简单高效 baseline，全名是 Nonlinear Activation Free Network。它的核心观点是，在图像去噪、去模糊这类低层视觉任务中，传统的 ReLU、GELU、Sigmoid、Softmax 等非线性激活函数不是必须的，可以用更简单的乘法门控来替代。NAFNet 整体采用 U-Net 风格的 encoder-decoder 结构，核心模块是 NAFBlock。NAFBlock 主要由 LayerNorm、1×1 卷积、3×3 depthwise convolution、SimpleGate、简化通道注意力和残差连接组成。其中 SimpleGate 会把特征在通道维度一分为二，然后逐元素相乘，从而在没有显式激活函数的情况下引入非线性表达能力。NAFNet 的优势是结构简单、计算效率高、效果稳定，在图像去模糊和去噪 benchmark 上表现很强。工程上它很适合作为逐帧图像复原 baseline，比如视频逐帧去模糊或去噪；但它本身不利用时序信息，所以用于视频时可能出现闪烁，如果要做真正的视频增强，还需要结合 EDVR、BasicVSR++、RVRT 这类多帧视频模型。

最后记住一句话：

NAFNet 的本质是：用 U-Net 多尺度框架和 NAFBlock 替代复杂激活与注意力，通过 SimpleGate 这种极简乘法门控实现高效图像复原。