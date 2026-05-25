# Transformer

## 一、Transformer 是什么

Transformer 是一种深度学习网络结构，最早主要用于自然语言处理任务，比如机器翻译、文本生成、问答系统等。

它的核心思想是：

不再像 RNN 那样按顺序一个词一个词处理，而是通过 Attention 机制，让序列中的每个位置都可以直接关注其他位置，从而建模全局依赖关系。

简单说，Transformer 的核心能力是：

让模型知道：当前这个 token 应该重点看哪些其他 token。

## 二、为什么 Transformer 重要？

在 Transformer 出现之前，序列建模主要依赖：

RNN
LSTM
GRU

这些模型的特点是按顺序处理数据：

第 1 个词 → 第 2 个词 → 第 3 个词 → ...

问题是：

1. 难以并行训练
2. 长距离依赖容易丢失
3. 序列越长，信息传递越困难

Transformer 解决了这些问题。

它的优势是：

可以并行处理整个序列
更擅长捕捉长距离依赖
训练效率更高
适合大规模预训练
可以扩展到文本、图像、语音、多模态等任务

现在很多主流模型都基于 Transformer，例如：

BERT
GPT
T5
ViT
Swin Transformer
DETR
CLIP
SAM

## Transformer 的整体结构

原始 Transformer 是为机器翻译设计的，所以它包含两部分：

Encoder 编码器
Decoder 解码器

整体结构是：

输入序列
↓
Embedding
↓
Positional Encoding
↓
Encoder 堆叠
↓
Decoder 堆叠
↓
Linear + Softmax
↓
输出序列

可以理解为：

Encoder：理解输入
Decoder：生成输出

比如机器翻译：

输入：I love you
Encoder：理解英文句子
Decoder：生成：我爱你

## 四、Transformer 的三种典型结构

虽然原始 Transformer 是 Encoder-Decoder 结构，但后来发展出了三类主流结构。

1. Encoder-only：只使用 Encoder

代表模型：

BERT
RoBERTa
ALBERT
DeBERTa

适合任务：

文本分类
情感分析
实体识别
文本匹配
阅读理解
语义理解

特点：

擅长理解，不擅长自回归生成
可以同时看到左右上下文

例如：

输入：我今天很开心
输出：情感 = 正向
2. Decoder-only：只使用 Decoder

代表模型：

GPT 系列
LLaMA
Qwen
DeepSeek
Claude

适合任务：

文本生成
对话
代码生成
续写
推理
Agent

特点：

从左到右生成
当前 token 只能看到前面的 token
不能看到未来 token

例如：

输入：今天天气很好，我想
输出：出去散步。

GPT 这类模型就是 Decoder-only Transformer。

3. Encoder-Decoder：编码器 + 解码器

代表模型：

原始 Transformer
T5
BART
MarianMT

适合任务：

机器翻译
文本摘要
文本改写
问答生成
输入到输出的转换任务

特点：

Encoder 负责理解输入
Decoder 负责根据输入生成输出

例如：

输入：一篇长文章
输出：一段摘要

## 五、Transformer 的核心：Attention 机制

Transformer 最核心的部分是：

Self-Attention

也就是自注意力机制。

它的作用是：

对序列中的每个位置，计算它和其他位置之间的相关性，然后根据相关性加权融合其他位置的信息。

## 六、Q、K、V 是什么？

Self-Attention 里有三个非常重要的概念：

Query
Key
Value

简称：

Q, K, V

可以这样理解：

Query：我想找什么信息
Key：我拥有什么信息的标签
Value：我真正提供的信息内容

对于序列中的每个 token，模型都会生成自己的 Q、K、V。

然后某个 token 的 Query 会和所有 token 的 Key 做匹配，得到注意力权重，再用这些权重加权所有 token 的 Value。

## 七、Self-Attention 的计算过程

Self-Attention 可以分成 4 步。

第一步：输入 token embedding

假设输入序列长度为 n，每个 token 的向量维度是 d_model。

输入矩阵：

X ∈ R^(n × d_model)

例如：

n = 5
d_model = 512

那么：

X.shape = [5, 512]
第二步：生成 Q、K、V

通过三个线性层得到：

Q = XWq
K = XWk
V = XWv

其中：

Q：Query 矩阵
K：Key 矩阵
V：Value 矩阵

它们的 shape 通常是：

Q.shape = [n, d_k]
K.shape = [n, d_k]
V.shape = [n, d_v]
第三步：计算注意力分数

用 Q 和 K 做点乘：

score = QK^T

含义是：

每个 token 和每个 token 的相关性

如果序列长度是 n，那么：

QK^T.shape = [n, n]

这个 [n, n] 矩阵就是注意力分数矩阵。

例如：

第 i 行第 j 列表示：
第 i 个 token 对第 j 个 token 的关注程度
第四步：缩放、Softmax、加权求和

完整公式是：

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

其中：

QK^T：计算相关性
sqrt(d_k)：缩放，防止数值过大
softmax：转成注意力权重
V：根据权重加权求和

所以最终输出是：

每个 token 融合上下文信息后的新表示

## 八、为什么要除以 sqrt(d_k)？

因为 Q 和 K 做点乘时，如果维度 d_k 很大，点乘结果可能会变得很大。

如果数值太大，经过 Softmax 后会变得非常极端：

某个位置接近 1
其他位置接近 0

这样梯度会变小，训练不稳定。

所以除以：

sqrt(d_k)

可以让注意力分数保持在更合理的范围内。

这也是为什么这个机制叫：

Scaled Dot-Product Attention

也就是缩放点积注意力。

## 十、Multi-Head Attention 多头注意力

单个 Attention 可以理解为一种“关注方式”。

但是一句话里的关系可能有很多种：

语法关系
指代关系
位置关系
语义关系
实体关系
情感关系

如果只有一个 Attention，表达能力有限。

所以 Transformer 使用：

Multi-Head Attention

也就是多头注意力。

它的思想是：

用多个 Attention 头，从不同角度学习 token 之间的关系，然后把结果拼接起来。

例如有 8 个头：

Head 1：关注语法关系
Head 2：关注主谓关系
Head 3：关注指代关系
Head 4：关注实体关系
...

实际训练中不一定能这么明确解释每个头，但可以这样理解。

Multi-Head Attention 计算流程
输入 X
↓
分别映射出多组 Q、K、V
↓
每个 head 单独做 Attention
↓
把多个 head 的结果 concat
↓
再经过一个线性层输出

公式可以写成：

head_i = Attention(Q_i, K_i, V_i)

然后：

MultiHead(Q,K,V) = Concat(head_1, ..., head_h)Wo

其中：

h 是注意力头数量
Wo 是输出映射矩阵

## 十一、Positional Encoding 位置编码

Transformer 本身的 Self-Attention 不像 RNN 那样天然有顺序。

RNN 是按顺序处理的：

第 1 个词 → 第 2 个词 → 第 3 个词

所以它天然知道顺序。

但是 Transformer 是同时看整个序列：

所有 token 同时输入

如果不加入位置信息，它不知道：

“我 爱 你”

和：

“你 爱 我”

有什么区别。

所以 Transformer 需要加入：

Positional Encoding

也就是位置编码。

位置编码的作用

输入给 Transformer 的不是单纯 token embedding，而是：

输入表示 = Token Embedding + Position Embedding

也就是：

这个词是什么 + 这个词在第几个位置

这样模型才能知道词序。

常见位置编码方式

常见方式有：

正弦余弦位置编码
可学习位置编码
相对位置编码
旋转位置编码 RoPE
ALiBi

原始 Transformer 使用的是：

Sinusoidal Positional Encoding

也就是正弦余弦位置编码。

现代大语言模型常见的是：

RoPE 旋转位置编码

它更适合长文本建模。

## 十二、Feed Forward Network 前馈网络

每个 Transformer Block 中，除了 Attention，还有一个：

Feed Forward Network，FFN

它通常是一个两层 MLP：

Linear
↓
Activation
↓
Linear

公式：

FFN(x) = max(0, xW1 + b1)W2 + b2

现代模型中激活函数可能不是 ReLU，而是：

GELU
SwiGLU
SiLU

FFN 的作用是：

对每个 token 的表示进行非线性变换
增强模型表达能力

注意：

Attention 负责 token 之间的信息交互
FFN 负责对每个 token 自身的特征进行加工

可以这样理解：

Attention：让每个 token 和其他 token 交流
FFN：每个 token 自己思考加工

## 十三、Residual Connection 残差连接

Transformer 中大量使用残差连接。

结构类似：

x + Attention(x)
x + FFN(x)

残差连接的作用：

缓解深层网络训练困难
帮助梯度传播
保留原始信息
让模型更容易学习增量变化

如果没有残差连接，Transformer 堆得很深时会很难训练。

## 十四、Layer Normalization 层归一化

Transformer 中常用的是：

LayerNorm

而 CNN 中常用的是：

BatchNorm

LayerNorm 的作用是：

稳定训练
加快收敛
改善梯度传播

一个 Transformer Block 常见结构是：

x = x + SelfAttention(LayerNorm(x))
x = x + FFN(LayerNorm(x))

这种叫：

Pre-LN Transformer

也有一种结构是：

x = LayerNorm(x + SelfAttention(x))
x = LayerNorm(x + FFN(x))

这种叫：

Post-LN Transformer

现在很多大模型更常用 Pre-LN，因为深层训练更稳定。

## 十五、一个 Transformer Block 长什么样？

一个标准 Transformer Block 可以理解为：

输入 x
↓
LayerNorm
↓
Multi-Head Self-Attention
↓
残差连接
↓
LayerNorm
↓
Feed Forward Network
↓
残差连接
↓
输出 x

也就是：

x = x + MultiHeadAttention(LN(x))
x = x + FFN(LN(x))

多个 Block 堆叠起来，就形成了深层 Transformer。

例如：

12 层 Transformer
24 层 Transformer
32 层 Transformer
80 层 Transformer

## 十六、Encoder 结构详解

Transformer Encoder 的每一层主要包括：

Multi-Head Self-Attention
Feed Forward Network
Residual Connection
LayerNorm

Encoder 的特点是：

每个 token 可以看到整个输入序列

所以它适合理解任务。

例如：

文本分类
命名实体识别
句子匹配
图像分类
语义分割编码器

BERT 就是典型的 Encoder-only 模型。

## 十七、Decoder 结构详解

Transformer Decoder 每一层通常包括三个部分：

Masked Multi-Head Self-Attention
Cross-Attention
Feed Forward Network

其中：

Masked Self-Attention：看已经生成的内容
Cross-Attention：看 Encoder 的输出
FFN：进一步加工特征
什么是 Masked Self-Attention？

Decoder 在生成文本时，必须从左到右生成。

例如：

我 / 喜欢 / 机器 / 学习

在生成“机器”的时候，模型只能看到：

我 / 喜欢

不能提前看到：

学习

否则就是作弊。

所以需要 Mask，把未来位置遮住。

这就是：

Causal Mask

或者叫：

Look-ahead Mask

GPT 这类模型就使用 Masked Self-Attention。

什么是 Cross-Attention？

Cross-Attention 用在 Encoder-Decoder 模型里。

Self-Attention 是：

Q、K、V 都来自同一个序列

Cross-Attention 是：

Q 来自 Decoder
K、V 来自 Encoder 输出

含义是：

Decoder 在生成输出时，去关注 Encoder 理解到的输入信息

比如机器翻译：

Encoder 理解英文句子
Decoder 生成中文句子
Decoder 每生成一个中文词，就去看英文输入中相关的部分

## 十八、Transformer 和 RNN 的区别
对比项	RNN/LSTM	Transformer
处理方式	按顺序处理	并行处理
长距离依赖	容易衰减	更擅长建模
训练效率	较低	较高
核心机制	隐状态递归传递	Self-Attention
序列顺序	天然有顺序	需要位置编码
长序列计算	相对线性	Attention 通常是平方复杂度
代表模型	LSTM、GRU	BERT、GPT、T5

简单说：

RNN 是一步一步读句子；
Transformer 是同时看完整句话，并计算词与词之间的关系。
### 十九、Transformer 和 CNN 的区别
对比项	CNN	Transformer
核心操作	卷积	注意力
擅长	局部特征提取	全局关系建模
归纳偏置	强，天然适合图像局部结构	相对弱，更依赖数据
参数共享	卷积核空间共享	QKV 投影共享
位置信息	卷积天然带局部位置结构	需要位置编码
小数据表现	通常更稳	通常更依赖预训练
长距离依赖	需要堆叠扩大感受野	可以直接建模任意位置关系

可以这样理解：

CNN 更像局部扫描图像；
Transformer 更像让所有位置互相交流。

在图像任务中：

CNN 擅长提取边缘、纹理、局部形状
Transformer 擅长建模全局上下文和远距离关系

现在很多视觉模型会结合二者。

## 二十三、Transformer 为什么适合大模型？

Transformer 特别适合扩展成大模型，原因包括：

结构简单统一
并行计算效率高
可以堆叠很多层
适合大规模数据预训练
Self-Attention 能建模复杂上下文关系
模型容量可以随参数量扩大而提升

相比 RNN，Transformer 更容易利用 GPU/TPU 进行并行训练。

这也是 GPT、BERT、T5 等模型成功的重要原因。

## 二十四、Transformer 在 NLP 中的应用

Transformer 在自然语言处理中几乎是主流基础结构。

常见任务包括：

文本分类
情感分析
机器翻译
文本摘要
问答系统
信息抽取
命名实体识别
文本生成
代码生成
对话系统
检索增强生成

不同结构适合不同任务：

BERT：理解类任务
GPT：生成类任务
T5/BART：输入输出转换任务
##二十五、Transformer 在图像中的应用

Transformer 不只用于文本，也可以用于图像。

最典型的是：

Vision Transformer，ViT
1. ViT 的基本思想

ViT 把图像切成一个个 patch。

例如输入图像：

224 × 224 × 3

切成：

16 × 16

的小块。

那么 patch 数量是：

224 / 16 = 14
14 × 14 = 196 个 patch

每个 patch 被拉平成向量，然后映射成 token。

也就是说：

图像 → patch 序列

然后就可以像处理文本一样输入 Transformer。

流程是：

Image
↓
Patch Embedding
↓
Position Embedding
↓
Transformer Encoder
↓
Classification Head
↓
类别预测
2. ViT 和 CNN 的区别

CNN 是用卷积提取局部特征：

卷积核滑动扫描图像

ViT 是把图像变成 patch 序列：

patch 之间通过 Self-Attention 建模关系

ViT 的优势：

全局建模能力强
适合大规模预训练
结构统一
可扩展性强

ViT 的不足：

对数据量要求较高
小数据集上不一定比 CNN 稳
缺少 CNN 的局部归纳偏置
3. Swin Transformer

Swin Transformer 是视觉 Transformer 中非常重要的结构。

它的核心思想是：

窗口注意力 Window Attention
移动窗口 Shifted Window
层级特征结构

为什么要窗口注意力？

因为图像 patch 很多，如果全局 Attention 计算量太大。

所以 Swin 只在局部窗口里做 Attention：

每个窗口内部计算注意力

然后通过移动窗口，让不同窗口之间也能交换信息。

Swin Transformer 更适合：

图像分类
目标检测
语义分割
实例分割

因为它有类似 CNN 的多尺度层级特征。

## 二十六、Transformer 在目标检测中的应用

典型模型：

DETR

DETR 把目标检测看成集合预测问题。

传统目标检测模型通常需要：

Anchor
NMS
候选框筛选
手工设计匹配规则

DETR 使用 Transformer 后，可以直接输出一组目标预测：

类别 + 边界框

它的特点是：

端到端检测
减少手工组件
使用 object query
通过匈牙利匹配进行训练

不过原始 DETR 收敛较慢，后续有很多改进版本。

## 二十七、Transformer 在图像分割中的应用

Transformer 也广泛用于分割。

常见模型：

SETR
SegFormer
MaskFormer
Mask2Former
Swin-Unet
TransUNet
SAM

它们的优势是：

全局上下文建模能力强
对复杂场景关系理解更好
适合多尺度语义融合
可以提升大目标和长距离区域一致性

但在小数据集或边缘细节任务中，CNN 或 CNN-Transformer 混合结构仍然很常见。

## 二十八、Transformer 的优点

Transformer 的优点可以总结为：

并行能力强
擅长长距离依赖建模
结构统一，适合多种任务
可扩展性强
适合大规模预训练
全局信息交互能力强
迁移学习效果好

一句话：

Transformer 通过 Self-Attention 让序列中任意位置之间可以直接建立联系，因此非常适合建模复杂上下文关系。

## 二十九、Transformer 的缺点

Transformer 也有明显缺点。

1. 计算复杂度高

标准 Attention 是：

O(n²)

长序列时显存和计算量很大。

2. 对数据量依赖较强

相比 CNN，Transformer 的归纳偏置更弱。

所以在小数据集上，如果没有预训练，效果可能不稳定。

3. 位置建模需要额外设计

Transformer 本身没有天然顺序感，需要位置编码。

4. 可解释性仍然有限

虽然 Attention 权重可以一定程度上解释模型关注区域，但不能完全等价于模型推理原因。

## 三十五、面试中怎么介绍 Transformer

Transformer 是一种基于自注意力机制的深度学习网络结构，最早用于机器翻译任务。它的核心思想是通过 Self-Attention 让序列中任意两个位置之间都可以直接建立联系，从而更好地建模长距离依赖。相比 RNN 逐步处理序列，Transformer 可以并行处理整个序列，因此训练效率更高，也更适合大规模预训练。

Transformer 的基本模块包括 Multi-Head Self-Attention、Feed Forward Network、Residual Connection 和 LayerNorm。Self-Attention 会将输入映射成 Query、Key、Value，通过 softmax(QK^T / sqrt(d_k))V 计算每个 token 对其他 token 的关注权重，并融合上下文信息。Multi-Head Attention 则是从多个子空间学习不同类型的关系。由于 Transformer 本身没有顺序信息，所以需要加入位置编码。

根据结构不同，Transformer 可以分为 Encoder-only、Decoder-only 和 Encoder-Decoder 三类。BERT 是 Encoder-only，适合理解类任务；GPT 是 Decoder-only，适合文本生成；T5 和原始 Transformer 是 Encoder-Decoder，适合翻译、摘要等输入输出转换任务。现在 Transformer 也被广泛应用到计算机视觉中，比如 ViT、Swin Transformer、DETR 和 SegFormer 等。


