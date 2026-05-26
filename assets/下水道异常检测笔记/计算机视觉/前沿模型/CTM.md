# CTM

CTM = Continuous Thought Machine（连续思维机器） 来解释。它是 Sakana AI 提出的一种神经网络架构，核心思想是：让模型不是一次前向传播直接给答案，而是在一个内部时间轴上反复“思考”，并把神经元之间随时间变化的同步关系作为表示来做预测和决策。官方介绍里也明确说，CTM 用“neural activity over time 的同步”作为采取行动的表示，并把神经时间作为基础元素引入模型。

## 一、CTM 想解决什么问题？

传统神经网络大多是这样的：

输入 x
↓
Layer 1
↓
Layer 2
↓
Layer 3
↓
输出 y

比如 CNN、ViT、Transformer，本质上都是把输入经过一系列固定层数的计算，然后输出结果。

这种方式有几个特点：

1. 计算深度通常固定
2. 每层计算结束后就进入下一层
3. 表示通常是某一时刻的 activation vector
4. 模型内部没有显式“思考时间”

CTM 的出发点是：真实大脑不是简单的一次前向传播，而是神经元活动会随时间演化，不同神经元之间的同步关系也会携带信息。所以 CTM 不是只看某一层的静态激活，而是引入一个内部时间维度，让神经活动在多个 internal ticks 中展开。论文说 CTM 的两个核心创新是：每个神经元用自己的参数处理输入历史，以及把神经同步直接作为 latent representation。

你可以先这样理解：

Transformer 更像“一层一层处理 token”；CTM 更像“模型内部持续运转一段时间，在这个过程中观察数据、更新神经状态、计算神经元同步关系，然后逐步形成答案”。

## 二、CTM 的核心设计

CTM 有三个最重要的设计：

1. Internal recurrence：内部递归时间轴
2. Neuron-Level Models：神经元级模型
3. Synchronization Representation：同步表示

官方介绍也把这三点作为 CTM 的主要思想：内部递归让“思维”展开；每个神经元有私有的 MLP 模型处理输入历史；神经活动随时间的同步关系被直接用作模型观察和预测的表示。

## 三、传统模型和 CTM 的最大区别
1. 传统模型：层之间传递 activation

例如普通神经网络：

x → h1 → h2 → h3 → y

每一层输出一个 activation vector。

这些 activation 本身就是模型表示。

2. CTM：神经元活动随时间展开

CTM 中有一个内部时间轴：

tick 1
tick 2
tick 3
...
tick T

这个时间轴不一定来自输入数据本身。也就是说，即使输入是一张静态图片，CTM 也可以在内部运行多个 tick。

例如：

输入：一张图片
↓
CTM 内部思考 1 次
↓
CTM 内部思考 2 次
↓
CTM 内部思考 3 次
↓
...
↓
输出预测

论文称这个内部维度是 decoupled from data dimensions，也就是和输入数据维度解耦。它不是图像的宽高，也不是文本 token 序列，而是模型自己内部展开思考的时间轴。

## 四、CTM 架构整体流程

CTM 一次 internal tick 大致可以分成下面几个步骤：

上一时刻 post-activations
+
上一时刻 attention output
↓
Synapse Model
↓
pre-activations
↓
保存 pre-activation history
↓
Neuron-Level Models
↓
post-activations
↓
保存 post-activation history
↓
计算 neuron-to-neuron synchronization
↓
得到 synchronization representation
↓
用于输出预测，也用于生成 attention query 观察输入数据
↓
attention output 反馈到下一 tick

官方架构图说明里也写到：synapse model 产生 pre-activations；保留 pre-activation history；Neuron-Level Models 产生 post-activations；保存 post-activation history；根据 post-activation history 计算 synchronization matrix；再选择神经元对形成 latent representations，用于输出和 attention queries。

## 五、CTM 的几个关键变量

为了更好理解，可以先看几个变量：

a^t：第 t 个 internal tick 的 pre-activations
A^t：最近 M 个 pre-activations 组成的历史
z^t：第 t 个 internal tick 的 post-activations
Z^t：从开始到当前 tick 的 post-activation history
S^t：根据 post-activation history 计算出来的 synchronization matrix
o^t：通过 attention 观察输入数据得到的 attention output

官方说明中也给出了这些符号：a^t 是 pre-activations，A^t 是最近 pre-activations 的 FIFO 历史，z^t 是 post-activations，Z^t 是 post-activation history，S^t 是 synchronization matrix。

## 六、组件 1：Synapse Model 突触模型

CTM 里的 Synapse Model 可以理解成：

负责建模神经元之间相互作用的模块。

它的输入大致来自：

上一时刻的 post-activations
+
当前从数据中注意到的信息 attention output

然后输出：

pre-activations

也就是神经元在当前 tick 接收到的“输入信号”。

官方说明里说，CTM 用一个 synapse model 在共享 latent space 中连接神经元，并产生 pre-activations；技术报告里提到他们使用了 U-Net-like / U-NET-esque 的 MLP 结构作为 synapse model。

简单理解：

Synapse Model 负责“神经元之间怎么互相传信号”

它对应生物神经网络里的“突触连接”概念，但这里当然是抽象的人工神经网络模块。

## 七、组件 2：Pre-activation History

传统神经网络中，一个神经元通常只看当前输入：

当前输入 → 激活函数 → 当前输出

CTM 不一样。

CTM 会给每个神经元保留一段历史：

a^(t-M+1), a^(t-M+2), ..., a^t

也就是最近 M 个 tick 的 pre-activation。

这个历史用 FIFO 方式保存：

新 tick 来了 → 加入最新 pre-activation
历史太长 → 丢掉最旧的

官方说明也提到，CTM 保留最近 pre-activations 的 history，并且设计为固定长度 M 的 FIFO list。

这个设计的意义是：

神经元不是只根据当前信号做反应，而是根据一段时间内收到的信号历史来决定自己的输出。

这就引入了时间动态。

## 八、组件 3：Neuron-Level Models 神经元级模型

这是 CTM 很关键的设计。

传统神经元大致是：

z = activation(w · x + b)

也就是说，每个神经元通常只是一个线性加权 + 激活函数。

CTM 中的神经元更复杂。每个神经元都有自己的小模型，叫：

Neuron-Level Model, NLM

它会读取这个神经元最近一段时间的 pre-activation history，然后输出当前的 post-activation。

可以理解为：

某个神经元最近 M 次接收到的信号
↓
这个神经元自己的 MLP
↓
当前神经元输出 z^t

论文中说，每个神经元有一个 privately parameterized NLM，用自己的参数处理 pre-activation history，并生成 post-activation；官方 GitHub 也说明每个神经元使用 unique weight parameters 来处理历史输入信号。

这和传统神经网络差别很大。

传统神经元：

简单激活函数

CTM 神经元：

一个小型时间处理模型

所以 CTM 里的神经元本身就有更复杂的动态行为。

## 九、组件 4：Post-activation History

NLM 输出当前 tick 的 post-activations：

z^t

CTM 会保存所有历史 post-activations：

Z^t = [z^1, z^2, ..., z^t]

这个历史很重要，因为 CTM 后面要用它来计算神经元之间的同步关系。

传统神经网络一般只关心当前层的 activation。

CTM 关心的是：

神经元活动在多个 tick 上如何变化

也就是：

neural activity over time

官方说明中也写到，post-activation history 会被保存，并用于计算 neuron-to-neuron synchronization。

## 十、组件 5：Synchronization Matrix 同步矩阵

这是 CTM 最核心、也最容易困惑的地方。

CTM 不直接把某个时刻的 activation vector 当作最终表示，而是看：

哪些神经元在时间上表现出同步变化。

例如有两个神经元 A 和 B，它们在多个 tick 上的活动模式很相似：

tick 1: A 高，B 高
tick 2: A 低，B 低
tick 3: A 高，B 高
tick 4: A 中，B 中

这说明 A 和 B 的活动比较同步。

另外两个神经元 C 和 D：

tick 1: C 高，D 低
tick 2: C 低，D 高
tick 3: C 高，D 低

它们就不同步，甚至可能反相关。

CTM 会根据神经元的 post-activation history 计算一个 synchronization matrix：

S^t[i, j] = neuron i 和 neuron j 在时间上的同步程度

官方说明里明确说，CTM 跟踪神经活动随时间变化，并计算神经元对之间的同步关系；这个 synchronization representation 是 CTM 用来观察数据和预测的表示。

这点非常重要：

普通网络的表示是 activation；CTM 的表示是神经元活动之间的同步关系。

## 十一、为什么同步表示重要？

你可以用一个简单比喻理解。

假设一个乐队里有很多乐器：

鼓
贝斯
钢琴
吉他
小提琴

如果你只看某一瞬间谁声音大，信息有限。

但如果你观察一段时间，会发现：

鼓和贝斯总是一起进入节奏
钢琴和小提琴在某些段落呼应
吉他和鼓在某些时刻同步

这些“随时间的配合关系”反而更能体现音乐结构。

CTM 类似：

单个神经元当前激活值

不是最核心的表示。

更核心的是：

神经元之间随时间形成的同步模式

这种同步模式被用来决定模型下一步看哪里、怎么预测、怎么行动。

## 十二、组件 6：选取神经元对形成 latent representation

完整的 synchronization matrix 可能很大。

如果有 D 个神经元，那么同步矩阵大小可能是：

D × D

这会非常大。

所以 CTM 不一定使用所有神经元对，而是选择部分 neuron pairs 来构成 latent representation。

官方说明中也提到，他们从 synchronization matrix 中选择 neuron pairs，得到 latent representations，用于输出和调制数据。

可以理解为：

同步矩阵 S^t
↓
选出一部分关键同步关系
↓
形成当前 tick 的 latent representation
## 十三、组件 7：用同步表示做输出预测

CTM 会把 synchronization representation 经过线性投影，得到输出预测。

比如图像分类任务中：

synchronization representation
↓
linear head
↓
class logits
↓
预测类别

所以最终分类不是直接基于某一层 activation，而是基于：

神经元活动历史产生的同步表示

官方说明中提到，CTM 使用从同步矩阵选出的 latent representations 来产生 outputs 和 attention queries。

## 十四、组件 8：用同步表示生成 Attention Query

CTM 不是一次性读取全部输入信息，而是可以在每个 internal tick 中根据当前状态去“观察”输入数据。

比如图像分类任务中，CTM 可以像这样工作：

tick 1：先看图像某个区域
tick 2：根据内部状态再看另一个区域
tick 3：继续调整注意位置
...
最后预测类别

论文和官方页面都展示了 CTM 在图像任务中会出现类似“look around”的注意行为；论文还提到这种行为并不是通过专门监督信号训练出来的，而是从内部动态中自然出现的。

所以 CTM 的观察过程可以理解为：

当前同步表示
↓
生成 attention query
↓
对输入数据做 cross-attention
↓
得到 attention output
↓
反馈给下一 tick 的 synapse model

这样模型就能一边思考，一边决定下一步关注什么。

## 十五、CTM 一个 tick 的完整计算流程

把上面内容串起来，一个 tick 可以写成：

1. 当前 post-activations 和 attention output 输入 Synapse Model
2. Synapse Model 产生 pre-activations a^t
3. 把 a^t 加入 pre-activation history A^t
4. 每个神经元自己的 NLM 读取对应历史 A^t
5. NLM 输出 post-activation z^t
6. 把 z^t 加入 post-activation history Z^t
7. 根据 Z^t 计算 synchronization matrix S^t
8. 从 S^t 选取部分神经元对，形成 latent representation
9. latent representation 用于：
   - 输出预测
   - 生成 attention query
10. cross-attention 观察输入数据，得到 attention output
11. attention output 参与下一 tick

这个循环会重复多个 internal ticks。

## 十六、CTM 和 RNN/LSTM 的区别

你可能会觉得 CTM 有点像 RNN。

确实，CTM 有 recurrence，但它和 RNN/LSTM 不一样。

RNN/LSTM

RNN/LSTM 通常是：

输入序列 x1, x2, x3...
↓
每一步更新 hidden state
↓
hidden state 用于输出

它的时间通常来自输入数据本身。

比如句子：

word1 → word2 → word3

或者时间序列：

t1 → t2 → t3
CTM

CTM 的时间轴是内部生成的：

internal tick 1
internal tick 2
internal tick 3
...

即使输入是静态图片，它也可以运行多个内部 tick。

此外，CTM 不是只维护一个 hidden state，而是：

每个神经元有 pre-activation history
每个神经元有自己的 NLM
模型计算 neuron-to-neuron synchronization
同步关系作为 latent representation

论文也强调，CTM 与其他 recurrent architectures 的差异在于：它对 pre-activation histories 使用每个神经元私有参数的 NLM，并直接使用 neural synchronization 作为 latent representation。

一句话：

RNN 主要是 hidden state 随时间递归；CTM 是神经元活动历史和神经元同步关系随内部时间展开。

## 十七、CTM 和 Transformer 的区别

Transformer 的核心是：

token embedding
↓
self-attention
↓
MLP
↓
多层堆叠

它的表示通常是每个 token 的向量。

例如 ViT：

图像 patch tokens
↓
Transformer blocks
↓
CLS token / pooled feature
↓
分类

而 CTM：

输入数据
↓
内部 tick 反复运行
↓
神经元活动历史
↓
同步矩阵
↓
同步表示用于观察和预测

主要区别：

对比项	Transformer	CTM
时间/深度	固定层数为主	内部 tick 可展开
表示	token activation vectors	neuron synchronization
关注输入	self-attention / cross-attention	同步表示调制 attention
神经元	通常是简单激活单元	每个神经元有 NLM
推理过程	层级前向传播	内部动态迭代

可以简单说：

Transformer 主要靠 token 之间的 attention；CTM 主要靠神经元活动随时间形成的同步关系。

## 十八、CTM 为什么适合“思考”这个说法？

“思考”不是说它真的有人类意识，而是说它具备类似下面的计算形式：

不是一次前向传播直接输出
而是在内部多个 tick 中迭代更新状态
可以逐步观察数据
可以让计算时间展开
可以根据内部动态形成最终表示

官方页面也说，CTM 引入了一个 decoupled internal dimension，让 thought 可以在人工神经系统中展开；GitHub README 也说它有一个与输入数据解耦的内部时间轴，让 neuron activity unfold。

所以“Continuous Thought Machine”里的 thought 更准确地理解为：

持续展开的内部计算过程

不是心理学意义上的意识。

## 十九、CTM 在图像任务中怎么工作？

以图像分类为例，CTM 可以这样工作：

输入图像
↓
图像被编码成可被 attention 访问的数据表示
↓
CTM 内部 tick 开始运行
↓
同步表示生成 attention query
↓
模型在不同 tick 关注图像不同区域
↓
神经元活动继续演化
↓
同步表示逐渐形成
↓
输出类别预测

论文展示了 CTM 在 ImageNet-1K 分类中的示例，并提到它的 attention process 会呈现出类似“看来看去”的轨迹。

这和普通 CNN/ViT 的区别是：

CNN/ViT：一次前向传播得到图像表示
CTM：多个内部 tick 中动态观察和更新表示
## 二十、CTM 在迷宫任务中怎么工作？

CTM 的一个代表展示是 2D maze solving。

普通模型做迷宫可能需要：

明确的位置编码
搜索算法
路径规划模块

CTM 展示了一种不同方式：

通过内部 tick 逐步观察迷宫
形成类似路径的内部动态
最终给出路线或动作

论文提到 CTM 在 2D 迷宫任务中可以形成内部地图，并且在没有位置编码的情况下学习“look around”和路线相关行为。

这也是 CTM 受到关注的原因之一：它的行为看起来不只是简单分类，而是有一个逐步展开的内部过程。

## 二十一、CTM 的能力展示

公开资料中提到 CTM 被用于多种任务，包括：

ImageNet 分类
2D 迷宫求解
排序
奇偶校验
问答
强化学习任务

GitHub README 中列出了这些任务，并提供了对应代码结构；论文摘要和方法部分也提到 CTM 在 ImageNet-1K、2D maze、parity computation 等任务上展示了能力。

这说明 CTM 不是只为某一个任务设计的，而是想作为一种通用神经网络架构来探索。

## 二十二、CTM 的优势
1. 有内部计算时间

传统模型通常固定层数，CTM 可以让内部状态在多个 tick 中展开。

这带来一个直观优势：

简单问题可以少想
复杂问题可以多想

论文中也提到 CTM 展示了 native adaptive computation time，也就是内部计算时间可以成为模型能力的一部分。

2. 表示更动态

普通网络表示通常是静态向量：

h = [0.1, 0.3, -0.2, ...]

CTM 表示的是：

神经元之间随时间的同步关系

这比单个时间点的 activation 多了时间维度的信息。

3. 可解释性更强

因为 CTM 有内部 tick，有 attention 轨迹，有神经元活动历史，所以可以观察：

每个 tick 模型在看哪里
神经元活动如何变化
哪些神经元同步
模型什么时候形成预测

论文展示了 CTM 的注意过程和神经活动可视化，并把它作为解释内部过程的一个自然途径。

4. 更接近生物启发

CTM 不是简单复制生物大脑，但它显式引入了：

神经元时间动态
神经元级处理
神经同步

论文也说 CTM 试图在神经抽象和生物合理性之间取得平衡。

## 二十三、CTM 的局限

CTM 目前更像一个研究型架构，不是像 ResNet、Transformer、YOLO 那样已经广泛工业落地的标准方案。

主要局限包括：

1. 架构复杂，实现和调参难度更高
2. 多 tick 内部计算可能增加推理成本
3. 神经元级私有模型会带来额外参数和计算设计问题
4. 对大规模任务、工业任务、LLM 规模扩展还需要更多验证
5. 生态不如 Transformer 成熟

也就是说，你现在做 Kaggle 图像分类、目标检测、医学分割，不会直接用 CTM 替代 CNN、ViT、YOLO 或 Mask2Former。

它更适合你作为：

前沿架构理解
研究型面试谈资
神经网络动态计算方向
## 二十四、用一句话概括 CTM

CTM 可以概括成：

CTM 是一种引入内部时间轴的神经网络架构，它让神经元活动在多个 internal ticks 中展开，每个神经元用自己的小模型处理历史输入，并通过神经元活动的同步关系形成 latent representation，用这个同步表示来观察数据、产生输出和进行决策。