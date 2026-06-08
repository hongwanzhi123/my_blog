# VRT

VRT 全名是 Video Restoration Transformer，论文题目是 “VRT: A Video Restoration Transformer”。它可以理解为：把 Transformer 引入视频复原任务，用注意力机制同时完成多帧特征提取、帧间对齐和时序融合。

它和你前面问的 EDVR、BasicVSR、BasicVSR++ 最大区别是：

EDVR 依赖滑动窗口 + PCD 可变形卷积对齐；BasicVSR 系列依赖循环传播 + 光流/可变形对齐；VRT 则希望用 Transformer 的注意力机制，在并行框架下建模更长时间范围的多帧关系。

## 1. VRT 解决的核心问题

视频复原任务包括视频超分、视频去模糊、视频去噪、视频插帧等。它们共同的问题是：输入不是单张图像，而是一段连续视频帧。模型需要利用相邻帧的信息来恢复当前帧，但相邻帧之间往往存在运动、遮挡和错位。VRT 论文指出，视频复原相比图像复原的关键难点，就是如何充分利用多个相邻但通常没有对齐的帧。

在 VRT 之前，常见视频复原框架主要有两类：

第一类是 滑动窗口方法，比如 EDVR。它通常输入当前帧附近几帧，只恢复中心帧。这类方法窗口内对齐融合能力强，但长时序建模能力有限，而且推理时每一帧会被重复处理，计算利用率不高。VRT 论文也指出，滑动窗口方法通常 frame-by-frame 地处理长视频，特征利用效率较低。

第二类是 循环传播方法，比如 BasicVSR / BasicVSR++。它可以沿时间方向传播特征，利用较长的视频序列。但循环结构不容易并行，而且远距离信息可能经过多步传播后衰减。VRT 论文中也提到 recurrent 方法存在并行性差、长距离依赖建模能力不足，以及少帧视频上性能下降等问题。

所以 VRT 想解决的是：

既要像滑动窗口方法一样并行处理多帧，又要像循环方法一样建模长时序依赖，同时还要解决帧间运动对齐问题。

## 2. VRT 的核心思想

VRT 的核心可以概括成一句话：

把视频序列切成小片段，用 Temporal Mutual Self Attention 在片段内进行帧间对齐和融合，再通过 shifted clip 机制让不同片段交互，并结合 parallel warping 处理大运动。

论文中明确说，VRT 由多个尺度组成，每个尺度包含两类模块：Temporal Mutual Self Attention，TMSA 和 Parallel Warping。TMSA 把视频划分为小 clips，在 clip 内用 mutual attention 联合完成运动估计、特征对齐和特征融合，同时用 self-attention 做特征提取；为了实现跨 clip 交互，VRT 每隔一层会对视频序列做 shift。

整体流程可以理解成：

输入低质量视频帧序列
        ↓
浅层卷积提取特征
        ↓
多尺度 Transformer 特征提取
        ↓
每个尺度内：
    TMSA：局部时空注意力，对齐 + 融合
    Parallel Warping：进一步融合邻近帧信息，处理大运动
        ↓
多尺度特征上采样与跳连融合
        ↓
重建模块
        ↓
输出高质量视频帧序列

注意，VRT 不是像 EDVR 那样一次只输出中心帧，而是可以 并行预测多个输出帧。论文强调 VRT 具有 parallel frame prediction 和 long-range temporal dependency modelling 能力。

## 3. VRT 的整体架构

VRT 可以分成两个大部分：

第一部分：Feature Extraction 特征提取

输入视频序列后，VRT 先用一个 2D 卷积提取浅层特征。然后进入多尺度结构，在不同分辨率下处理运动和时序关系。论文中说，VRT 会在多尺度框架下对不同图像分辨率的帧进行对齐；在每个尺度上，通过 TMSA 和 parallel warping 提取特征、处理运动、融合多帧信息，并在相同尺度之间加入 skip connection。

第二部分：Reconstruction 重建

经过多尺度特征提取和融合后，VRT 根据每一帧对应的特征独立重建高质量帧。对于视频超分任务，它使用 sub-pixel convolution，也就是 PixelShuffle 类似的上采样模块；对于视频去模糊这类输入输出分辨率相同的任务，一个卷积层就可以完成重建。论文还使用 global residual learning，只预测双线性上采样低质量序列和 ground truth 之间的残差。

所以 VRT 的架构不是单一尺度的 Transformer，而是：

浅层特征
    ↓
多尺度编码 / 对齐 / 融合
    ↓
多尺度解码 / 跳连
    ↓
进一步 TMSA refinement
    ↓
任务相关重建头

## 4. 核心模块一：TMSA

TMSA 全称是：

Temporal Mutual Self Attention

也就是 时间互注意力 + 自注意力模块。

它里面有两个关键概念：

Mutual Attention：互注意力
Self Attention：自注意力

二者分工不同：

Mutual Attention：负责帧间对齐与帧间融合
Self Attention：负责当前特征内部建模与特征提取

论文中明确说，在 TMSA 中，mutual attention 主要用于相邻 two-frame clips 之间的 mutual alignment，而 self-attention 用于 feature extraction。

## 5. Mutual Attention：把注意力看成“软对齐”

这是 VRT 最核心、最值得理解的地方。

传统视频复原需要显式对齐，比如：

EDVR：用可变形卷积对齐
BasicVSR：用光流 warp 对齐
BasicVSR++：用光流引导可变形卷积对齐

VRT 的想法是：

注意力本身就可以看成一种隐式运动估计和软对齐。

假设有两帧：

Reference frame：当前要恢复的参考帧
Supporting frame：提供信息的辅助帧

VRT 用 reference frame 的特征生成 Query，用 supporting frame 的特征生成 Key 和 Value。然后 Query 和 Key 做相似度匹配，得到 attention map，再用这个 attention map 对 supporting frame 的 Value 加权求和。论文中解释说，这个 attention map 反映了参考帧元素和辅助帧元素之间的相关性。

直观理解：

当前帧某个位置想恢复一个车牌字符
        ↓
Query 去辅助帧中找相似特征
        ↓
找到相似位置后，把辅助帧的信息加权拿过来
        ↓
完成对齐与融合

如果某个辅助帧位置和当前帧位置最相似，那么 attention 权重会集中到那里。这种情况下，它就很像光流 warping：把辅助帧对应位置的信息搬到当前帧位置。论文中也明确说，在一种极端情况下，mutual attention 等价于给定光流向量的图像 warping；一般情况下，它可以看作 image warping 的“soft version”。

所以 VRT 的 Mutual Attention 可以这样理解：

光流 warping 是硬匹配：一个位置主要找一个对应点。
Mutual Attention 是软匹配：一个位置可以从多个相关位置加权取信息。

## 6. Mutual Attention 相比光流对齐的优势

VRT 论文认为 mutual attention 相比显式光流估计 + warping 有几个优势。

第一，它不是只关注一个目标像素，而是可以从 supporting frame 中自适应保留多个相关位置的信息，因此比普通 warping 更灵活。论文还提到，它可以避免没有匹配位置时出现的 black hole artifacts。

第二，它没有传统 CNN 光流估计中的局部性归纳偏置。换句话说，光流估计通常更依赖局部邻域，如果相邻物体运动方向不同，局部估计可能困难；而 attention 可以在窗口内直接找相似特征。论文中也把这一点列为 mutual attention 的优势之一。

第三，它把运动估计和特征 warping 合并到了一个 attention 过程中。传统方法往往先在 RGB 图像上估计光流，再用光流去 warp 特征；VRT 的 mutual attention 则直接在特征层完成隐式匹配和融合。

这点在面试里很有价值，你可以这样说：

VRT 不是不要对齐，而是把显式光流对齐变成了基于注意力的隐式软对齐。注意力权重本身就表示跨帧特征的相关性，因此可以同时完成匹配、对齐和融合。

## 7. 为什么 VRT 还需要局部窗口

Transformer 的注意力复杂度和 token 数量是平方关系。视频帧本身分辨率高，如果直接对整段视频、整张图像做全局注意力，计算量和显存都无法接受。

所以 VRT 和 Swin Transformer 类似，会把每一帧在空间上划分成不重叠的局部窗口，只在窗口内部做注意力。论文明确说，由于 attention 的复杂度与窗口内元素数量呈平方关系，在完整图像上做 global attention 通常不可行，因此 VRT 将每帧划分为空间局部窗口，并使用 shifted window 机制建立跨窗口连接。

这意味着 VRT 的注意力是局部时空注意力，而不是全局时空注意力。

它大概是：

视频序列
    ↓
按时间切成小 clip
    ↓
每帧按空间切成小 window
    ↓
在 clip-window 内做 attention

这样既能利用 Transformer 的建模能力，又能控制显存和计算量。

## 8. Temporal Shift：为什么要移动 clip

如果每一层都把视频切成固定的 2-frame clips，比如：

第 1 层：
(1,2), (3,4), (5,6), (7,8)

那么第 2 帧和第 3 帧永远不在同一个 clip 里，信息交互就会受限。

所以 VRT 借鉴 shifted window 的思想，在时间维度上也做 shift。下一层可能变成：

第 2 层：
(2,3), (4,5), (6,7)

这样原本不同 clip 的帧就可以发生交互。论文中说，TMSA 会把序列划分成 2-frame clips，并且每隔一层 shift 一次，以实现 cross-clip interactions。

这个机制非常重要，因为它让 VRT 虽然局部计算，但经过多层堆叠后可以建立更长时间范围的信息传递。

你可以这样理解：

不 shift：
帧 1 只和帧 2 交互
帧 3 只和帧 4 交互

shift 后：
帧 2 又能和帧 3 交互
帧 4 又能和帧 5 交互

多层堆叠后：
信息可以逐渐传到更远的帧

## 9. 为什么 VRT 还要使用更大的 temporal window

仅靠 2-frame clip 和 shift，远距离帧之间的信息需要经过很多层才能传过去。对于视频复原来说，有些远处帧可能包含非常关键的细节。

因此 VRT 在后面部分的 TMSA 中使用更大的 temporal window size，让远距离帧可以直接交互。论文中提到，只使用很小的 temporal window 不能充分发挥模型潜力，所以在最后四分之一的 TMSA 模块中使用更大的 temporal window size，使远距离帧可以直接交互。

直观理解：

前面层：小 temporal window，局部时序对齐，计算量低
后面层：大 temporal window，增强远距离时序依赖

这就是 VRT 的一个折中设计：既控制计算量，又增强长时序建模。

## 10. 核心模块二：Parallel Warping

很多人会疑惑：既然 mutual attention 已经能做软对齐，为什么还需要 parallel warping？

原因是：VRT 的 attention 是在局部窗口内做的。如果物体运动很大，目标位置可能跑出了当前 attention window，那么 mutual attention 就不一定能找到正确对应位置。

论文中也明确指出，由于空间窗口划分，mutual attention 可能不能很好处理大运动，所以 VRT 在每个网络 stage 的末尾加入 feature warping 来处理大运动。

Parallel Warping 做的事情大概是：

对于当前帧特征
    ↓
计算邻近帧到当前帧的光流
    ↓
把前一帧和后一帧特征 warp 到当前帧
    ↓
和当前帧特征 concat
    ↓
用 MLP 融合并降维

论文中还提到，它参考 BasicVSR++ 的思路，预测 residual flow，并使用 deformable convolution 做 deformable alignment。

所以 VRT 不是完全抛弃光流/warping，而是采用组合策略：

TMSA / Mutual Attention：
处理窗口内的软对齐和细粒度融合

Parallel Warping：
补充处理大运动和跨窗口位移

这点非常关键。你可以把它理解成：

Mutual Attention 负责灵活的局部跨帧融合；Parallel Warping 负责更显式地处理大运动。

## 11. 多尺度结构为什么重要

视频里运动大小是不固定的。小运动可以在高分辨率特征上处理；大运动如果直接在原尺度上对齐，会很困难。

VRT 使用多尺度结构，在不同分辨率上提取、对齐和融合特征。低分辨率尺度更适合处理大运动，因为原图中移动很多像素的物体，在低分辨率特征上位移会变小；高分辨率尺度更适合恢复边缘和纹理细节。论文中说，VRT 的多尺度网络可以在不同图像分辨率下对齐帧，并通过 TMSA 和 parallel warping 在不同尺度上处理物体或相机运动。

这和 EDVR 的 PCD 金字塔思想有点类似：

低尺度：处理大运动、全局结构
高尺度：处理小运动、局部细节

但 VRT 的差别在于，它在每个尺度上主要用 Transformer attention + warping 来融合信息，而不是单纯 CNN + DCN。

## 12. VRT 可以做哪些任务

VRT 不是只为视频超分设计的，它是一个比较通用的视频复原框架。论文实验覆盖了：

视频超分；

视频去模糊；

视频去噪；

视频插帧；

时空视频超分。

论文摘要中说，VRT 在五类任务、十四个 benchmark 数据集上进行实验，并取得最高 2.16 dB 的提升。

官方 GitHub 页面也说明，VRT 是官方 PyTorch 实现，并提供预训练模型、测试脚本和 visual results。该仓库中列出的 quick testing 包括视频超分等任务，并提示如果显存不足可以减小 tile。

这说明 VRT 的定位不是“只做 SR 的单任务模型”，而是：

面向多种视频复原任务的统一 Transformer 框架。

## 13. VRT 和 EDVR 的区别

EDVR 是典型的滑动窗口视频复原模型：

当前帧附近若干帧
    ↓
PCD Alignment 对齐
    ↓
TSA Fusion 融合
    ↓
输出中心帧

VRT 则是并行预测多帧：

输入一段视频序列
    ↓
多尺度 TMSA + Parallel Warping
    ↓
并行输出多个高质量帧

二者核心差异是：

对比	EDVR	VRT
框架	滑动窗口	并行多帧预测
对齐	PCD 可变形卷积	Mutual Attention + Parallel Warping
融合	TSA 时空注意力	TMSA 时序互注意力
长时序	较弱，窗口限制	更强，shifted clip + 大 temporal window
计算	CNN/DCN 为主	Transformer 为主，显存更高

VRT 论文实验中也与 EDVR 对比，指出 VRT 在不同数据集上相对 EDVR 有 0.50 到 1.57 dB 的提升，并且 VRT 是同时输出所有帧，而不是像 EDVR 那样逐帧预测。

## 14. VRT 和 BasicVSR / BasicVSR++ 的区别

BasicVSR 系列的核心是 循环传播：

BasicVSR：
双向传播 + 光流 warping

BasicVSR++：
二阶网格传播 + 光流引导可变形对齐

VRT 的核心是 并行 Transformer 建模：

VRT：
多尺度 TMSA + shifted clip + parallel warping

可以这样理解：

对比	BasicVSR / BasicVSR++	VRT
主体结构	Recurrent propagation	Parallel Transformer
长时序信息	通过 hidden state 传播	通过 attention 和 shifted clip 交互
对齐方式	光流 / flow-guided DCN	mutual attention + warping
并行性	较差，时间递归	更好，可以并行输出多帧
显存	相对低	通常较高
部署难度	BasicVSR 较容易，BasicVSR++ 中等	Transformer 显存压力大

VRT 论文中的表格显示，在 Vimeo-90K 等数据集上，VRT 相比 BasicVSR++ 也有提升；论文还提到，当 VRT 使用更长序列训练时，PSNR 进一步提升，体现出其时序建模潜力。

不过从工程角度，VRT 并不一定比 BasicVSR++ 更容易落地。它的参数量、显存和推理压力通常更大。官方仓库也提示测试时如果 out-of-memory，可以减小 --tile，但性能可能略有下降。

## 15. VRT 和 RVRT 的关系

RVRT 可以看成 VRT 之后更偏工程折中的版本。VRT 强在并行和长时序建模，但模型规模、测试显存和运行时间都比较重。VRT 官方仓库的新闻中也提到 RVRT 是一个在模型大小、测试显存和 runtime 上更 balanced 的 Recurrent Video Restoration Transformer。

简单理解：

VRT：
更强的并行 Transformer 视频复原框架，但比较重

RVRT：
把 recurrent 思路和 Transformer 结合，追求效果、显存、速度的平衡

所以如果你做研究汇报，VRT 是很好的 Transformer 视频复原代表；如果你做工程项目，RVRT 往往更值得关注。

## 16. VRT 的优点

第一，长时序建模能力强。
VRT 通过 shifted clip、larger temporal window 和多层 TMSA，让远距离帧之间可以逐步或直接发生信息交互，比普通滑动窗口模型更适合建模长时序依赖。论文明确把 VRT 的特点概括为 parallel computation 和 long-range dependency modelling。

第二，对齐和融合统一。
传统方法通常先估计光流或 offset，再做对齐和融合；VRT 的 mutual attention 可以把帧间匹配、软对齐和特征融合放在一个 attention 过程中完成。论文也将 mutual attention 描述为隐式运动估计后的 soft warping。

第三，可以并行输出多帧。
相比 EDVR 这种滑动窗口逐帧输出，VRT 可以输入一段序列后并行重建高质量帧，减少重复处理，并更适合 Transformer 的并行计算模式。

第四，任务通用性强。
VRT 不只做视频超分，还能用于视频去噪、去模糊、插帧和时空超分。论文在多项任务上验证了它的泛化能力。

## 17. VRT 的缺点

第一，显存压力大。
Transformer attention 本身显存开销较大，视频任务又多了时间维度。虽然 VRT 使用空间窗口、时间 clip、tile 测试来降低开销，但相比 BasicVSR 这类轻量循环模型，部署压力仍然明显。官方仓库也提示显存不足时需要减小 tile。

第二，工程部署复杂。
VRT 里面既有 Transformer，又有 parallel warping、deformable alignment 等操作。如果要转 ONNX / TensorRT / 移动端，复杂度比单帧 CNN 或 BasicVSR 更高。

第三，真实世界视频不一定直接适配。
VRT 主要是在标准视频复原数据集上验证。真实视频包含复杂压缩、噪声、过锐化和混合退化时，往往还需要真实退化建模或类似 RealBasicVSR 的清理策略。

第四，速度不一定适合实时。
VRT 更适合高质量离线视频复原，不是移动端实时视频超分的首选。对于实时或端侧场景，BasicVSR-lite、Real-ESRGAN-ncnn、RFDN/IMDN 这类更轻量模型通常更现实。

## 18. 工程实现时要注意什么

如果你要跑 VRT，输入一般是：

B, T, C, H, W

其中 B 是 batch size，T 是帧数，C 是通道数，H/W 是低质量帧尺寸。输出也是序列形式：

B, T, C, scale*H, scale*W

视频超分任务中，VRT 使用 sub-pixel convolution 进行上采样；视频去模糊、去噪这类分辨率不变的任务，则使用对应的卷积重建头。

实际推理大视频时，通常要注意：

1. 控制输入帧数 T
2. 使用 tile 推理防止爆显存
3. 设置 tile overlap 避免块边界伪影
4. 长视频分段处理
5. 保证相邻段有帧重叠，减少时间边界不连续

官方仓库 quick testing 中也使用了 --tile 和 --tile_overlap 参数，并提示减小 tile 可以缓解显存不足

## 19. 面试里怎么讲 VRT


VRT 是一个基于 Transformer 的视频复原框架，主要解决滑动窗口方法长时序建模不足、循环传播方法并行性差和远距离信息衰减的问题。它不是逐帧恢复中心帧，而是输入一段视频后并行重建多个输出帧。结构上，VRT 使用多尺度框架，在不同分辨率下进行特征提取、帧间对齐和时序融合。每个尺度主要由 TMSA 和 Parallel Warping 两类模块组成。TMSA 中的 mutual attention 用当前参考帧特征作为 Query，用辅助帧特征作为 Key 和 Value，通过注意力权重建立跨帧对应关系，因此可以看作一种隐式运动估计和软 warping；self-attention 则用于特征提取。为了控制计算量，VRT 在空间上使用局部窗口，在时间上把视频分成小 clips，并通过 shifted clip 机制实现跨 clip 信息交互。由于局部窗口 attention 对大运动处理有限，VRT 又在每个 stage 末尾加入 parallel warping，通过光流和可变形对齐进一步融合邻近帧信息。整体上，VRT 的优势是并行性好、长时序建模能力强、对齐和融合统一，缺点是显存和计算量较大，工程部署比 BasicVSR 系列更复杂。

最核心的一句话：

VRT 的本质是：用 Transformer 的 mutual attention 把视频帧间“匹配、对齐、融合”统一起来，再通过 shifted temporal clips、多尺度结构和 parallel warping 建模长时序与大运动。