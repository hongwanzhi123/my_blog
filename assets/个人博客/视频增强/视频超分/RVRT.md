# RVRT

RVRT 全名是 Recurrent Video Restoration Transformer with Guided Deformable Attention，可以翻译成 带引导可变形注意力的循环视频复原 Transformer。它是 NeurIPS 2022 的视频复原方法，目标是同时兼顾 VRT 这类 Transformer 并行建模能力 和 BasicVSR 这类循环传播模型的参数/显存效率。论文和官方仓库都说明，RVRT 面向视频超分、视频去模糊、视频去噪三类任务。

最核心的一句话是：

RVRT = 全局循环传播 + 局部 clip 内并行 Transformer + 跨 clip 的 Guided Deformable Attention 对齐。

## 1. RVRT 为什么被提出

在 RVRT 之前，视频复原方法大致有两类。

第一类是 并行式方法，比如 EDVR、VRT。它们可以一次处理多帧，帧间信息融合充分，并行性好；但缺点是模型大、显存高，尤其输入长视频时开销很大。论文指出，并行方法可以直接融合多帧信息，但通常模型尺寸大、长序列视频显存消耗高。

第二类是 循环式方法，比如 BasicVSR、BasicVSR++。它们按时间顺序传播 hidden feature，共享参数，所以模型相对小、显存更友好；但缺点是长距离依赖容易丢失，而且逐帧递推导致并行性差。论文也指出，循环模型虽然共享参数、尺寸较小，但顺序处理会带来长距离信息损失、噪声放大，以及难以并行的问题。

RVRT 的目标就是取两者优点：

不要像 VRT 那样一次把整段视频全部并行处理，太重；也不要像 BasicVSR 那样逐帧递推，信息太容易衰减。

所以它把视频切成一个个 clip 小片段，在 clip 内并行处理多帧，在 clip 与 clip 之间循环传播。论文明确说，RVRT 把视频分成固定长度 clips，然后用前一个 clip 的已推理特征来估计后一个 clip 的特征；clip 内的不同帧特征则通过 self-attention 联合提取、隐式对齐和融合。

## 2. RVRT 的整体结构

RVRT 整体由三部分组成：

低质量视频输入
        ↓
Shallow Feature Extraction 浅层特征提取
        ↓
Recurrent Feature Refinement 循环特征精炼
        ↓
HQ Frame Reconstruction 高质量帧重建

论文中明确说，RVRT 包含三部分：浅层特征提取、循环特征精炼和高质量帧重建。浅层阶段先用卷积提取输入视频特征，再用若干 Residual Swin Transformer Blocks，也就是 RSTB，进一步提取浅层特征；最后通过 RSTB 和 PixelShuffle 重建高质量视频。

更直观地说：

输入视频帧序列：
x1, x2, x3, x4, x5, x6, ...

切成 clips：
Clip 1: x1, x2
Clip 2: x3, x4
Clip 3: x5, x6
...

RVRT 做的事：
Clip 内：多帧并行 Transformer 建模
Clip 间：前一个 clip 的特征循环传给后一个 clip

如果 clip 长度是 2，那么它不是逐帧传播：

x1 → x2 → x3 → x4

而是按 clip 传播：

(x1, x2) → (x3, x4) → (x5, x6)

这样做的好处是：时间序列长度被缩短了。原来 100 帧如果逐帧传播是 100 个时间步；如果每个 clip 2 帧，就变成 50 个时间步。传播步数减少，长时序信息损失和噪声放大也会缓解。

## 3. RVRT 的核心思想：全局循环，局部并行

RVRT 最重要的设计就是：

Globally Recurrent
Locally Parallel

也就是：

全局上是循环模型：clip 与 clip 之间按时间递推，前一个 clip 的特征帮助后一个 clip 恢复。
局部上是并行模型：同一个 clip 内的多帧一起进入 Transformer，帧之间可以同时交互。

论文总结得很清楚：RVRT 在全局上以 recurrent 方式传播不同视频 clip 的特征，在局部上联合并行更新一个 clip 内不同帧的特征；对任意单帧来说，它既能利用时间上累计的全局信息，也能利用 clip 内 self-attention 提取的局部多帧信息。

这就是它和 BasicVSR / VRT 的关键区别。

方法	时间建模方式	优点	缺点
BasicVSR	逐帧循环传播	轻量，显存较低	并行性差，长距离信息可能衰减
VRT	多帧并行 Transformer	多帧融合强，并行性好	模型大，显存高
RVRT	clip 内并行，clip 间循环	效果、显存、速度更折中	结构更复杂，仍依赖光流/GDA

## 4. Recurrent Feature Refinement：循环特征精炼

RVRT 的核心模块之一是 RFR：Recurrent Feature Refinement。

假设视频被分成多个 clip：

F1, F2, F3, ..., Ft

在第 t 个 clip 上，RVRT 不只使用当前 clip 的浅层特征，还会使用前一个 clip 已经精炼过的特征。流程可以理解为：

前一个 clip 的特征
        ↓
GDA 对齐到当前 clip
        ↓
和当前 clip 的浅层特征、前面层特征融合
        ↓
MRSTB 精炼
        ↓
得到当前 clip 的精炼特征

论文中的公式表达是：先用 Guided Deformable Attention 把第 t-1 个 clip 的特征对齐到第 t 个 clip，然后 RFR 使用当前 clip 的浅层特征、前面 RFR 层的特征，以及对齐后的前一 clip 特征共同更新当前 clip 特征。

这一点很重要，因为 RVRT 不是简单把 BasicVSR 的 CNN block 换成 Transformer block。它真正改变的是时间单位：从 frame-level recurrent 变成 clip-level recurrent。

## 5. MRSTB：clip 内多帧联合建模

RFR 里面用的是 MRSTB，可以理解成 modified Residual Swin Transformer Block。

普通 SwinIR / RSTB 通常是在图像二维窗口内做注意力：

h × w window

RVRT 把它扩展成视频 clip 内的三维窗口：

N × h × w window

其中 N 是 clip 长度。论文明确说，MRSTB 把原来的 2D h × w attention window 升级成 3D N × h × w attention window，使得 clip 内每一帧可以同时 attend 到自己和其他帧，从而实现隐式特征聚合。

这意味着：

在同一个 clip 内：
第 1 帧可以看第 2 帧
第 2 帧可以看第 1 帧
它们不是逐帧传播，而是并行交互

这和 BasicVSR 的区别非常明显。

BasicVSR 的传播是：

第 t 帧 ← 第 t-1 帧 hidden feature

RVRT 的 clip 内建模是：

Clip 内多帧一起做 attention
不同帧特征同时更新

所以 RVRT 在局部短时间范围内更像 VRT，在全局长时间范围内更像 BasicVSR。

## 6. 为什么 RVRT 要做双向时间传播

RVRT 不只是从前往后传播，还会交替利用正向和反向时间信息。论文提到，为了在时间上向前和向后累计信息，RVRT 会在偶数个 recurrent feature refinement modules 中反转视频序列。

直观理解就是：

RFR 第 1 层：Clip1 → Clip2 → Clip3
RFR 第 2 层：Clip3 → Clip2 → Clip1
RFR 第 3 层：Clip1 → Clip2 → Clip3
...

这样每个 clip 不只获得过去 clip 的信息，也能通过后续层获得未来 clip 的信息。

这和 BasicVSR 的 forward / backward propagation 有点类似，但 RVRT 的传播单位是 clip，而不是单帧。

## 7. GDA：Guided Deformable Attention

RVRT 最重要的创新模块是 GDA：Guided Deformable Attention，引导可变形注意力。

它解决的问题是：

前一个 clip 的特征怎么对齐到当前 clip？

在 BasicVSR 里，对齐通常是 frame-to-frame：

第 t-1 帧 → 第 t 帧

但 RVRT 是 clip-to-clip：

Clip t-1 → Clip t

假设每个 clip 有 2 帧：

Clip t-1: A1, A2
Clip t:   B1, B2

要对齐 Clip t-1 到 Clip t，并不是只做一个帧对齐，而是可能存在多个对应关系：

A1 → B1
A2 → B1
A1 → B2
A2 → B2

如果用普通 frame-to-frame alignment，就要逐对对齐，再额外融合。RVRT 认为这样麻烦且不够高效，所以提出 GDA，直接做一阶段 video clip-to-clip alignment。论文明确说，GDA 是为 clip-to-clip alignment 设计的 one-stage video-to-video alignment 方法。

## 8. GDA 的工作过程

GDA 可以拆成三步理解。

第一步：光流预对齐

GDA 先用光流做粗对齐。论文中说，受光流估计设计启发，GDA 先用 optical flow 对 t-1 clip 的特征进行 pre-align。实际中，第一层光流由输入 LQ 视频通过 SpyNet 估计。

可以理解成：

前一个 clip 特征
        ↓ 光流 warping
粗略对齐到当前 clip

光流给出的是大致运动方向。

第二步：预测 offsets 做可变形采样

光流不一定准，所以 GDA 继续预测 offset。它把当前 clip 特征、光流预对齐后的前一个 clip 特征、光流本身拼接起来，通过一个小 CNN 预测 offset。论文中公式也说明，offset 来自这些特征的 concat，并用若干卷积和 ReLU 层预测。

直观理解：

光流：告诉模型大概在哪里找
offset：让模型在附近进一步微调

最终采样位置是：

光流位置 + offset

论文明确说，GDA 根据光流和 offset 之和指示的预测位置，从前一个 clip 中采样相关特征。

第三步：注意力加权聚合

采样到多个候选位置后，GDA 不像普通 deformable convolution 那样使用固定卷积权重，而是用 attention 动态决定每个候选位置的重要性。

它构造：

Q：来自当前 clip 当前帧位置
K：来自前一个 clip 中采样位置
V：来自前一个 clip 中采样位置

然后计算：

Attention(Q, K) × V

论文中明确写到，GDA 根据 Q 和 K 计算 attention weights，再对 V 做加权求和得到对齐特征。

也就是说：

GDA = 光流引导 + 可变形采样 + 注意力聚合

## 9. GDA 和光流 warping / DCN / VRT mutual attention 的区别

这个点面试里很有价值。

9.1 和普通光流 warping 的区别

普通光流 warping 通常是：

一个像素位置 → 一个对应位置

GDA 是：

一个参考位置 → 多个相关采样位置 → attention 加权聚合

论文也明确说，相比光流 warping 只从一帧采样一个点，GDA 可以从视频 clip 中采样多个相关位置。

所以 GDA 比普通光流更灵活。

9.2 和 Deformable Convolution 的区别

可变形卷积也会采样多个位置，但它的聚合权重来自卷积核，是相对固定的。GDA 的聚合权重则由 Query-Key 相似性动态计算。论文指出，和 deformable convolution 使用固定权重不同，GDA 生成动态权重来聚合不同位置的特征。

所以：

DCN：采样位置可变，聚合权重偏固定
GDA：采样位置可变，聚合权重也动态
9.3 和 VRT 的 Mutual Attention 区别

VRT 的 mutual attention 受局部 attention window 限制，如果运动太大，匹配点可能跑出窗口；如果做全局 attention，计算量又很大。RVRT 的 GDA 通过光流和 offset 主动预测采样位置，可以从任意相关位置取样，同时避免全局 attention 的巨大计算量。论文中也说，GDA 不像 mutual attention 那样受局部 attention 小感受野限制，也避免了全局 attention 的巨大计算负担。

所以 GDA 可以理解成：

用光流先给一个搜索方向，再用 offset 找多个候选位置，最后用 attention 动态选择最有用的信息。

## 10. RVRT 的数据流例子

假设输入 8 帧视频，clip 长度为 2：

x1, x2, x3, x4, x5, x6, x7, x8

先切成：

C1 = (x1, x2)
C2 = (x3, x4)
C3 = (x5, x6)
C4 = (x7, x8)

RVRT 的一层 RFR 大致做：

C1 浅层特征 → MRSTB 精炼 → F1

F1 通过 GDA 对齐到 C2
C2 浅层特征 + 对齐后的 F1 → MRSTB 精炼 → F2

F2 通过 GDA 对齐到 C3
C3 浅层特征 + 对齐后的 F2 → MRSTB 精炼 → F3

F3 通过 GDA 对齐到 C4
C4 浅层特征 + 对齐后的 F3 → MRSTB 精炼 → F4

下一层可能反过来：

C4 → C3 → C2 → C1

多层堆叠以后，每个 clip 都能获得更长时间范围的信息。

## 11. RVRT 和 VRT 的区别

你前面问过 VRT，所以这里重点对比。

VRT 是比较“纯并行”的 Transformer 视频复原框架。它把视频划分成 clips，通过 TMSA 和 shifted clip 实现长时序建模。优点是建模能力强，缺点是模型和显存比较重。VRT 论文也把自己定位为 parallel frame prediction 和 long-range temporal dependency modelling 的视频复原 Transformer。

RVRT 则把 recurrent 引入 Transformer：

VRT：更偏并行 Transformer
RVRT：clip 内并行，clip 间循环
对比	VRT	RVRT
主体思想	并行视频 Transformer	循环视频 Transformer
时间单位	clips / windows	clips
长时序建模	shifted clips + temporal windows	clip 间 recurrent propagation
对齐	TMSA mutual attention + parallel warping	Guided Deformable Attention
优点	并行性强，建模充分	参数、显存、速度更平衡
缺点	更重	仍有循环依赖，结构复杂

论文表格中，在 320×180 LQ 输入上，VRT 参数量 35.6M、测试显存 2149M、运行时间 243ms、PSNR 32.19；RVRT 参数量 10.8M、测试显存 1056M、运行时间 183ms、PSNR 32.75。这个结果说明 RVRT 在该实验设置下比 VRT 更小、更省显存、更快，同时 PSNR 更高。

## 12. RVRT 和 BasicVSR++ 的区别

BasicVSR++ 是循环传播 CNN 系列的代表：

BasicVSR++：
二阶网格传播 + 光流引导可变形对齐

RVRT 是 Transformer + clip-level recurrent：

RVRT：
clip 内 Transformer 并行建模 + clip 间 GDA 循环对齐传播
对比	BasicVSR++	RVRT
主体	CNN/Recurrent VSR	Transformer/Recurrent VRT
时间传播	frame-level 二阶网格传播	clip-level 循环传播
对齐方式	flow-guided deformable alignment	guided deformable attention
局部多帧融合	主要通过传播和卷积	clip 内 3D window self-attention
优点	高效、结构成熟	Transformer 表达更强，性能更好
缺点	表达能力受 CNN 限制	仍更复杂，部署更难

论文中 RVRT 在 REDS4、Vimeo-90K-T、Vid4、UDM10 等视频超分测试上与 BasicVSR++、VRT、EDVR 等方法比较；在表格结果中，RVRT 在 REDS4 上 PSNR/SSIM 为 32.75/0.9113，而 BasicVSR++ 是 32.39/0.9069，VRT 是 32.19/0.9006。论文还总结说，相比 BasicVSR++，RVRT 在 PSNR 上有约 0.2 到 0.5dB 的提升。

## 13. RVRT 和 EDVR 的区别

EDVR 是滑动窗口式方法：

输入中心帧附近若干帧
        ↓
PCD 对齐
        ↓
TSA 融合
        ↓
输出中心帧

RVRT 是 clip-level recurrent 方法：

输入整段视频
        ↓
切成 clips
        ↓
clip 内并行 Transformer
        ↓
clip 间 GDA 循环传播
        ↓
输出整段高质量视频

EDVR 强在窗口内对齐和融合，但长时序能力有限；RVRT 可以通过 clip 传播累计更长时间范围的信息，同时又比全并行 Transformer 更省显存。论文表格中，在同一 320×180 LQ 输入设置下，EDVR 参数量 20.6M、测试显存 3535M、运行时间 378ms、PSNR 31.09；RVRT 参数量 10.8M、测试显存 1056M、运行时间 183ms、PSNR 32.75。

## 14. RVRT 的实验结果说明什么

RVRT 论文在视频超分、视频去模糊、视频去噪三个任务上做了实验。官方仓库也列出了视频超分数据集 REDS、Vimeo90K、Vid4、UDM10，视频去模糊数据集 GoPro、DVD，以及视频去噪数据集 DAVIS、Set8。

在视频超分任务上，论文表格显示 RVRT 在 REDS4、Vid4 等数据集上表现强于 EDVR、BasicVSR、IconVSR、VRT、BasicVSR++ 等方法。论文也写到，RVRT 在 REDS4 和 Vid4 上达到最好表现，并且相比 BasicVSR++ 有 0.2 到 0.5dB PSNR 的提升。

在视频去模糊任务上，RVRT 在 DVD 和 GoPro 数据集上也取得很强结果。比如论文表 7 中 DVD 上 RVRT 为 34.30/0.9655，VRT 为 34.24/0.9651；GoPro 上 RVRT 为 34.92/0.9738，VRT 为 34.81/0.9724。

在视频去噪任务上，论文显示 RVRT 在 DAVIS 和 Set8 上与 VRT 竞争，其中高噪声水平下表现略优，同时模型更小、速度更快。论文写到，在 1280×720 LQ 输入上，RVRT 的 denoising 模型为 12.8M 参数、0.2s runtime，而 VRT 是 18.4M 参数、1.5s runtime。

## 15. RVRT 的优点
15.1 效果、速度、显存比较平衡

RVRT 最大的卖点不是“绝对最轻”，也不是“纯 Transformer 最大模型”，而是 balanced model size, testing memory and runtime。官方仓库和论文摘要都强调了这个特点。

15.2 比普通 recurrent 模型更会利用局部多帧信息

BasicVSR 是逐帧递推；RVRT 一个 clip 内多帧一起进入 3D window attention，所以局部多帧关系更充分。MRSTB 的 3D N × h × w attention window 让 clip 内每帧能同时 attend 到自己和其他帧，实现隐式特征聚合。

15.3 比纯并行 Transformer 更省资源

VRT 这类并行 Transformer 的模型和显存都较大；RVRT 通过 clip-level recurrent 传播减少了视频序列长度，同时共享一部分时序处理结构。论文实验中 RVRT 比 VRT 参数更少、显存更低、速度更快。

15.4 GDA 对跨 clip 对齐更灵活

GDA 不只是光流 warping，也不只是 DCN，而是光流引导、多位置采样和 attention 动态聚合的结合。论文总结说，GDA 在光流引导下从多个邻近位置自适应聚合信息。

## 16. RVRT 的局限

第一，RVRT 仍然依赖光流预对齐。论文的 limitation 部分明确指出，光流预对齐的复杂度会随 clip 长度呈二次增长；作者也提到一种可能方向是开发 video-to-video optical flow estimation，直接预测所有光流。

第二，RVRT 工程部署比 Real-ESRGAN、BasicVSR 复杂。它包含 Transformer、clip-level recurrent、光流估计、GDA 采样和注意力聚合，ONNX / TensorRT / 移动端部署都不算轻松。官方仓库测试命令里也提供了 --tile 和 --tile_overlap，并提示如果 out-of-memory 可以减小 tile，但性能可能略降。

第三，它主要是标准视频复原框架，不等于真实世界视频增强的完整方案。如果输入是强压缩、强噪声、真实退化很复杂的视频，仍然需要真实退化建模、清理模块或者真实数据微调；这一点和 RealBasicVSR 的关注点不同。

## 17. 工程上怎么理解 RVRT

如果你要在项目里用 RVRT，需要关注这些点。

输入格式一般是视频序列：

B, T, C, H, W

其中 B 是 batch，T 是帧数，C 是通道数，H/W 是低质量帧尺寸。视频超分输出一般是：

B, T, C, sH, sW

其中 s 是上采样倍率。论文中也定义了低质量视频为 T × H × W × C，目标是重建 T × sH × sW × C 的高质量视频。

长视频推理通常需要分段和 tile。因为 RVRT 虽然比 VRT 省显存，但仍然是视频 Transformer，处理长视频和高分辨率视频时显存压力不小。官方 quick testing 中明确提供了 --tile 和 --tile_overlap 参数，并提示 OOM 时可以减小 tile。

部署难点主要有三个：SpyNet 光流、GDA 中的采样/attention、Transformer 块。相比 Real-ESRGAN 这种单帧 CNN，RVRT 更适合服务器离线处理或高质量视频修复，不太适合作为移动端实时模型。

## 18. 面试里怎么讲 RVRT


RVRT 是一个循环视频复原 Transformer，目标是在并行 Transformer 方法和循环传播方法之间取得折中。传统并行方法比如 VRT 可以同时融合多帧信息，但模型和显存开销较大；循环方法比如 BasicVSR 参数较少，但逐帧传播会导致长距离信息损失和并行性差。RVRT 把视频划分为多个 clip，在每个 clip 内使用 3D window 的 modified RSTB，让多帧特征并行地通过 self-attention 交互；在 clip 与 clip 之间，则采用 recurrent feature refinement，用前一个 clip 的特征去精炼后一个 clip 的特征。为了完成 clip-to-clip 对齐，RVRT 提出了 Guided Deformable Attention。GDA 先用光流进行粗对齐，再预测多个 offset 采样相关位置，最后通过 Query-Key attention 动态聚合 Value 特征。这样它比普通光流 warping 更灵活，比 DCN 的固定权重聚合更自适应，也比全局 attention 更省计算。整体上，RVRT 的优势是效果、显存和速度比较平衡，适合高质量视频超分、去模糊和去噪；缺点是结构复杂，仍依赖光流和 Transformer，工程部署难度高于 BasicVSR 或 Real-ESRGAN。


RVRT 的本质是：把视频按 clip 切分，clip 内用 Transformer 并行融合，clip 间用 recurrent 传播长期信息，再用 GDA 完成跨 clip 的动态对齐。