# RealBasicVSR

RealBasicVSR 可以理解为：把 BasicVSR 从“合成退化视频超分”推进到“真实世界低清视频增强”的方法。

BasicVSR / BasicVSR++ 主要面对比较干净、规则的退化，比如 bicubic downsampling、blur-downsampling；而真实视频里常见的是压缩块、噪声、过锐化、运动模糊、低码率、重复编码、颜色偏移等混合退化。RealBasicVSR 的核心问题是：长时序传播本来能帮助恢复细节，但在真实退化场景下，也会把噪声和伪影一起传播、放大。 这篇论文的完整题目是 Investigating Tradeoffs in Real-World Video Super-Resolution.

## 1. RealBasicVSR 解决的核心矛盾

在普通视频超分里，长时序传播通常是好事。比如 BasicVSR 从前往后、从后往前传播特征，可以利用远处帧的信息恢复当前帧细节。

但是 RealBasicVSR 发现：在真实世界视频里，长时序传播是一把双刃剑。

如果视频退化比较轻，长时序传播可以聚合多帧信息，让输出更清楚；但如果输入视频本身有严重噪声、压缩块、伪影，传播过程会把这些坏信息也当作“细节”传下去，最终导致噪声增强、伪影扩散、纹理异常。论文明确指出，在 real-world VSR 中，长序列传播既能增强细节，也可能引入令人不舒服的伪影，因此存在“细节增强”和“伪影抑制”之间的权衡。

所以 RealBasicVSR 的核心思想是：

在做时序传播之前，先把每一帧清理干净，减少噪声和压缩伪影，然后再交给 BasicVSR 做长时序超分。

也就是：

真实低清视频
    ↓
图像清理模块：先去噪、去压缩伪影、弱化退化
    ↓
BasicVSR：再做双向时序传播和超分
    ↓
高清增强视频

## 2. RealBasicVSR 和 BasicVSR 的关系

RealBasicVSR 不是完全重新发明一个视频超分框架，而是在 BasicVSR 前面加了一个关键的 image cleaning module 图像清理模块。

BasicVSR 原本是：

LR 视频帧序列
    ↓
光流估计
    ↓
Backward propagation
    ↓
Forward propagation
    ↓
上采样重建
    ↓
SR 视频

RealBasicVSR 变成：

真实低清视频帧序列
    ↓
Image Cleaning Module
    ↓
清理后的低清视频帧序列
    ↓
BasicVSR
    ↓
超分结果

论文中的公式也很简单。对于输入第 i 帧 x_i，先经过清理模块 C 得到清理后的帧：

x̃_i = C(x_i)

然后把清理后的序列 {x̃_i} 送入 VSR 网络 S，得到输出 {y_i}。论文明确说明，cleaning module 放在 BasicVSR 前面，输入帧先独立经过 cleaning module，清理后的序列再送入 BasicVSR，并且整个网络端到端训练。

所以 RealBasicVSR 的一句话定义是：

RealBasicVSR = Image Cleaning Module + BasicVSR。

## 3. 为什么真实视频不能直接用 BasicVSR

BasicVSR 依赖时序传播。传播的前提是：传过去的信息大多数是有用的。

在合成退化数据里，比如 bicubic 降采样，输入虽然低清，但噪声和伪影比较少。模型从其他帧传播来的信息通常是纹理、边缘、结构，所以长时序传播越充分，效果越好。

但真实视频不一样。真实视频里有：

压缩块；

噪声；

过度锐化边缘；

运动模糊；

低码率导致的蚊噪；

反复转码导致的伪影；

帧间压缩带来的时空不一致退化。

如果 BasicVSR 直接传播这些特征，模型可能会把压缩块、噪声、错误纹理也增强出来。论文中专门对比了 non-blind VSR 和 real-world VSR：在非盲退化中，序列越长能聚合更多有用信息；但在真实退化中，传播更长序列会放大噪声和伪影。

这就是 RealBasicVSR 最重要的观察：

真实视频超分的难点不是“不知道怎么利用多帧”，而是“不知道哪些多帧信息是细节，哪些是伪影”。

## 4. 核心模块一：Image Cleaning Module

Image Cleaning Module 是 RealBasicVSR 最核心的设计。

它的作用不是直接输出高清图，而是在低分辨率空间里做预清理：

输入低清帧
    ↓
去除噪声 / 压缩伪影 / 异常退化
    ↓
输出更干净的低清帧

然后 BasicVSR 再基于这些“更干净的低清帧”做时序传播和超分。

论文里说，清理模块用于在 temporal propagation 之前抑制退化，使输入中的退化对后续 VSR 网络影响更弱。这个设计概念上很简单，但论文强调清理模块的设计需要小心，因为如果清理过度，会损失真实纹理；如果清理不足，伪影仍然会被传播放大。

你可以把它理解成：

不加 cleaning：
噪声 + 细节 → BasicVSR 传播 → 噪声和细节都被增强

加 cleaning：
先削弱噪声和伪影 → BasicVSR 传播 → 更倾向增强真实结构和细节

## 5. Cleaning Loss：为什么要约束清理结果

一个关键问题是：清理模块到底应该输出什么？

它不是输出 HR 图像，而是输出一个“干净的 LR 图像”。论文用低分辨率 ground truth 来监督 cleaning module。具体做法是把 HR ground truth z_i 下采样成低分辨率图像 d(z_i)，然后让清理模块输出 x̃_i 接近这个干净低分辨率图像。论文使用 Charbonnier loss 作为 cleaning loss，同时还使用输出 fidelity loss 约束最终超分结果。

也就是说，训练目标包含两部分：

第一，清理模块输出要像“干净 LR”：

L_clean = clean frame 和 downsampled GT 之间的误差

第二，最终 SR 输出要像 HR ground truth：

L_out = SR output 和 HR GT 之间的误差

这个设计非常关键。因为如果没有 cleaning loss，清理模块可能不会真正学会“清理输入”，而只是把压力交给后面的 BasicVSR。论文的分析图也指出，cleaning loss 对去除伪影很重要。

## 6. 为什么 cleaning module 不接收 GAN / perceptual loss 梯度

RealBasicVSR 后期会用 perceptual loss 和 adversarial loss 做感知质量增强，也就是让输出看起来更真实、更锐利。

但是论文中有一个很细的设计：在使用 perceptual loss 和 adversarial loss 微调时，cleaning module 不接收来自这两个 loss 的梯度。

这个点很重要。

因为 cleaning module 的职责是“清理”，不是“生成纹理”。如果让 GAN loss 直接影响 cleaning module，清理模块可能会倾向于制造高频纹理，反而把输入变得更“花”、更不稳定。

所以职责划分是：

Cleaning module：
负责低分辨率清理，偏保守，去噪去伪影

BasicVSR + 后续重建：
负责利用多帧信息生成高清细节

GAN / perceptual：
主要作用在最终输出，不直接驱动 cleaning module

这也是 RealBasicVSR 比简单“前面加个去噪网络”更细致的地方。

## 7. Dynamic Refinement 动态精炼

真实视频退化程度不一样。有些视频只是轻微压缩，有些视频噪声很重，有些视频有严重块效应。

如果所有视频都只过一遍 cleaning module，可能不够灵活：

轻度退化：一遍就够，过多清理会损失细节；

重度退化：一遍不够，还残留伪影；

极重退化：需要多次清理，但多次清理可能变平滑。

所以 RealBasicVSR 提出了 Dynamic Refinement 动态精炼。

它的思想是：在测试阶段，可以重复应用 cleaning module。如果连续两次清理结果变化还比较大，说明仍有明显退化，就继续清理；如果变化已经小于阈值，就停止。

论文给出的判断逻辑是：如果当前清理结果和上一次清理结果的平均差异大于阈值 θ，就继续执行下一次 cleaning；否则停止。论文还给出经验设置：非 GAN 模型 θ=1.5，GAN 模型 θ=5 是合理设置。

直观理解：

输入帧
  ↓ cleaning 第 1 次
如果变化还很大 → 继续 cleaning 第 2 次
如果变化很小 → 停止
  ↓
送入 BasicVSR

这个机制带来的好处是：

用户可以在“更干净”和“更锐利”之间调节。

多清理几次，输出更干净、更平滑；少清理几次，保留更多细节，但伪影风险更大。

## . RealBasicVSR 的完整推理流程

假设输入一段真实低清视频：

x1, x2, x3, ..., xT
第一步：逐帧清理

每一帧单独经过 cleaning module：

x̃1 = C(x1)
x̃2 = C(x2)
x̃3 = C(x3)
...
x̃T = C(xT)

注意，这里的 cleaning 是逐帧的，不需要时序传播。它先把每一帧里的明显退化削弱。

第二步：动态精炼

对于退化严重的帧，可以多次清理：

x̃_i^1 = C(x_i)
x̃_i^2 = C(x̃_i^1)
x̃_i^3 = C(x̃_i^2)
...

直到相邻两次变化小于阈值。

第三步：送入 BasicVSR

清理后的序列进入 BasicVSR：

{x̃1, x̃2, ..., x̃T}
    ↓
Backward propagation
    ↓
Forward propagation
    ↓
Upsampling

BasicVSR 利用长时序信息恢复细节。

第四步：输出高清视频

得到：

y1, y2, ..., yT

最终输出既利用了多帧信息，又减少了真实退化被放大的风险。

## 9. RealBasicVSR 的训练退化设计

RealBasicVSR 不只是改了网络结构，还研究了真实世界 VSR 的训练问题。

真实视频退化非常复杂，所以训练时需要模拟多种退化。论文采用类似 Real-ESRGAN 的二阶退化模型，包括随机 blur、resize、noise、JPEG compression 等图像级退化；此外，论文还加入了 video compression，因为视频压缩会引入时空变化的退化，更贴近真实视频。

训练退化可以理解成：

HR 视频
    ↓
随机模糊
    ↓
随机缩放
    ↓
随机噪声
    ↓
JPEG 压缩
    ↓
视频压缩
    ↓
模拟真实低清视频

为什么要加入 video compression？

因为真实视频很多都是 MP4、H.264、H.265 编码过的。视频压缩不是简单逐帧 JPEG，它会利用帧间预测，所以压缩伪影也具有时间相关性。论文指出，video compression 会隐式考虑帧间依赖，产生时空变化退化，并且加入视频压缩后观察到性能提升。

## 10. Stochastic Degradation 随机退化方案

真实世界 VSR 训练很慢。因为每次训练要加载一段视频序列，如果 batch size 是 B，序列长度是 L，那么每次迭代 CPU 要加载 B×L 张图像。论文指出，真实世界 VSR 通常需要更大 batch size 来稳定梯度，但这会带来严重 I/O bottleneck。

RealBasicVSR 提出了一个 stochastic degradation scheme 来减少训练时间。

它的想法是：不要每次真的加载完整 L 帧，而是加载 L/2 帧，然后通过时间翻转构造序列，从而减少 CPU 加载压力。为了避免简单翻转导致数据变化不足，论文把每帧退化参数建模成 random walk，也就是相邻帧的退化参数逐步变化。论文报告这个方案可以减少最高约 40% 的训练时间，同时保持性能。

这个点工程上非常有启发：

视频模型训练慢，不只是 GPU 算得慢，很多时候是数据读取和退化生成太慢。

RealBasicVSR 的 stochastic degradation 本质上是在不明显损失效果的前提下，减少数据加载和退化生成负担。

## 11. Batch-Length Tradeoff：为什么训练要用更长序列

论文还研究了一个很实际的问题：在计算资源固定时，应该选择更大的 batch size，还是更长的 sequence length？

比如总预算固定：

B × L = 480

可以选择：

B=48, L=10
B=24, L=20
B=16, L=30

论文发现，使用更长序列训练更好。原因是 RealBasicVSR 是时序传播模型，如果训练时只看短序列，测试时却处理长视频，就会出现训练和推理之间的 domain gap，模型不适应长时间传播，可能产生颜色伪影和细节扭曲。论文实验显示，随着 L 从 10 增加到 20、30，颜色伪影明显减少，因此建议在计算受限时优先使用更长序列而不是更大 batch。

这对你做项目很有用：

如果你训练视频超分模型，不要只追求 batch size，大量短片段训练不一定好；时序模型需要足够长的序列来学习传播稳定性。

## 12. VideoLQ 数据集

RealBasicVSR 还提出了 VideoLQ 数据集，用于真实世界视频超分评估。

原因是传统 VSR 数据集很多是合成退化，比如 bicubic 下采样；RealVSR 这类数据集虽然是真实拍摄，但退化主要来自特定设备，比如 iPhone 双摄系统，泛化性有限。VideoLQ 收集了来自不同视频平台的低质量真实视频，覆盖不同内容、分辨率和退化类型，用作真实世界 VSR 的公共评估基准。论文中说明 VideoLQ 包含多种真实低质量视频序列，具有丰富纹理和模式。

这说明 RealBasicVSR 关注的不是标准 benchmark 上 PSNR 多高，而是：

真实低清视频看起来能不能更清楚、更自然、更少伪影。

## 13. RealBasicVSR 的实验表现

论文在 VideoLQ 上比较了多种方法，包括图像模型 RealSR、DAN、Real-ESRGAN、BSRGAN，以及视频模型 BasicVSR++、RealVSR、DBVSR 等。实验表格中，RealBasicVSR 在 NRQM、NIQE、PI、BRISQUE 等无参考质量指标上取得最好结果，并且推理速度也较快；论文报告 RealBasicVSR 参数量约 6.3M，720×1280 输出尺寸下在 V100 上 runtime 为 63ms。

不过这里要注意：这些是论文环境和论文指标下的结果，不代表在所有真实视频上都一定最好。真实视频超分非常依赖输入退化类型，比如动漫、真人、监控、低码率直播、老电影胶片，最佳模型可能不同。

## 14. RealBasicVSR 和 Real-ESRGAN 的区别

很多人会把 RealBasicVSR 和 Real-ESRGAN 混在一起，因为它们都面向 real-world restoration。

但它们本质不一样。

对比	Real-ESRGAN	RealBasicVSR
类型	图像超分 / 单帧增强	视频超分
是否利用时序	不利用，逐帧处理	利用 BasicVSR 长时序传播
主要问题	单帧真实退化	真实视频退化 + 时序传播伪影
优点	部署简单，效果直观	时间一致性和细节聚合更好
缺点	视频可能闪烁	训练/推理更复杂

Real-ESRGAN 可以处理视频，但通常是拆帧逐帧处理。这样做简单，但没有真正利用相邻帧信息，容易出现纹理闪烁。

RealBasicVSR 真正把视频当作序列处理，能从远处帧聚合细节。论文定性图中也展示了它可以利用长时序信息恢复其他方法难以恢复的文字细节。

一句话：

Real-ESRGAN 是真实图像超分模型；RealBasicVSR 是真实视频超分模型。

## 15. RealBasicVSR 和 BasicVSR++ 的区别

BasicVSR++ 主要解决的是标准视频超分中的传播和对齐问题：

更强传播：二阶网格传播
更强对齐：光流引导可变形对齐

RealBasicVSR 解决的是真实视频超分中的退化传播问题：

真实退化复杂
长时序传播会放大伪影
所以先清理再传播

可以这样对比：

对比	BasicVSR++	RealBasicVSR
目标	标准 VSR / 合成退化	真实世界 VSR
主要矛盾	信息传播和帧间对齐不够强	真实退化会被传播放大
核心模块	二阶网格传播 + flow-guided DCN	image cleaning + BasicVSR
训练退化	多为标准退化	二阶真实退化 + 视频压缩
工程用途	高质量 benchmark / 离线 VSR	真实低清视频增强

所以如果你做面试项目：

标准数据集比如 Vimeo-90K、REDS：讲 BasicVSR++ 更合适；

真实低清视频、老视频、网络压缩视频：讲 RealBasicVSR 更合适。

## 16. RealBasicVSR 的优点
16.1 适合真实视频

它不是只针对 bicubic 退化，而是考虑真实视频中的复杂退化，并且训练时加入 blur、resize、noise、JPEG、video compression 等退化组合。

16.2 能利用长时序信息

相比 Real-ESRGAN 这种逐帧方法，RealBasicVSR 继承 BasicVSR 的双向传播能力，可以从远处帧聚合信息，恢复当前帧缺失细节。论文也强调 RealBasicVSR 能有效利用 long-term information 来合成细节。

16.3 抑制传播伪影

清理模块降低了输入退化对后续传播的影响，缓解了真实退化在循环传播中被放大的问题。

16.4 结构相对简单

它没有像 Transformer 视频模型那样复杂，也没有引入非常重的注意力结构，而是基于 BasicVSR 加一个 cleaning module。官方仓库也说明 RealBasicVSR 的代码和 demo 已公开，基于 MMEditing 体系实现。

## 17. RealBasicVSR 的局限
17.1 可能过度平滑

如果 cleaning 过强，噪声和伪影少了，但真实高频纹理也可能被清理掉，输出会偏平滑。Dynamic Refinement 虽然可以调节，但本质上仍然存在“干净”和“锐利”的权衡。论文也明确说，该方案允许在 smoothness 和 detailedness 之间做灵活权衡。

17.2 仍然依赖 BasicVSR 的对齐能力

RealBasicVSR 的 VSR backbone 是 BasicVSR，不是 BasicVSR++。所以它主要还是依赖光流特征 warping。极端运动、遮挡、复杂非刚体运动下，对齐问题仍然可能存在。

17.3 对不同类型视频泛化不一定一致

真实视频退化太复杂。监控视频、动漫视频、直播压缩视频、老电影、手机夜景视频退化类型不同。RealBasicVSR 虽然比标准 VSR 更适合真实场景，但不等于所有场景都最优。

17.4 工程部署比逐帧模型复杂

相比 Real-ESRGAN 拆帧逐帧处理，RealBasicVSR 需要处理序列输入、光流、传播状态、长视频分段、显存控制等问题。它更像真正的视频模型，工程复杂度自然更高。

## 18. 工程部署时要注意什么
18.1 输入格式

一般输入是：

B, T, C, H, W

输出是：

B, T, C, scale*H, scale*W

其中 T 是视频片段长度。

18.2 长视频要分段

如果视频很长，不能一次性把几千帧都送进去。常见做法是：

每次处理一段，比如 30 / 50 / 100 帧
相邻片段之间保留 overlap
最后再拼接

这样可以减少显存压力，同时避免片段边界处不连续。

18.3 Cleaning 次数要调

如果视频很脏，可以增加 dynamic refinement 次数；

如果视频本身比较干净，减少 cleaning，避免细节被抹掉。

18.4 真实项目中要配合视频编解码

完整视频增强工程通常是：

ffmpeg 解码视频帧
    ↓
RealBasicVSR 推理
    ↓
保存增强帧
    ↓
ffmpeg 合成视频
    ↓
保留或重新合成音频

如果做面试 demo，最好展示：

原视频；

增强后视频；

局部区域放大对比；

逐帧方法和 RealBasicVSR 的对比；

是否出现闪烁。

## 19. 面试时怎么讲 RealBasicVSR

RealBasicVSR 是面向真实世界视频超分的方法，可以看作在 BasicVSR 前面加入了图像清理模块。BasicVSR 在标准视频超分中通过双向时序传播利用长时序信息，但在真实视频中，输入往往包含噪声、压缩块、模糊和复杂伪影。如果直接做长时序传播，模型可能会把这些退化也当作细节传播并放大，导致输出出现严重伪影。RealBasicVSR 的核心思想是在时序传播之前先对每一帧做低分辨率清理，得到更干净的输入序列，再送入 BasicVSR 做超分。为了监督这个清理模块，作者使用下采样后的 HR 图像作为低分辨率 clean target，并使用 cleaning loss 约束它；测试时还设计了 dynamic refinement，根据连续清理结果的变化决定是否多次清理，从而在细节和伪影抑制之间做权衡。训练方面，它采用类似 Real-ESRGAN 的二阶退化，并加入视频压缩来模拟真实视频退化，同时提出 stochastic degradation 减少训练 I/O 开销，并指出在计算预算固定时，使用更长序列比更大 batch 更有利于时序稳定性。

最核心的一句话是：

RealBasicVSR 的本质是：先清理真实低清视频中的退化，避免噪声和伪影在 BasicVSR 的长时序传播中被放大，然后再利用多帧信息恢复高清细节。