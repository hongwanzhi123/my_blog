# FastDVDnet

FastDVDnet 是一个非常经典、偏工程实用的视频去噪模型，完整论文名是 FastDVDnet: Towards Real-Time Deep Video Denoising Without Flow Estimation，CVPR 2020。它的核心特点是：

不用显式光流，不做运动补偿，却能利用多帧时序信息完成视频去噪。

它可以看成 DVDnet 的改进版。DVDnet 需要先用光流把邻近帧对齐到中心帧，再做时间去噪；FastDVDnet 则直接通过多尺度 U-Net 结构和两阶段级联结构隐式处理运动，从而省掉昂贵的 optical flow / motion compensation，速度大幅提升。论文明确指出，FastDVDnet 在保持较好去噪质量和时间一致性的同时，避免了显式运动估计，并且运行速度比 DVDnet、VNLnet、VNLB 等方法快很多

## 1. FastDVDnet 解决的问题

视频去噪的目标是：

输入：带噪视频帧序列
输出：干净、稳定、少闪烁的视频帧序列

视频相比单张图像有一个优势：相邻帧之间有冗余信息。同一个物体会连续出现在多帧中，噪声通常是随机变化的，所以如果能利用多帧，就可以更好地区分“真实结构”和“随机噪声”。

但是视频去噪比图像去噪难，因为相邻帧之间有运动。如果直接融合多帧，容易出现重影、模糊和边缘错位。传统方法或 DVDnet 通常会使用光流进行运动补偿；FastDVDnet 的关键创新就是：不显式估计光流，而是让网络结构自己学习如何处理轻微错位和运动。 论文中也强调，FastDVDnet 通过 modified U-Net blocks、多尺度结构、两阶段级联和端到端训练来隐式处理运动，并避免光流错误带来的伪影。

## 2. FastDVDnet 的整体结构

FastDVDnet 输入 5 个连续视频帧，用来恢复中间帧：

输入：
I_{t-2}, I_{t-1}, I_t, I_{t+1}, I_{t+2}

输出：
去噪后的 I_t

它不是一次性把 5 帧全部丢进一个大网络，而是采用 两阶段级联结构：

5 帧输入
    ↓
第一阶段：三个 Denoising Block 1
    ↓
得到三个中间去噪结果
    ↓
第二阶段：一个 Denoising Block 2
    ↓
输出中心帧去噪结果

论文中的 Figure 1 明确说明，FastDVDnet 使用 5 个连续帧去噪中间帧；第一阶段把 5 帧拆成 3 个连续三帧组，分别送入 3 个共享权重的 Denoising Block 1；第二阶段再把这 3 个输出送入 Denoising Block 2，得到中心帧的估计。

具体来说：

第一阶段：

Block1 输入 1：
I_{t-2}, I_{t-1}, I_t
    ↓
输出中间结果 O_{t-1}

Block1 输入 2：
I_{t-1}, I_t, I_{t+1}
    ↓
输出中间结果 O_t

Block1 输入 3：
I_t, I_{t+1}, I_{t+2}
    ↓
输出中间结果 O_{t+1}

然后第二阶段：

Block2 输入：
O_{t-1}, O_t, O_{t+1}
    ↓
输出最终去噪中心帧 \hat{I}_t

这个设计非常巧妙。第一阶段先在局部三帧窗口内做初步去噪，第二阶段再对第一阶段的结果做进一步时间融合和细化。

## 3. 为什么不用光流也能处理运动

DVDnet 的逻辑是：

先估计光流
    ↓
把邻近帧 warp 到中心帧
    ↓
再融合多帧

FastDVDnet 的逻辑是：

不估计光流
    ↓
直接输入多帧
    ↓
让多尺度 U-Net 自己学习跨帧错位关系

为什么这样可行？

因为 FastDVDnet 的 denoising block 是一个 multi-scale U-Net。多尺度结构有更大的感受野，可以在不同尺度上观察相邻帧之间的对应关系。低分辨率尺度下，物体运动的像素位移会变小，网络更容易学习到帧间关系；高分辨率尺度负责恢复边缘和纹理细节。

论文明确说，FastDVDnet 避免了显式光流估计，但为了保持性能，引入了几项关键机制：多尺度 denoising block、两阶段级联结构、端到端训练和 residual learning。这些设计让模型能够隐式处理运动，并避免光流错误造成的伪影。

直观理解：

显式光流：
先告诉网络“这个点移动到了哪里”

FastDVDnet：
不给光流，让 U-Net 在多尺度特征里自己学“哪里和哪里相关”

这也是 FastDVDnet 比 DVDnet 更快、更简单的重要原因。

## 4. Denoising Block 的结构

FastDVDnet 的基本单元是 Denoising Block。Denoising Block 1 和 Denoising Block 2 的结构相同，都是 modified U-Net。

一个 Denoising Block 的输入是：

3 帧图像 + noise map

输出是：

去噪后的中心帧

论文中说明，Denoising Block 与标准 U-Net 有几个不同：

编码器输入被改成 三帧 + 噪声图；
解码器中的上采样使用 PixelShuffle，减少 gridding artifacts；
编码器和解码器的跳连融合使用 逐像素相加，而不是通道拼接，以降低显存；
模块使用 residual learning，在中心噪声帧和输出之间建立残差连接；
每个 denoising block 总共包含 16 个卷积层，大多数卷积后接 ReLU 和 BatchNorm。

可以把 Denoising Block 理解成：

三帧输入
    ↓
Encoder：逐步下采样，扩大感受野
    ↓
Bottleneck：融合多帧上下文
    ↓
Decoder：逐步上采样
    ↓
Skip connection：保留局部细节
    ↓
Residual output：输出去噪中心帧

它不是普通单帧 U-Net，而是一个 时空 U-Net：空间上通过 encoder-decoder 建模局部结构，时间上通过三帧输入学习帧间冗余。

## 5. Noise Map 的作用

FastDVDnet 和 DVDnet 一样，也引入了 noise map。

noise map 是一张和输入图像空间大小相关的图，用来告诉网络当前噪声强度。对于高斯噪声，如果整张图噪声强度相同，那么 noise map 是常数图：

M(x, y) = σ

例如：

σ = 25

那么 noise map 每个位置都填 25 或归一化后的对应值。

论文中说明，noise map 作为单独输入，表示输入噪声的每像素标准差；对于高斯噪声它是常量图，对于泊松噪声它可以依赖图像强度。noise map 还能作为用户控制参数，在“去噪强度”和“细节保留”之间做权衡。

它的意义是：

噪声小：
网络少去噪，保留更多细节

噪声大：
网络强去噪，压制更多噪声

这样一个模型就可以处理多个噪声强度，而不需要为每个 σ 单独训练一个模型。

## 6. 两阶段级联为什么重要

FastDVDnet 的两阶段设计不是随便堆的。论文做过对比：如果直接用一个 block 输入 5 帧，效果不如两阶段级联，最多会有接近 0.9 dB 的 PSNR 差距，而且单阶段 5 输入结构更容易出现 flickering。

为什么两阶段更好？

因为它把难题分解了。

如果一次性输入 5 帧，网络需要同时完成：

1. 从 5 帧中找有用信息
2. 处理帧间运动
3. 去除噪声
4. 保留细节
5. 保证时间一致性

任务比较复杂。

两阶段结构则是：

第一阶段：
在三个局部三帧窗口中做初步去噪

第二阶段：
对第一阶段的三个中间结果做进一步融合和细化

也就是：

原始 noisy frames
    ↓
局部初步去噪
    ↓
更干净、更稳定的中间帧
    ↓
再次融合
    ↓
最终中心帧

这有点像人看视频去噪：先从小范围邻近帧中降低噪声，再从更稳定的中间结果里恢复中心帧。

## 7. FastDVDnet 如何减少闪烁

视频去噪不只是每帧 PSNR 高，还要播放时稳定。单帧去噪模型容易出现：

第 t 帧残留一种噪声
第 t+1 帧残留另一种噪声
连续播放时残余噪声随机跳动
    ↓
闪烁

FastDVDnet 的解决方式是利用多帧输入和两阶段级联，让相邻输出帧之间的残余误差更相关、更稳定。

论文指出，使用 temporal neighbors 有两个好处：一是相邻帧提供额外信息帮助去噪，二是可以减少 flickering，因为每帧残余误差会更相关。论文实验也强调 FastDVDnet 输出具有很好的 temporal coherence、低 flickering 和较好的细节保留。

直观理解：

真实结构：
多帧中持续出现 → 保留

随机噪声：
每帧随机变化 → 去掉

时间不稳定残差：
通过多帧融合被压低

所以 FastDVDnet 比单帧去噪网络更适合视频。

## 8. FastDVDnet 的训练方式

FastDVDnet 主要在合成 AWGN 噪声设定下训练。

论文训练细节大致是：

训练数据：DAVIS training set
输入 patch：5 帧连续视频 patch
空间大小：96 × 96
时间长度：5 帧
噪声类型：AWGN
噪声强度：σ ∈ [5, 50]
优化器：Adam
训练轮数：80 epochs
batch size：96

论文中明确说，训练样本是从 DAVIS 数据集中随机裁剪的 5 帧时空 patch，给 clean patches 添加 σ∈[5,50] 的 AWGN，并构建对应常量 noise map；损失函数是网络输出中心帧和 clean center patch 之间的 L2 loss。

训练目标可以写成：

输入：
noisy 5-frame sequence + noise map

目标：
clean center frame

损失：

L = || \hat{I}_t - I_t ||^2

也就是说，FastDVDnet 不是输出 5 帧，而是针对当前窗口输出中心帧。

## 9. FastDVDnet 推理流程

假设输入视频：

I_1, I_2, I_3, ..., I_T

要去噪第 t 帧，取窗口：

I_{t-2}, I_{t-1}, I_t, I_{t+1}, I_{t+2}

然后：

第一阶段：
Denoising Block 1(I_{t-2}, I_{t-1}, I_t)     → O_{t-1}
Denoising Block 1(I_{t-1}, I_t, I_{t+1})     → O_t
Denoising Block 1(I_t, I_{t+1}, I_{t+2})     → O_{t+1}

第二阶段：
Denoising Block 2(O_{t-1}, O_t, O_{t+1})     → \hat{I}_t

整段视频就是滑动窗口逐帧处理：

窗口 1：输出第 3 帧
窗口 2：输出第 4 帧
窗口 3：输出第 5 帧
...

视频开头和结尾没有足够邻近帧时，工程上一般用：

边界复制
镜像 padding
循环 padding

最常见是边界复制或镜像填充。

## 10. FastDVDnet 和 DVDnet 的区别

这是面试很容易问的点。

对比	DVDnet	FastDVDnet
是否用光流	使用显式光流 / 运动补偿	不使用显式光流
结构	空间去噪 + 光流对齐 + 时间去噪	两阶段级联 multi-scale U-Net
训练方式	两部分分开训练	端到端训练
运动处理	依赖光流对齐	网络隐式学习运动
速度	光流耗时大	快很多
伪影风险	光流错会产生伪影	避免光流错误伪影
适用性	结构清晰但工程较重	更适合快速视频去噪 baseline

论文明确指出，FastDVDnet 与 DVDnet 最大区别是：不再使用显式运动估计，而是通过架构隐式处理运动；这样不仅减少运行时间，也避免了遮挡、强噪声等情况下错误光流带来的伪影。

一句话：

DVDnet 是“先对齐再融合”，FastDVDnet 是“不显式对齐，让网络自己学融合”。

## 11. FastDVDnet 和 VRT / RVRT 的区别

FastDVDnet 是轻量 CNN 视频去噪模型；VRT / RVRT 是更复杂的视频复原 Transformer。

对比	FastDVDnet	VRT / RVRT
主体结构	CNN / U-Net	Transformer / Attention
输入窗口	5 帧	可处理更长序列
对齐方式	无显式对齐	attention、warping、GDA 等
速度	快	较慢
显存	低	高
效果上限	中等偏强	高质量离线更强
工程定位	实用 baseline / 轻量去噪	高质量视频复原

如果你做面试项目，FastDVDnet 的优点是非常好讲、容易跑通、代码简单、速度较快；缺点是它主要针对 AWGN，不是最新最强的真实视频去噪方案。

## 12. FastDVDnet 的优点

第一，速度快。
FastDVDnet 省掉了 DVDnet 中最耗时的光流运动补偿，论文称其运行速度比 VNLB 快三个数量级，比 DVDnet 和 VNLnet 快一个数量级。

第二，结构简单。
输入 5 帧，两阶段级联，核心模块是 modified U-Net，没有复杂光流、DCN 或 Transformer。

第三，时间一致性好。
论文中指出 FastDVDnet 输出具有明显的 temporal coherence 和较低 flickering，尤其在平坦区域能减少低频残余噪声的时间不一致问题。

第四，一个模型支持多个噪声强度。
通过 noise map，FastDVDnet 可以处理不同噪声水平，不需要为每个 σ 单独训练模型。

第五，避免光流错误伪影。
在遮挡、强噪声、大运动区域，显式光流容易错；FastDVDnet 不依赖光流，因此避免了一类由错误 flow 引入的伪影。

## 13. FastDVDnet 的缺点

第一，主要针对 AWGN。
原论文实验主要是加性白高斯噪声。真实视频噪声往往包括压缩块、低光照彩噪、ISP 锐化伪影、码率损伤等，直接用 FastDVDnet 不一定最稳。论文也说明实验主要关注 AWGN，虽然方法可以扩展到空间变化噪声或其他噪声类型。

第二，长时序建模有限。
FastDVDnet 通常只看 5 帧窗口，不像 BasicVSR / RVRT 那样有长时序传播。

第三，大运动场景仍可能有融合不足。
虽然它可以隐式处理运动，但没有显式光流或 attention 的长距离匹配能力。运动过大、遮挡复杂时，可能不如 RVRT、VRT 或带对齐机制的方法。

第四，输出中心帧，需要滑动窗口。
它不是一次输出整段视频，而是窗口式逐帧处理；工程上要处理边界帧、padding、滑窗效率等问题。

## 14. 工程上怎么用 FastDVDnet

完整工程流程通常是：

视频解码
    ↓
读取连续帧
    ↓
归一化到 [0,1]
    ↓
估计或指定 noise level σ
    ↓
构造 noise map
    ↓
5 帧滑动窗口输入 FastDVDnet
    ↓
输出去噪帧
    ↓
保存帧序列
    ↓
ffmpeg 合成视频并保留音频

官方 GitHub 提供了 PyTorch 实现，并且说明这是一个不使用 motion compensation 的快速深度视频去噪算法。

工程注意点：

噪声强度要设对。
σ 设太小，噪声残留；σ 设太大，细节被抹掉。
边界帧要 padding。
前两帧和后两帧没有完整 5 帧窗口，需要复制或镜像填充。
真实视频最好先判断噪声类型。
如果是压缩块效应，FastDVDnet 未必是最合适的；可能需要视频去压缩模型或真实退化训练。
可以作为项目 baseline。
先跑 FastDVDnet，再和传统 FFmpeg hqdn3d、OpenCV NLM、RVRT 等对比，项目完整度会更高。

## 15. 面试里怎么讲 FastDVDnet

FastDVDnet 是一个经典的快速视频去噪网络，可以看作 DVDnet 的改进版。DVDnet 需要显式估计光流，把邻近帧运动补偿到中心帧后再做时间去噪，但光流计算耗时，而且在遮挡、强噪声和大运动场景下容易出错。FastDVDnet 取消了显式 motion compensation，采用两阶段级联的 modified U-Net 结构来隐式处理帧间运动。具体来说，它输入 5 个连续帧，第一阶段把它们拆成 3 个三帧组，分别送入共享权重的 Denoising Block，得到 3 个中间去噪结果；第二阶段再把这 3 个中间结果输入另一个 Denoising Block，输出中心帧的去噪结果。每个 Denoising Block 是一个多尺度 U-Net，输入三帧和 noise map，使用 PixelShuffle 上采样、残差学习和跳连结构。FastDVDnet 的优势是速度快、结构简单、无需光流、时间一致性较好，并且通过 noise map 支持多个噪声强度；缺点是主要针对 AWGN，真实低光噪声和压缩噪声场景下可能需要额外训练或更复杂模型。

最后记住一句话：

FastDVDnet 的本质是：用两阶段级联的多尺度 U-Net，在不显式估计光流的情况下，从 5 帧邻域中隐式学习时空冗余，从而实现快速、稳定的视频去噪。

