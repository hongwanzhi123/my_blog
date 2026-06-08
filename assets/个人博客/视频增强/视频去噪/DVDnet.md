# DVDnet

DVDnet 是深度视频去噪里比较早、比较经典的 CNN 方法，全名可以理解为 Deep Video Denoising Network。论文题目是 “DVDnet: A Fast Network for Deep Video Denoising”，发表于 ICIP 2019。它的定位是：用一个相对简单、速度较快的 CNN 框架，替代传统 VNLB、V-BM4D 这类很慢的 patch-based 视频去噪方法，同时保证较好的时间一致性。 论文强调 DVDnet 输出具有较好的 temporal coherence、较低 flickering、较强降噪和细节保留能力

## 1. DVDnet 要解决什么问题

视频去噪的目标是：给定一段带噪视频，输出一段干净视频。

假设干净视频帧是：

I_t

带噪视频帧是：

Ĩ_t = I_t + N_t

其中 N_t 是噪声。DVDnet 主要研究的是 AWGN，加性白高斯噪声，也就是比较标准的合成噪声设定。论文实验集中在 zero-mean white Gaussian noise，并说明方法可以扩展到空间变化噪声或其他噪声类型。

视频去噪和图像去噪的区别在于：视频中有多帧信息。当前帧某个区域很吵，但相邻帧里同一个物体可能也出现了，如果能把相邻帧对齐过来，就可以利用多帧平均和互补信息降低噪声。

但是直接用相邻帧有两个问题：

第一，不同帧之间有运动，不能直接相加；

第二，如果每帧独立去噪，单帧看起来还行，但播放时容易出现闪烁。

DVDnet 的核心设计就是为了解决这两个问题：

先做单帧空间去噪，再做运动补偿，然后用时间去噪网络融合多帧，减少闪烁并提升细节稳定性。

## 2. DVDnet 的整体流程

DVDnet 是一个 两阶段视频去噪框架：

输入连续 5 帧带噪视频
        ↓
第一阶段：Spatial Denoiser
每一帧单独做空间去噪
        ↓
运动估计 / 运动补偿
把相邻帧对齐到中心帧
        ↓
第二阶段：Temporal Denoiser
融合中心帧和对齐后的邻近帧
        ↓
输出中心帧去噪结果

论文中描述 DVDnet 处理中心帧时，会把它的 2T 个邻近帧也作为输入；第一阶段对 2T+1 帧分别做空间去噪，第二阶段把去噪后的邻近帧通过光流配准到中心帧，再把 2T+1 个对齐帧拼接输入 temporal denoising block。

常见设置是：

T = 2
输入帧数 = 2T + 1 = 5

也就是：

I_{t-2}, I_{t-1}, I_t, I_{t+1}, I_{t+2}

最终输出：

去噪后的 I_t

## 3. 第一阶段：Spatial Denoiser 空间去噪

DVDnet 的第一阶段是 单帧去噪。

它对每一帧分别进行去噪：

Ĩ_{t-2} → Î_{t-2}
Ĩ_{t-1} → Î_{t-1}
Ĩ_t     → Î_t
Ĩ_{t+1} → Î_{t+1}
Ĩ_{t+2} → Î_{t+2}

这一阶段的作用是：先把每一帧中的明显噪声压下去，让后续光流估计和运动补偿更容易。

为什么要先单帧去噪再估计运动？

因为带噪图像直接估计光流会比较困难。噪声会干扰特征匹配，导致光流不稳定。如果先做一遍空间去噪，画面更干净，运动估计也更可靠。

论文也明确提到，两阶段拆分的好处之一是可以先对每一帧做 individual pre-processing，而且 motion compensation 是在 pre-denoised images 上进行的，这会简化运动补偿任务。

## 4. 第二阶段：Temporal Denoiser 时间去噪

第一阶段得到的是每帧独立去噪结果。问题是：

每一帧单独去噪 → 单帧质量不错
但连续播放 → 可能闪烁

因为每一帧的残余误差不同，噪声残留在时间上不一致，就会出现 flickering。

所以 DVDnet 第二阶段使用 temporal denoiser。它会把相邻帧的信息融合起来，让输出结果在时间上更稳定。

流程是：

空间去噪后的邻近帧
        ↓
用光流对齐到中心帧
        ↓
与中心帧一起 concat
        ↓
输入 temporal denoiser
        ↓
输出中心帧最终去噪结果

论文中解释，使用 temporal neighbors 有助于减少 flickering，因为每帧的残余误差会更相关、更稳定。

所以 DVDnet 的 temporal denoiser 不是简单做时域平均，而是一个 CNN，用来学习：

哪些邻近帧信息可靠
哪些区域应该融合
哪些残余噪声应该进一步去掉
哪些细节应该保留

## 5. 运动补偿：为什么 DVDnet 需要光流

DVDnet 和后来的 FastDVDnet 一个很大区别是：DVDnet 使用显式运动补偿，FastDVDnet 不使用运动补偿。

DVDnet 的第二阶段需要把相邻帧对齐到中心帧。比如要恢复第 t 帧，输入有：

Î_{t-2}, Î_{t-1}, Î_t, Î_{t+1}, Î_{t+2}

其中 Î_{t-1} 和 Î_{t+1} 中的物体位置可能和 Î_t 不一样。如果直接 concat，网络会看到错位内容，容易产生重影或模糊。

所以 DVDnet 会估计光流，然后做 warping：

Î_{t-1} → warp 到 t 帧坐标
Î_{t+1} → warp 到 t 帧坐标

最终 temporal denoiser 输入的是：

warp(Î_{t-2}), warp(Î_{t-1}), Î_t, warp(Î_{t+1}), warp(Î_{t+2})

论文训练 temporal denoiser 时使用 DeepFlow 做 optical flow 估计和 motion compensation；GitHub README 也说明 DVDnet 和 VNLB 的 flow maps 使用 DeepFlow 计算。

这也是 DVDnet 的一个工程痛点：光流计算很耗时。

论文中报告，DVDnet 去噪一帧 960×540 彩色视频在 GPU 上大约 8 秒，其中约 6 秒花在 temporal neighboring frames 的 motion compensation 上；相比 V-BM4D 和 VNLB，它仍然快很多，但运动补偿占了主要耗时。

## 6. Noise Map 噪声图：一个模型处理多种噪声强度

DVDnet 还有一个重要输入：noise map。

普通去噪网络经常针对某个固定噪声强度训练，比如只处理：

σ = 25

但真实或工程环境中，噪声强度可能变化：

σ = 5, 10, 25, 50 ...

DVDnet 希望一个模型能处理多个噪声强度，所以它把噪声强度作为一张图输入网络：

noise map M

如果整张图噪声强度相同，noise map 里所有位置都是同一个值：

M(x, y) = σ

如果噪声是空间变化的，也可以让不同位置有不同噪声强度。

论文明确说，spatial 和 temporal denoiser 都加入 noise map 作为输入，目的是允许处理 spatially varying noise。 官方 README 也说明模型训练噪声范围是 [5, 55]。

这个设计很工程化，因为推理时你只需要传入：

视频序列 + 噪声强度估计

模型就能根据噪声强弱调整去噪力度。

## 7. Residual Learning：预测噪声而不是直接预测干净图

DVDnet 的 spatial denoiser 使用了 residual learning。

也就是说，网络不是直接输出干净图：

网络输出 = clean image

而是输出噪声估计：

网络输出 = estimated noise

然后：

denoised image = noisy image - estimated noise

论文中公式写得很清楚：如果 spatial denoiser 输出输入噪声估计 F_spa(Ĩ; θ_spa) = N_hat，那么去噪结果就是 Ĩ - F_spa(Ĩ; θ_spa)。

为什么这样做有效？

因为去噪任务里，输入图像和输出图像大部分内容相同，真正需要模型学习的是“哪些部分是噪声”。让网络学习残差通常比直接生成整张干净图更容易。

直观理解：

直接预测干净图：
网络要学图像内容 + 去噪

预测噪声：
网络主要学噪声分布，然后从输入中减掉

这和 DnCNN、FFDNet 等经典图像去噪方法的思想是一致的。

## 8. DVDnet 的网络结构细节

DVDnet 的 spatial denoising block 和 temporal denoising block 都是标准 feed-forward CNN。

论文给出的关键结构参数是：

模块	卷积层数	特征通道数	卷积核	激活
Spatial denoiser	12 层	96	3×3	ReLU
Temporal denoiser	6 层	96	3×3	ReLU

训练时，卷积和 ReLU 之间有 BatchNorm；测试时，BN 被替换成等价的 affine layer。两个 block 都有 residual connection，并且输入会先降到四分之一分辨率处理，再恢复到原分辨率，以降低运行时间和内存开销。

这里“降到四分之一分辨率”可以理解成类似 FFDNet 的思想：把空间分辨率降低，把局部邻域信息重新组织到通道维度中。好处是：

H × W 的大图
    ↓
H/2 × W/2 的特征

空间尺寸变小，卷积计算量下降，但局部信息仍然保留在通道里。

## 9. DVDnet 的训练方式

DVDnet 的两个阶段是分开训练的：

先训练 spatial denoiser
再用 spatial denoiser 的输出训练 temporal denoiser

论文中说，spatial 和 temporal denoising parts are trained separately，先训练 spatial denoiser，因为它的输出要用于训练 temporal denoiser。

9.1 Spatial denoiser 训练

Spatial denoiser 使用 Waterloo Exploration Database 中裁剪的图像 patch 训练。训练时给干净图加 AWGN，噪声强度从 [0, 55] 采样，patch 大小是 50×50。

训练样本可以表示为：

输入：noisy patch + noise map
目标：clean patch

或者从 residual learning 角度看：

输入：noisy patch + noise map
网络预测：noise
输出：noisy patch - predicted noise
9.2 Temporal denoiser 训练

Temporal denoiser 使用 DAVIS 训练集，patch 大小是 44×44，时间长度是 5 帧，也就是 2T+1=5。训练时先给干净视频 patch 加噪声，再经过 spatial denoiser，之后用 DeepFlow 把邻近帧 motion-compensate 到中心帧，最后训练 temporal denoiser 输出中心帧干净结果。

训练样本可以表示为：

输入：
对齐后的 5 帧 spatial-denoised patches + noise map

目标：
中心帧 clean patch

## 10. DVDnet 的推理流程

假设你有一段视频：

Frame 1, Frame 2, Frame 3, ..., Frame T

要去噪第 t 帧，DVDnet 大致这样做：

1. 取 5 帧窗口：
   t-2, t-1, t, t+1, t+2

2. 每一帧单独经过 spatial denoiser：
   得到初步去噪结果

3. 用光流估计相邻帧到中心帧的运动：
   t-2 → t
   t-1 → t
   t+1 → t
   t+2 → t

4. 对邻近帧做 motion compensation：
   把它们 warp 到中心帧坐标系

5. 拼接 5 帧和 noise map：
   输入 temporal denoiser

6. 输出第 t 帧最终去噪结果

对于整段视频，就是滑动窗口逐帧处理：

窗口 1：恢复第 3 帧
窗口 2：恢复第 4 帧
窗口 3：恢复第 5 帧
...

视频开头和结尾窗口不完整时，通常需要边界复制、镜像 padding 或只从中间帧开始输出。

## 11. DVDnet 为什么能减少闪烁

单帧去噪模型的问题是：

第 t 帧残留一些噪声
第 t+1 帧残留另一些噪声
播放时这些残留不一致 → 闪烁

DVDnet 使用对齐后的邻近帧作为输入，让 temporal denoiser 在同一空间位置看到多个时间样本：

同一个物体位置在 5 帧中的表现

这样网络可以判断哪些内容是稳定结构，哪些内容是随机噪声。

稳定结构：

多帧都出现 → 应该保留

随机噪声：

每帧随机变化 → 应该去掉

所以 DVDnet 不只是提升 PSNR，更重要的是提升视频播放时的时间稳定性。

论文也明确把 temporal coherence 和 low flickering 作为 DVDnet 输出的重要特性。

## 12. DVDnet 和传统 VNLB / V-BM4D 的关系

在 DVDnet 之前，视频去噪强方法主要是传统 patch-based 方法，比如：

V-BM4D
VNLB

这些方法通常会在空间和时间邻域中搜索相似 patch，然后做协同滤波或贝叶斯估计。它们效果不错，但速度慢。

论文中提到，当时 VNLB 在质量上是很强的视频去噪算法，但运行时间很长，甚至去噪单帧可能需要几分钟；DVDnet 在中高噪声下和 VNLB 相比有竞争力，同时速度快很多。

可以这样理解：

方法	思路	优点	缺点
V-BM4D	传统块匹配 + 4D 变换滤波	稳定、无需训练	慢，参数复杂
VNLB	非局部贝叶斯 patch 方法	质量强	非常慢
DVDnet	CNN + 运动补偿 + 多帧融合	快，时间稳定性好	依赖光流，真实噪声泛化有限

## 13. DVDnet 和 FastDVDnet 的区别

FastDVDnet 是 DVDnet 的后续改进。官方 FastDVDnet 仓库明确说，它是一个不使用 motion compensation 的快速深度视频去噪方法，并标注 DVDnet 是 previous deep video denoising algorithm。

二者核心区别是：

对比	DVDnet	FastDVDnet
是否使用光流	使用 DeepFlow 做运动补偿	不使用显式运动补偿
结构	spatial denoiser + motion compensation + temporal denoiser	多级 CNN 直接融合多帧
速度	比传统方法快，但光流耗时大	更快，更适合实时
工程复杂度	需要光流模块	更简单
风险	光流错误会影响结果	由网络隐式学习对齐，避免光流开销

为什么 FastDVDnet 后来更常用？

因为 DVDnet 的主要瓶颈在 motion compensation。论文中 DVDnet 去噪一帧 960×540 约 8 秒，其中约 6 秒用于运动补偿。 FastDVDnet 取消显式光流后，工程上更快、更简单。

所以你可以把 DVDnet 看成：

深度视频去噪从“显式运动补偿 + CNN 融合”走向“无光流 CNN 隐式融合”的过渡代表。

## 14. DVDnet 的优点
14.1 结构清晰，容易理解

DVDnet 把视频去噪拆成三个很自然的步骤：

单帧去噪
运动对齐
多帧融合

这个逻辑非常适合面试讲解。

14.2 时间一致性比单帧去噪好

因为 temporal denoiser 会融合邻近帧，输出不容易出现逐帧随机变化的残余噪声。

14.3 一个模型可以处理多个噪声强度

noise map 的设计让 DVDnet 不需要为每个 σ 单独训练模型。官方 README 也说明预训练模型覆盖噪声范围 [5,55]。

14.4 比传统强去噪方法快很多

论文报告 DVDnet 对 960×540 彩色帧的 GPU 推理约 8 秒，比 V-BM4D 和 VNLB 快很多；但也要注意，这个速度以 2019 年实验设置为背景。

## 15. DVDnet 的缺点
15.1 依赖光流 / 运动补偿

DVDnet 的 temporal denoiser 依赖对齐后的邻近帧。如果光流估计错误，邻近帧内容就会错位，可能导致模糊、重影或细节损失。

典型困难场景：

大运动
遮挡
低纹理区域
强噪声区域
运动模糊
边界区域
15.2 运动补偿耗时大

论文中 DVDnet 的主要耗时来自 temporal neighboring frames 的 motion compensation。 这也是后续 FastDVDnet 取消显式运动补偿的重要原因。

15.3 主要面向 AWGN

DVDnet 实验主要关注高斯噪声。真实视频噪声通常更复杂，包括：

压缩噪声
低光照彩噪
ISP 锐化伪影
传感器固定模式噪声
码率不足导致的块效应

所以 DVDnet 直接用于真实视频可能不如 RealBasicVSR、RVRT、EMVD、真实 RAW 去噪模型稳定。

15.4 逐中心帧滑动处理，长时序利用有限

DVDnet 通常用 5 帧窗口恢复中心帧。它不像 BasicVSR / RVRT 那样通过循环传播利用很长的视频信息。因此长时序建模能力有限。

## 16. 工程上怎么理解 DVDnet

如果你要把 DVDnet 放在工程视频去噪体系里，可以这样定位：

传统方法：
V-BM4D / VNLB
    ↓
早期深度视频去噪：
DVDnet = 光流运动补偿 + CNN 多帧融合
    ↓
更实用快速深度去噪：
FastDVDnet = 无光流多帧 CNN
    ↓
端侧实时 / RAW：
EMVD / BSVD / RViDeNet
    ↓
高质量离线：
VRT / RVRT

所以现在工程上直接用 DVDnet 的场景不算最多，但它非常适合学习视频去噪基本思想：

多帧信息利用
运动补偿
时间一致性
噪声图控制去噪强度
残差学习
两阶段去噪

## 17. 面试里怎么讲 DVDnet


DVDnet 是一个经典的深度视频去噪方法，它把视频去噪拆成两阶段。第一阶段是 spatial denoiser，对输入窗口中的每一帧单独进行空间去噪，先去掉大部分噪声；第二阶段是 temporal denoiser，先用光流把已经空间去噪的邻近帧运动补偿到中心帧坐标系，再把中心帧和对齐后的邻近帧拼接输入时间去噪网络，输出中心帧最终结果。这样做的好处是，空间去噪可以让后续光流估计更可靠，时间去噪可以利用多帧信息减少残余噪声和闪烁。DVDnet 还引入 noise map，使一个模型可以处理不同噪声强度；同时采用 residual learning，让网络预测噪声残差，再从输入中减去噪声。它的优点是结构清晰、速度相比传统 VNLB/V-BM4D 快很多、时间一致性较好；缺点是依赖光流运动补偿，光流计算耗时且在遮挡和大运动下容易出错，所以后续 FastDVDnet 取消了显式运动补偿，改用 CNN 直接从多帧中隐式学习时空融合。

最后记住一句话：

DVDnet 的本质是：先单帧空间去噪，再用光流把邻近帧对齐到中心帧，最后用时间 CNN 融合多帧信息，从而在降低噪声的同时减少视频闪烁。