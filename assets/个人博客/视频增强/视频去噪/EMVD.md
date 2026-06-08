# EMVD

EMVD 全名是 Efficient Multi-Stage Video Denoising with Recurrent Spatio-Temporal Fusion，是 CVPR 2021 的一个高效视频去噪模型。它的定位不是追求最大模型、最高离线指标，而是解决一个非常工程化的问题：如何在移动端/端侧设备上，用很低的计算量完成高质量视频去噪。论文明确提出 EMVD 由三个级联阶段组成：temporal fusion 时域融合、spatial denoising 空域去噪、spatio-temporal refinement 时空精修，并且这些阶段以 recurrent 的方式处理视频序列。

## 1. EMVD 解决的是什么问题

视频去噪的核心目标是：

输入：带噪视频帧序列
输出：干净、稳定、细节尽量保留的视频

相比图像去噪，视频去噪有一个优势：相邻帧之间存在大量重复信息。真实结构在多帧中通常持续存在，而噪声往往是随机的。所以理论上，多帧信息可以帮助降低噪声。

但工程上有三个难点：

1. 多帧模型通常计算量大
2. 显式光流/运动估计很耗时
3. 端侧设备内存和算力有限

很多高质量视频去噪模型，比如 EDVR、RViDeNet、FastDVDnet，效果不错，但在移动端实时运行很困难。EMVD 的目标就是：在非常低的 GFLOPs 和内存占用下，尽可能接近甚至超过复杂模型的去噪效果。论文报告 EMVD 在 Huawei P40 Pro 商用 SoC 上处理 720p 序列时达到 36ms、112MB DDR，占用明显低于 FastDVDnet，同时 PSNR/SSIM 更高。

## 2. EMVD 的一句话理解

EMVD 可以这样理解：

它不是一次输入很多帧做大网络融合，而是每来一帧，就把当前帧和历史融合结果递归融合，再做轻量去噪和细节精修。

整体流程是：

当前 noisy frame
        ↓
可学习可逆变换：分离颜色和频率信息
        ↓
Temporal Fusion：当前帧 + 历史融合结果
        ↓
Spatial Denoising：去除融合后残余噪声
        ↓
Spatio-Temporal Refinement：从融合图中找回高频细节
        ↓
逆变换
        ↓
输出当前去噪帧
        ↓
把融合状态传给下一帧

所以 EMVD 的本质是：

历史多帧信息压缩在一个 recurrent state 里
当前帧来了以后只和这个 state 融合
不需要每次都把很多邻近帧重新送进大网络

这就是它高效的根本原因。

## 3. EMVD 和 FastDVDnet / DVDnet 的区别

你前面问过 DVDnet 和 FastDVDnet，所以这里先对比一下。

方法	多帧利用方式	是否显式光流	输入方式	工程特点
DVDnet	5 帧窗口 + 光流对齐	使用光流	滑动窗口	结构清楚，但光流耗时
FastDVDnet	5 帧窗口 + U-Net 隐式融合	不使用光流	滑动窗口	快，适合 baseline
EMVD	历史融合结果 + 当前帧递归融合	不依赖显式光流	recurrent 单帧流式处理	更适合端侧实时

FastDVDnet 每次要取 5 帧窗口恢复中心帧，而 EMVD 更像在线流式模型：第 t 帧只需要当前帧和上一时刻保存下来的融合状态。这样它天然适合实时视频流、手机相机、监控摄像头这类场景。

## 4. EMVD 的输入：RAW 视频，而不是普通 RGB 视频

EMVD 主要面向 RAW video denoising。论文中的观测模型使用 packed raw 数据，例如 Bayer CFA 的 RG1G2B 四通道形式，并假设噪声是 signal-dependent 的异方差高斯噪声，也就是噪声方差和信号强度有关。论文中将噪声方差写成和 shot noise、read noise 相关的形式，这很符合真实相机 RAW 低光噪声的特点。

这点很重要。RAW 域和 sRGB 域不一样：

RAW 域：
传感器原始数据
噪声模型更清楚
没有经过锐化、压缩、色彩映射
更适合低光去噪

sRGB 域：
已经经过 ISP、demosaic、白平衡、色彩变换、锐化、压缩
噪声和伪影更复杂

所以 EMVD 更像是相机 ISP pipeline 前段的低照度视频去噪模块，而不是普通视频后处理软件中的 RGB 去噪模型。

## 5. 核心设计一：Learnable Invertible Transform 可学习可逆变换

EMVD 不是直接在原始 RAW 通道上处理，而是先做一个可学习的变换，把数据映射到更适合去噪的变换域。这个变换包含两类：

1. Color Transform：颜色变换
2. Frequency Transform：频率变换

论文说明，color transform 用 point-wise convolution 实现，用于把 packed RAW 的多个颜色通道解耦到类似亮度/色度的表示；frequency transform 则受到 wavelet 变换启发，用可学习线性滤波器分离不同频率信息。两个变换都是线性的、可逆的，并通过正则项约束可逆性。

直观理解：

原始 RAW：
颜色和频率信息混在一起
噪声也混在里面

变换域：
低频 / 高频更容易区分
亮度 / 色度更容易区分
有利于网络用更小模型完成去噪

这和传统图像处理里的 wavelet denoising 有点像。传统方法会把图像分解成低频和高频，低频保留结构，高频处理噪声和细节。EMVD 的区别是：它不固定使用 Haar 或 DCT，而是让网络学习一组更适合 RAW 视频去噪的可逆变换。

## 6. 核心设计二：Temporal Fusion 时域融合

Temporal Fusion 是 EMVD 的第一阶段，也是最关键的阶段。

它的目标是：

利用历史帧信息降低当前帧噪声，同时避免运动区域产生拖影。

EMVD 不保存所有历史帧，而是保存上一时刻的 fused frame。当前时刻输入是：

当前 noisy frame：z_t
上一时刻 fused frame：ȳ_{t-1}

然后通过一个小网络 FCNN 预测融合权重，把两者做凸组合：

当前融合结果 = 历史融合结果 × 历史权重 + 当前帧 × 当前权重

论文中写成递归凸组合，并且融合权重由 FCNN 根据当前帧低频部分、上一融合帧低频部分的差异以及噪声方差预测。论文图示也说明，预测权重可以区分动态区域和静态区域。

直观上：

静态区域：
多帧内容一致
可以更多使用历史融合结果
噪声会越平均越小

运动区域：
当前帧和历史帧差异大
应该更多相信当前帧
避免拖影

所以 temporal fusion 的本质是一个自适应时域滤波器：

静止处强融合
运动处弱融合

这和传统视频降噪里的时域平均很像，但 EMVD 用 CNN 自适应预测融合权重，避免简单平均带来的运动拖影。

## 7. EMVD 为什么不显式估计光流

很多视频复原方法会用光流或 deformable convolution 对齐多帧，比如 DVDnet 用光流，EDVR 用 PCD，BasicVSR 用 SPyNet。EMVD 为了端侧效率，没有走这条路线。

它的做法是：

不显式估计 dense optical flow
而是在低分辨率变换域中预测融合权重
让融合权重自动决定当前帧和历史帧的使用比例

论文也提到，fusion 在较低分辨率上执行，一个变换域位置对应原图中更大的邻域，因此本身具备一定程度的运动补偿能力；此外，融合也可以扩展成空间自适应卷积形式。

这就是 EMVD 的工程取舍：

不用光流：
速度快，内存低，部署简单

代价：
大运动、遮挡、复杂运动下，对齐能力不如光流/Transformer/Deformable 模型

所以 EMVD 适合端侧实时，而不是追求离线最强画质。

## 8. 核心设计三：Spatial Denoising 空域去噪

Temporal Fusion 已经通过多帧递归融合降低了噪声，但它不能完全去掉噪声。原因包括：

1. 初始帧没有历史信息
2. 动态区域不能强融合
3. 运动复杂时历史信息不能充分利用
4. 低信噪比区域仍然残留噪声

所以 EMVD 第二阶段使用一个 denoising network，论文中叫 DCNN，对 fused image 继续做空域去噪。DCNN 的输入包括：

1. 融合后的图像 ȳ_t
2. 当前 noisy frame 的低频部分 z_LL|t
3. 融合图像的噪声方差 σ̄_t²

论文解释说，融合图已经比原始 noisy frame 更容易去噪，但仍需要 DCNN 处理残余噪声；同时把当前 noisy frame 的低频部分输入给 DCNN，是为了让网络还能从未被融合破坏的原始当前帧里提取有价值信息。

这个设计很有意思。它不是盲目相信融合结果，而是让网络同时看：

融合图：噪声少，但可能平滑或运动区域不准
当前帧低频：真实当前内容，但噪声更大
噪声方差：告诉网络该去多强

这比普通单帧去噪更有针对性。

## 9. 噪声方差为什么重要

EMVD 明确把噪声方差输入网络。这一点非常工程化。

RAW 噪声不是固定 σ 的高斯噪声，而是和亮度、ISO、传感器读出噪声相关。亮区域和暗区域噪声分布不同，不同 ISO 下噪声强度也不同。EMVD 使用 signal-dependent noise model，并在融合过程中递归更新 fused image 的噪声方差。论文指出，fusion 是线性的，因此可以根据统计性质递归计算融合后噪声方差，而且随着融合进行，方差会逐步降低。

这对模型很重要：

噪声大：
网络强去噪

噪声小：
网络弱去噪，保细节

融合多帧后：
噪声方差降低，后续去噪强度也应该降低

论文消融结果也显示，去掉方差输入会导致明显性能下降；表中无 variance 输入时 PSNR 从 42.63 降到 41.39。

## 10. 核心设计四：Spatio-Temporal Refinement 时空精修

第三阶段是 Refinement，它解决的是去噪中的经典矛盾：

去噪越强 → 越干净，但细节容易被抹掉
保细节越多 → 越锐利，但噪声容易残留

EMVD 的思路是：

fused image：
细节多，但还有噪声

denoised image：
噪声少，但可能过平滑

refinement：
自适应融合两者

论文中的 refinement 公式也是凸组合：

最终输出 = fused image × fused 权重 + denoised image × denoised 权重

这些权重由 RCNN 预测。论文解释说，refinement weights 和 fusion weights 含义不同：fusion weights 用于聚合时间上一致的信息来降低噪声，而 refinement weights 用于从 fused image 中识别高频纹理和边缘信息，用来补回 denoising 阶段可能损失的细节。

直观理解：

平坦区域：
更多使用 denoised image
保证干净

纹理/边缘区域：
适当从 fused image 中取回高频
避免过平滑

这就是 EMVD 的“干净”和“细节”之间的平衡机制。

## 11. EMVD 的三个 CNN：FCNN、DCNN、RCNN

EMVD 里面不是一个大 U-Net，而是三个很小的 CNN，各自职责非常明确：

模块	名称	输入	输出	作用
Temporal Fusion	FCNN	当前帧与历史融合图的低频差异、噪声方差	融合权重	决定当前帧和历史帧怎么融合
Spatial Denoising	DCNN	fused image、当前帧低频、噪声方差	denoised image	去除融合后残余噪声
Refinement	RCNN	denoised image、fused image、噪声方差	精修权重	找回高频细节

论文消融中说明，出于效率考虑，EMVD 的基础配置中每个 CNN 只使用两层 3×3 卷积、16 个 filters、ReLU，再加一个输出卷积；这个轻量配置对应约 5.38 GFLOPs。

这也是 EMVD 工程友好的原因：它不是一个庞大的 end-to-end 黑盒网络，而是由几个职责明确的小模块组成。

## 12. EMVD 的训练方式

EMVD 是 recurrent 模型，所以训练时要按时间展开。论文训练时从视频中随机裁剪 128×128 的时空 patch，并特别注意保持 Bayer CFA pattern；EMVD 使用 n=25 的序列长度训练，而 RViDeNet 使用 3 帧，FastDVDnet 和 EDVR 使用 5 帧。论文解释，EMVD 是 recurrent model，因此受益于更长序列，因为可以通过时间反向传播学习长期融合行为。

损失函数包括：

L = Lr + Lc + Lf

其中：

Lr：最终输出和 GT 的 L1 损失
Lc：约束 color transform 可逆
Lf：约束 frequency transform 可逆

论文强调，EMVD 不需要给 fusion、denoising、refinement 三个中间阶段额外加监督，只用最终输出 loss 加变换可逆性约束就能收敛到期望行为。

这点很值得面试讲：

EMVD 的结构具有可解释性，但训练上仍然是端到端的，不需要手工给每个阶段设计复杂监督。

## 13. EMVD 为什么适合工程落地

EMVD 的工程价值主要有四点。

13.1 低计算量

EMVD 基础版本约 5.38 GFLOPs，而论文图中同样 RAW 视频去噪任务里，RViDeNet 复杂度远高于 EMVD；论文还展示 EMVD 在低复杂度约束下可以超过其他方法，并且在某些设置下相较 EDVR、RViDeNet 有数百倍复杂度优势。

13.2 低内存

论文在 Huawei P40 Pro 上报告，FastDVDnet 需要 724MB DDR，而 EMVD 需要 112MB DDR，内存占用明显更低。

13.3 流式处理

EMVD 是 recurrent 模型，不需要等待未来帧，也不需要保存一大段视频窗口。每来一帧就可以处理一帧，天然适合：

手机相机预览
实时视频流
监控摄像头
车载相机
工业相机
低延迟视频增强
13.4 可解释性强

EMVD 的每一阶段都有明确含义：

Fusion：用时间冗余降噪
Denoising：进一步清理残余噪声
Refinement：补回高频细节
Transform：让颜色和频率更容易处理

这比“一个大网络直接输入输出”的模型更容易调参、分析和部署。

## 14. EMVD 的局限

EMVD 很适合端侧实时，但它不是万能的。

14.1 对大运动和遮挡不如显式对齐模型

EMVD 没有显式光流、deformable alignment 或 Transformer attention。它主要通过低分辨率融合权重处理运动。因此在大运动、强遮挡、复杂非刚体运动时，它的跨帧利用能力不如 EDVR、BasicVSR++、RVRT 这类模型。

14.2 更偏 RAW 视频去噪

EMVD 的设计和噪声模型都非常贴近 RAW 域。如果直接拿它处理普通 RGB 压缩视频，不一定最合适。RGB 压缩视频里有 JPEG/H.264/H.265 压缩伪影、锐化伪影、色彩映射误差，这和 RAW 噪声不是一个问题。

14.3 依赖噪声参数估计

EMVD 把噪声方差作为重要输入。真实相机工程里可以从 ISO、sensor calibration、noise model 估计这些参数；但在未知来源视频中，准确估计噪声参数并不简单。

14.4 画质上限不一定比大模型高

在服务器离线高质量视频修复中，RVRT、VRT、BasicVSR++ 这类大模型通常有更强表达能力。EMVD 的优势是效率和端侧部署，而不是无限制算力下的极致画质。

## 15. EMVD 和其他视频去噪模型怎么选
场景	更适合的模型
手机 RAW 视频实时去噪	EMVD / EMVD 改进版
端侧低算力视频去噪	EMVD / FastDVDnet
普通 RGB 高斯噪声视频	FastDVDnet
高质量离线视频去噪	RVRT / VRT
真实 RAW 低光视频	EMVD / RViDeNet / Real-LLRVD
监控低照度实时增强	EMVD 思路 + ISP pipeline
无干净 GT 场景	UDVD / 自监督视频去噪
追求工程简单	FastDVDnet
追求端侧实时和可解释性	EMVD

可以这样理解：

FastDVDnet：
适合做 RGB 视频去噪 baseline，简单直接

EMVD：
适合 RAW 域、端侧、实时、低内存

RVRT / VRT：
适合高质量离线，算力充足

RViDeNet：
适合 RAW 视频高质量去噪，但计算更重

## 16. 面试里怎么讲 EMVD

EMVD 是一个面向端侧实时 RAW 视频去噪的高效 recurrent 模型。它的核心不是堆一个很大的网络，而是把视频去噪拆成三个可解释阶段：时域融合、空域去噪和时空精修。首先，它通过可学习的可逆颜色和频率变换，把 RAW 数据映射到更适合去噪的变换域，分离颜色和频率信息。然后在 temporal fusion 阶段，它递归地融合当前 noisy frame 和上一时刻的 fused frame，通过 FCNN 预测自适应融合权重：静态区域更多利用历史融合结果来降低噪声，运动区域更多保留当前帧来避免拖影。接着 spatial denoising 阶段用 DCNN 去除融合后残余噪声，并结合噪声方差控制去噪强度。最后 refinement 阶段用 RCNN 在 fused image 和 denoised image 之间自适应选择，平坦区域更多使用干净的 denoised image，边缘纹理区域从 fused image 中取回高频细节。EMVD 的优点是计算量低、内存小、可以流式处理，适合手机相机和端侧 RAW 视频去噪；缺点是没有显式光流或注意力对齐，大运动和复杂遮挡下不如 RVRT、EDVR、BasicVSR++ 这类更重的视频复原模型。

最后记住这一句：

EMVD 的本质是：用可学习变换域降低处理复杂度，用 recurrent temporal fusion 累积多帧降噪能力，再用轻量 denoising 和 refinement 在“干净”和“细节”之间做平衡。