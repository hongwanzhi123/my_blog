# RViDeNet

RViDeNet 全名可以理解为 Raw Video Denoising Network，来自 CVPR 2020 论文 “Supervised Raw Video Denoising With a Benchmark Dataset on Dynamic Scenes”。它是 RAW 视频去噪方向的经典方法之一，主要解决的是：真实低照度 RAW 视频中噪声很强，而且动态场景很难获得 noisy-clean 配对数据，如何做监督式视频去噪？ 论文同时提出了 CRVD 数据集 和 RViDeNet 网络。官方也开源了实现。

## 1. RViDeNet 解决的是什么问题

普通视频去噪模型，比如 DVDnet、FastDVDnet，很多是针对 sRGB 视频 + 合成高斯噪声 设计的。但真实低照度视频里，噪声通常来自传感器 RAW 数据，和简单高斯噪声差别很大。

真实低照度 RAW 视频的问题包括：

1. 高 ISO 带来强烈 shot noise 和 read noise
2. Bayer RAW 具有 RGGB 马赛克排列
3. 噪声在进入 ISP 前更原始，但也更适合建模
4. 动态场景很难获得干净 GT
5. 多帧之间存在运动，需要时序对齐和融合

RViDeNet 的目标不是处理普通压缩 RGB 视频，而是处理 RAW 域真实视频噪声。论文也强调，RAW 域包含更多原始视觉信息，噪声相比 sRGB 域更简单；很多图像复原工作已经证明直接在 RAW 域处理通常比在 sRGB 后处理更有优势。

一句话理解：

RViDeNet 是一个面向真实动态 RAW 视频的监督式多帧去噪网络，它利用 Bayer RAW 的通道结构、空间相关性、通道相关性和时间相关性来去除真实低光噪声。

## 2. 为什么 RAW 视频去噪比 sRGB 视频去噪更适合低光场景

普通相机成像流程大致是：

传感器 RAW
    ↓
黑电平校正 / 白平衡
    ↓
Demosaic 去马赛克
    ↓
颜色校正
    ↓
降噪 / 锐化 / Gamma
    ↓
压缩编码
    ↓
sRGB / 视频

如果等到 sRGB 阶段再去噪，噪声已经经过 ISP 变换，可能被锐化、插值、颜色变换和压缩放大，变得更复杂。

RAW 域的优势是：

1. 噪声更接近传感器原始噪声
2. 没有经过 ISP 复杂非线性处理
3. 保留了更多原始亮度和颜色信息
4. 可以在去噪后再接不同 ISP 得到用户想要的 sRGB 结果

RViDeNet 的论文也说，它不仅输出 RAW 去噪结果，还可以通过 ISP 模块输出 sRGB 结果，这样用户可以根据自己的 ISP 偏好生成最终图像。

所以，RViDeNet 更适合这类场景：

手机夜景 RAW 视频
安防低照度 RAW 视频
工业相机低光 RAW 视频
相机 ISP 前端视频去噪
真实传感器噪声建模

而不是普通 MP4 / H.264 压缩视频后处理。

## 3. RViDeNet 的数据贡献：CRVD 数据集

RViDeNet 论文很重要的一点是提出了 CRVD：Captured Raw Video Denoising Dataset。

真实 RAW 视频去噪最大的难点是：动态场景没有干净 GT。

静态图像可以这样做：

同一个静态场景拍很多张 noisy 图
    ↓
多张平均
    ↓
得到近似 clean 图像

但动态视频不行。因为物体在动，如果直接多帧平均，会产生运动模糊。

RViDeNet 的数据采集思路很巧妙：他们使用可控运动物体，比如玩具。每一个“运动时刻”先让物体保持静止，连续拍多张 noisy RAW，通过平均得到该时刻的 clean frame；然后移动物体到下一个位置，再重复这个过程。最后按照运动顺序把这些单帧拼成动态 noisy-clean 视频对。论文说明，他们构建了 55 组 noisy-clean 视频，ISO 范围从 1600 到 25600。

可以理解成：

真实动态视频 GT 难采集
        ↓
人为控制物体运动
        ↓
每个时刻都静止拍多张 noisy frame
        ↓
平均得到该时刻 clean frame
        ↓
把多个时刻按顺序组成 clean video

这个数据集的价值很大，因为它第一次让 RAW 动态视频去噪可以做监督训练和定量评估。论文中也明确说，这样的数据集既能支持监督训练，也能支持真实 noisy 视频的量化评估。

## 4. RViDeNet 的整体框架

RViDeNet 输入的是连续 RAW 帧，论文中使用 3 帧输入恢复中间帧：

输入：
I_{t-1}, I_t, I_{t+1}

目标：
恢复中间帧 I_t 的干净 RAW

整体结构可以概括为：

连续 RAW noisy frames
        ↓
Pre-denoising 预去噪
        ↓
Packing：按照 Bayer pattern 拆成 RGBG 四个子序列
        ↓
Alignment：可变形卷积对齐邻近帧特征
        ↓
Non-local Attention：空间 / 通道 / 时间注意力增强特征
        ↓
Temporal Fusion：根据相似性融合多帧特征
        ↓
Spatial Fusion：融合 RGBG 子序列，重建干净 RAW
        ↓
ISP Module：可选输出 sRGB 结果

论文把 RViDeNet 的贡献总结为：把 RAW 序列拆成 RGBG 子序列，然后分别经过 pre-denoising、alignment、non-local attention、temporal fusion，最后通过 spatial fusion 重建无噪声 RAW，并可以通过 ISP 得到 sRGB 输出。

## 5. 模块一：Pre-denoising 预去噪

RViDeNet 的第一步是 预去噪。

为什么要先预去噪？

因为后面要做帧间对齐，而真实低光 RAW 噪声很强。如果直接在强噪声图上估计帧间对应关系，噪声会严重干扰对齐。

论文中也明确说，噪声会严重干扰 dense correspondence 的预测，而设计良好的 pre-denoising module 可以帮助估计对应关系。因此他们先训练一个单帧 U-Net 作为预去噪模块。

可以理解为：

强噪声 RAW
    ↓
单帧 U-Net 初步去噪
    ↓
得到更稳定的特征
    ↓
再做多帧对齐和融合

这个设计和 DVDnet 有点相似：DVDnet 也是先 spatial denoising，再做 motion compensation；RViDeNet 则是在 RAW 域里先预去噪，再进行 deformable alignment。

## 6. 模块二：Packing：把 Bayer RAW 拆成 RGBG 四个子序列

RAW 图像不是普通 RGB 三通道图，而是 Bayer 马赛克排列。常见 Bayer pattern 是：

R G
G B

也就是说，RAW 图中相邻像素对应不同颜色滤波器。直接把 RAW 当成单通道图像处理，会遇到一个问题：

空间上相邻的像素并不一定是同一种颜色通道

所以 RViDeNet 将 RAW 帧按照 Bayer pattern 拆成四个子帧，也就是：

R 子序列
G1 子序列
G2 子序列
B 子序列

论文中把它称为 RGBG sequences。这样每个子序列中的像素都来自同一种颜色滤波器，更适合进行空间卷积和时序对齐。论文明确说明，由于 noisy input 具有 Bayer pattern 特征，因此将其拆成四个 separated sequences，即 RGBG sequences。

直观理解：

原始 Bayer RAW：
R G R G ...
G B G B ...
R G R G ...
G B G B ...

Packing 后：
R 通道：只取 R 位置
G1 通道：只取第一类 G 位置
G2 通道：只取第二类 G 位置
B 通道：只取 B 位置

好处是：

1. 避免不同颜色滤波器像素混在一起
2. 让同色像素在空间上更规整
3. 方便后续卷积学习同通道结构
4. 保留 RAW 域信息，不提前 demosaic

这点是 RViDeNet 和普通 RGB 视频去噪模型最大的区别之一。

## 7. 模块三：Alignment：可变形卷积对齐

视频去噪必须利用相邻帧，但相邻帧之间有运动。

如果直接融合：

I_{t-1}, I_t, I_{t+1}

会出现错位、重影、细节模糊。所以 RViDeNet 需要把邻近帧对齐到中心帧。

RViDeNet 没有使用显式光流，而是使用 deformable convolution，可变形卷积 对齐输入帧。论文中说明，它借鉴视频复原工作，使用 deformable convolutions 对齐输入帧，而不是使用显式 flow 信息。

为什么用可变形卷积？

普通卷积采样位置固定：

固定 3×3 邻域

可变形卷积会学习偏移：

根据运动和内容，动态决定从哪里采样

所以它可以在特征层完成隐式对齐：

邻近帧特征
    ↓
预测 offset
    ↓
可变形卷积采样
    ↓
对齐到中心帧

相比光流，它的优点是：

1. 不需要显式估计 dense optical flow
2. 可以在特征层处理噪声和局部错位
3. 对真实 RAW 噪声干扰可能更鲁棒

缺点是：

1. DCN 部署比普通卷积麻烦
2. 大运动下仍可能对齐不足
3. ONNX / TensorRT 可能需要自定义插件

## 7. 模块三：Alignment：可变形卷积对齐

视频去噪必须利用相邻帧，但相邻帧之间有运动。

如果直接融合：

I_{t-1}, I_t, I_{t+1}

会出现错位、重影、细节模糊。所以 RViDeNet 需要把邻近帧对齐到中心帧。

RViDeNet 没有使用显式光流，而是使用 deformable convolution，可变形卷积 对齐输入帧。论文中说明，它借鉴视频复原工作，使用 deformable convolutions 对齐输入帧，而不是使用显式 flow 信息。

为什么用可变形卷积？

普通卷积采样位置固定：

固定 3×3 邻域

可变形卷积会学习偏移：

根据运动和内容，动态决定从哪里采样

所以它可以在特征层完成隐式对齐：

邻近帧特征
    ↓
预测 offset
    ↓
可变形卷积采样
    ↓
对齐到中心帧

相比光流，它的优点是：

1. 不需要显式估计 dense optical flow
2. 可以在特征层处理噪声和局部错位
3. 对真实 RAW 噪声干扰可能更鲁棒

缺点是：

1. DCN 部署比普通卷积麻烦
2. 大运动下仍可能对齐不足
3. ONNX / TensorRT 可能需要自定义插件

## 8. 模块四：Non-local Attention：空间、通道、时间三类相关性

RViDeNet 不只是对齐和拼接，它还专门设计了 non-local attention 来利用长程相关性。

论文中说，由于 3D non-local attention 计算成本很高，所以他们使用 separated attention modules，也就是把注意力拆成：

1. Spatial Attention 空间注意力
2. Channel Attention 通道注意力
3. Temporal Attention 时间注意力

然后把这三类增强后的特征融合起来，并通过残差连接保留原始输入。为了降低空间注意力的计算和显存，论文使用 criss-cross attention 实现 spatial attention。

这三个注意力分别解决不同问题。

8.1 Spatial Attention：空间相关性

空间注意力关注：

同一帧内部，哪些空间位置彼此相关？

比如 RAW 图像中相似纹理、重复结构、边缘区域，可以通过空间注意力互相参考。

作用是：

增强非局部空间结构
帮助恢复纹理
抑制随机噪声
8.2 Channel Attention：通道相关性

RViDeNet 处理的是 RGBG 子序列。不同颜色通道之间有相关性，例如亮度结构在 R/G/B 中往往对应相似边缘。

通道注意力关注：

哪些颜色/特征通道更重要？
哪些通道之间可以互相补充？

作用是：

利用 RAW 通道相关性
增强颜色一致性
降低某些通道的噪声干扰
8.3 Temporal Attention：时间相关性

时间注意力关注：

邻近帧中哪些时间位置的信息更可靠？

因为即使用 DCN 对齐了，仍然可能存在遮挡、对齐错误、运动区域不可靠。时间注意力可以让网络判断哪些邻近帧信息应该更多使用。

作用是：

稳定时序信息
降低错位帧影响
提高视频时间一致性

这三个注意力对应 RViDeNet 名字里的核心思想：充分利用 spatial、channel、temporal correlations

## 9. 模块五：Temporal Fusion 时间融合

即使邻近帧已经对齐，也不能简单平均。

原因是：

1. 有些区域被遮挡
2. 有些区域对齐不准
3. 有些邻近帧噪声更强
4. 有些运动区域不适合强融合

所以 RViDeNet 设计了 temporal fusion。论文中说，尽管邻近帧特征已经和中心帧对齐，但由于遮挡和对齐误差，它们对中心帧去噪的贡献仍然不同。RViDeNet 会计算邻近帧和中心帧之间的相似性，并用相似性对邻近特征加权，再通过卷积聚合。

直观理解：

对齐后的邻近帧特征
        ↓
和中心帧特征计算相似性
        ↓
相似区域权重大
不相似区域权重小
        ↓
多帧融合

这和很多视频复原方法的思想一致：

不是所有参考帧都可靠，模型要学会选择性融合。

## 10. 模块六：Spatial Fusion 空间融合

前面 RViDeNet 是把 RAW 拆成 RGBG 四个子序列分别处理的。

但最终还要恢复完整 RAW 帧，所以需要把四个子序列融合回去。

这个步骤就是 spatial fusion：

R 子序列特征
G1 子序列特征
G2 子序列特征
B 子序列特征
        ↓
空间融合
        ↓
重建 clean RAW frame

这个模块的作用是：

1. 汇总四个 Bayer 子通道的信息
2. 重建完整 RAW 去噪结果
3. 保持 Bayer RAW 结构

之后可以选择输出 RAW，也可以接 ISP 模块输出 sRGB。论文强调，RViDeNet 既可以输出 RAW 去噪结果，也可以通过 ISP 输出 sRGB，使用户可以根据偏好的 ISP 生成最终结果。

## 11. RViDeNet 的完整推理流程

假设输入三帧 RAW：

I_{t-1}^raw, I_t^raw, I_{t+1}^raw

完整流程可以写成：

1. 对每帧做预去噪
   ↓
2. 按 Bayer pattern pack 成 R/G1/G2/B 四个子序列
   ↓
3. 每个子序列分别提取特征
   ↓
4. 使用 deformable convolution 对齐邻近帧到中心帧
   ↓
5. 使用 spatial/channel/temporal non-local attention 增强特征
   ↓
6. 通过 temporal fusion 融合三帧特征
   ↓
7. 通过 spatial fusion 融合 RGBG 子序列
   ↓
8. 输出 clean RAW
   ↓
9. 可选：通过 ISP 输出 sRGB

一句话：

RViDeNet 先尊重 RAW Bayer 结构，把 RAW 拆成同色子序列；再利用 DCN 做时序对齐，用非局部注意力建模空间、通道和时间相关性；最后融合回完整 RAW 并输出去噪结果。

## 12. RViDeNet 和 EMVD 的区别

你刚刚问过 EMVD，所以这里重点对比。

对比项	RViDeNet	EMVD
任务	RAW 视频去噪	RAW 视频去噪
论文时间	CVPR 2020	CVPR 2021
核心思路	多帧对齐 + 非局部注意力 + RGBG 分支融合	递归时域融合 + 空域去噪 + 时空精修
输入方式	通常 3 帧恢复中间帧	流式 recurrent，当前帧 + 历史状态
对齐方式	可变形卷积对齐	不显式对齐，用融合权重控制历史/当前
计算复杂度	相对较高	明显更轻量
工程定位	高质量 RAW 视频去噪 baseline	端侧实时 RAW 视频去噪
部署难点	DCN、注意力、RAW pipeline	噪声参数、RAW pipeline，但模型较轻

简单说：

RViDeNet：
更像“高质量 RAW 视频去噪网络”，结构较完整，利用对齐和注意力。

EMVD：
更像“端侧实时 RAW 视频去噪系统”，结构轻量，可流式处理。

如果你做研究汇报，RViDeNet 很值得讲；如果你做移动端部署，EMVD 更贴近工程落地。

## 3. RViDeNet 和 FastDVDnet 的区别
对比项	FastDVDnet	RViDeNet
输入域	sRGB / 普通视频帧	RAW Bayer 视频
噪声类型	主要 AWGN 合成噪声	真实 RAW 低光噪声
是否显式对齐	无显式光流	可变形卷积对齐
是否考虑 Bayer	不考虑	专门 pack 成 RGBG 子序列
输出	去噪中心帧	clean RAW + 可选 sRGB
适合场景	普通 RGB 视频高斯去噪 baseline	相机 RAW 低照度视频去噪

FastDVDnet 更适合普通视频去噪 demo；RViDeNet 更适合讲真实传感器噪声和 RAW 域低光视频增强。

## 14. RViDeNet 的优点
14.1 面向真实 RAW 噪声

它不是只在合成高斯噪声上训练，而是构建了真实 noisy-clean RAW 视频数据集，并在真实低照度 RAW 场景中验证。论文也说明，传统视频去噪方法通常针对高斯或合成噪声，没有考虑低照度采集条件下的复杂真实噪声。

14.2 充分利用 RAW Bayer 结构

它没有直接把 RAW 当普通图像处理，而是根据 Bayer pattern 拆成 RGBG 子序列，这更符合 RAW 数据物理结构。

14.3 利用多帧时序信息

通过对齐、时间注意力和时间融合，它能利用邻近帧降低随机噪声，同时提高时间一致性。

14.4 同时输出 RAW 和 sRGB

这对工程很有价值。RAW 输出可以进入后续 ISP，sRGB 输出可以直接显示。论文明确强调，这给不同用户的 ISP 偏好提供了灵活性。

## 15. RViDeNet 的局限
15.1 工程部署比 EMVD 更复杂

RViDeNet 包含：

U-Net 预去噪
Bayer packing
可变形卷积对齐
non-local attention
temporal fusion
spatial fusion
ISP

这比轻量 CNN 或 EMVD 更复杂，端侧部署难度更高。

15.2 依赖 RAW 输入

如果你只有普通 MP4 视频，已经是 sRGB 或 YUV 压缩格式，那么 RViDeNet 的 RAW 域优势发挥不出来。它更适合接入相机 RAW pipeline，而不是普通视频文件后处理。

15.3 数据采集方式有局限

CRVD 的动态数据是通过可控物体运动采集的。它解决了真实 noisy-clean 动态视频难配对的问题，但场景复杂度、真实运动模式和真实夜间环境仍然有限。后来的 Real-LLRVD、RViDeformer 等工作也是在继续补这个方向。RViDeformer 官方仓库也说明它面向更大 RAW 视频去噪基准。

15.4 可变形卷积和注意力模块不够轻量

如果追求移动端实时，RViDeNet 不是最优选择。EMVD、BRVE 这类方法更偏低算力部署。

## 16. 工程上怎么用 RViDeNet

如果要在真实工程里用 RViDeNet，需要完整 RAW pipeline：

相机采集 RAW Bayer 视频
        ↓
黑电平校正 / 白平衡相关预处理
        ↓
RViDeNet RAW 去噪
        ↓
输出 clean RAW
        ↓
ISP：
    demosaic
    color correction
    tone mapping
    gamma
        ↓
输出 sRGB / YUV 视频

关键注意点：

1. 输入必须是 RAW Bayer，而不是普通 RGB 视频
2. 不同相机传感器噪声分布不同，需要重新标定或微调
3. ISO、曝光、黑电平、白平衡都会影响效果
4. 如果要部署到端侧，需要处理 DCN 和注意力模块的推理优化
5. sRGB 结果高度依赖后续 ISP 质量

所以 RViDeNet 更适合：

相机厂商 / ISP 研发
RAW 视频去噪研究
低照度传感器视频增强
高质量离线 RAW 视频去噪

不太适合：

普通用户 MP4 视频增强
视频会议实时降噪
移动端低算力实时处理

## 17. 面试里怎么讲 RViDeNet


RViDeNet 是一个面向真实 RAW 视频去噪的监督式网络。它首先解决了动态 RAW 视频没有 noisy-clean 配对数据的问题：作者通过控制玩具等物体运动，让每个运动时刻短暂停止并连续拍摄多张高 ISO noisy RAW，通过多帧平均得到该时刻的 clean frame，再把这些时刻按顺序组成动态视频，从而构建 CRVD 数据集。网络方面，RViDeNet 充分利用 RAW Bayer 数据的结构特点，把输入 RAW 按照 Bayer pattern 拆成 R、G1、G2、B 四个子序列。每个子序列先经过预去噪，降低真实噪声对后续对齐的干扰；然后使用可变形卷积把邻近帧特征对齐到中心帧；接着通过空间、通道和时间 non-local attention 建模长程相关性；再根据邻近帧与中心帧的相似性做 temporal fusion，选择性融合多帧信息；最后通过 spatial fusion 把 RGBG 子序列融合成完整 clean RAW，并可通过 ISP 输出 sRGB。它的优势是针对真实 RAW 噪声和 Bayer 结构设计，去噪效果和时间一致性较好；缺点是依赖 RAW 输入，模型结构较复杂，端侧实时部署不如 EMVD 轻量。

最后记住这一句：

RViDeNet 的本质是：在 RAW Bayer 域中，把视频拆成 RGBG 子序列，通过预去噪、可变形对齐、非局部注意力和时序融合，充分利用空间、通道和时间相关性来去除真实低光视频噪声。