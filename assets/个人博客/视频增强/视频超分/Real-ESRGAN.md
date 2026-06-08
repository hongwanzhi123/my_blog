# Real-ESRGAN

Real-ESRGAN 可以理解为：把 ESRGAN 从“标准图像超分”推进到“真实世界图像/视频帧增强”的工程化方法。它的完整论文名是 Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data。核心目标不是在 bicubic 下采样这种干净退化上刷 PSNR，而是处理真实图片里常见的复杂退化，比如模糊、噪声、JPEG 压缩、振铃伪影、过冲伪影等。论文明确说 Real-ESRGAN 扩展自 ESRGAN，并且只使用合成数据训练，通过高阶退化建模模拟真实世界退化.

## 1. Real-ESRGAN 解决的是什么问题

传统超分通常假设低清图是高清图经过某种固定退化得到的，例如：

HR 图像
  ↓ bicubic downsampling
LR 图像

这种设定太理想了。真实图片通常不是单纯 bicubic 下采样，而是经历了复杂退化：

真实图像退化可能包括：
模糊
噪声
JPEG 压缩
相机处理
多次缩放
多次转码
锐化过度
振铃伪影
过冲伪影

所以普通 ESRGAN 在真实图片上经常会出现两个问题：

一是把噪声当成纹理增强，导致画面脏；

二是生成过强高频纹理，导致假细节、边缘伪影。

Real-ESRGAN 的目标就是解决这种 blind super-resolution 问题，也就是：不知道真实退化过程是什么，但仍然希望把图像恢复得清晰、自然、少伪影。论文摘要中也明确说，以前很多 blind SR 方法仍难以处理 general real-world degraded images，因此 Real-ESRGAN 使用更真实的合成退化过程来训练模型。

## 2. Real-ESRGAN 和 ESRGAN 的关系

Real-ESRGAN 不是完全推翻 ESRGAN，而是在 ESRGAN 基础上做真实场景增强。

ESRGAN 的核心包括：

生成器：RRDBNet
判别器：GAN discriminator
损失：pixel loss + perceptual loss + adversarial loss
目标：生成感知质量更好的超分图像

Real-ESRGAN 继承了 ESRGAN 的生成器思想，仍然使用 RRDBNet 作为主要生成网络，但重点改进了两个地方：

第一，训练数据退化方式更真实；

第二，判别器更强、更稳定。

官方仓库也说明 Real-ESRGAN 是将 ESRGAN 扩展到 practical restoration application，并使用纯合成数据训练。

一句话：

ESRGAN 更像“感知质量导向的图像超分”；Real-ESRGAN 更像“面向真实退化图像的实用增强模型”。

## 3. Real-ESRGAN 的整体流程

Real-ESRGAN 的训练流程可以理解成：

高清真实图像 HR
        ↓
高阶退化模型合成低清图像 LR
        ↓
生成器 G：LR → SR
        ↓
和 HR 比较
        ↓
使用 pixel / perceptual / GAN loss 训练

推理时更简单：

真实低清图像 / 视频帧
        ↓
Real-ESRGAN 生成器
        ↓
高清增强图像 / 增强帧

Real-ESRGAN 的关键不只是网络结构，而是 退化建模。因为真实世界没有成对的低清/高清数据，所以论文使用纯合成数据训练，通过复杂退化过程模拟真实低清图像。

## 4. 核心一：High-order Degradation 高阶退化模型

Real-ESRGAN 最重要的创新之一是 high-order degradation modeling process。

普通超分退化一般是一阶的：

HR → blur → downsample → noise → JPEG → LR

但真实图片往往经历多次处理，比如：

原图
  ↓ 拍摄模糊
  ↓ 相机降噪/锐化
  ↓ 社交平台压缩
  ↓ 用户截图
  ↓ 再次压缩
  ↓ 再次缩放

所以 Real-ESRGAN 用 多阶段退化 来模拟真实情况。论文明确提出 high-order degradation modeling，用来更好模拟复杂真实退化。

可以把它理解成：

第一轮退化：
HR → blur → resize → noise → JPEG

第二轮退化：
再次 blur → resize → noise → JPEG

最后：
sinc filter / resize / JPEG 等组合

这种多轮退化比单轮退化更接近真实图片，因为真实图片经常经历反复压缩、缩放、上传、下载。

## 5. 高阶退化里每一步的作用
5.1 Blur 模糊

真实图像模糊来源很多：

运动模糊；

离焦模糊；

相机镜头模糊；

压缩前的滤波；

平台处理造成的平滑。

所以训练时会随机使用不同 blur kernel，让模型适应不同模糊类型。

5.2 Resize 随机缩放

真实图片可能被放大、缩小、再放大。比如：

原图 1080p
  ↓ 平台压缩到 720p
  ↓ 用户截图
  ↓ 再被 App 拉伸到 1080p

所以 Real-ESRGAN 的退化过程里会随机 upsample、downsample 或保持尺寸。

5.3 Noise 噪声

真实图片中有传感器噪声、压缩噪声、低光照噪声等。退化模型会加入不同噪声，避免模型只适应干净 LR。

5.4 JPEG 压缩

网络图片、短视频截图、社交平台图片基本都经过 JPEG 或视频压缩。JPEG 会产生块效应、蚊噪、边缘振铃等，所以 Real-ESRGAN 的退化模型会加入 JPEG compression。

5.5 Sinc Filter

Real-ESRGAN 还考虑了真实图像里常见的 ringing artifacts 和 overshoot artifacts。论文明确说使用 sinc filters 来建模这类常见伪影。

这点很重要，因为很多真实图片边缘周围会出现一圈波纹，或者强边缘附近出现亮边/黑边。普通退化模型如果不模拟这些现象，模型推理时很容易处理不好。

## 6. 核心二：RRDBNet 生成器

Real-ESRGAN 的生成器主要沿用 ESRGAN 的 RRDBNet。

RRDB 的全称是：

Residual-in-Residual Dense Block

可以理解为：

Dense Block
  + Residual connection
  + 外层 Residual connection

它的特点是：

第一，网络很深，但残差连接让训练更稳定；

第二，dense connection 有助于特征复用；

第三，不使用 BatchNorm，避免图像生成任务中 BN 带来的伪影和不稳定。

生成器整体大致是：

LR 输入
  ↓
浅层卷积
  ↓
多个 RRDB blocks
  ↓
上采样模块
  ↓
卷积输出 SR

Real-ESRGAN 的重点不是换掉 RRDBNet，而是让 RRDBNet 在更真实、更复杂的合成退化上训练，从而适应真实世界输入。官方仓库也列出默认模型 realesrgan-x4plus 和对应的非 GAN 版本 realesrnet-x4plus。

## 7. RealESRNet 和 RealESRGAN 的区别

官方模型里有两个名字很像：

RealESRNet
RealESRGAN

它们的区别可以这样理解：

RealESRNet：
只用重建类损失训练，更偏保真，结果更平滑、更稳定，假纹理少。

RealESRGAN：
在 RealESRNet 基础上引入 GAN 训练，结果更锐利、更有细节，但也更可能产生假纹理。

官方模型列表中同时提供了 realesrgan-x4plus 和 realesrnet-x4plus，前者是常用默认模型，后者可以理解为不使用 GAN 强化纹理的版本。

工程上可以这样选：

想要更自然、更少假细节：RealESRNet
想要更清晰、更锐利：RealESRGAN

## 8. 核心三：U-Net Discriminator with Spectral Normalization

Real-ESRGAN 的另一个关键改进是判别器。

ESRGAN 的判别器更偏整体判断，而 Real-ESRGAN 使用了 U-Net discriminator with spectral normalization。论文明确说使用带 spectral normalization 的 U-Net 判别器，目的是增强判别器能力并稳定训练。

为什么用 U-Net 判别器？

普通判别器更像输出一个整体真假判断：

这张图整体真不真？

U-Net 判别器更像可以输出更细粒度的判断：

这个区域真不真？
这个边缘真不真？
这个纹理真不真？

这对真实图像恢复很有用，因为伪影通常是局部的，比如：

边缘振铃；

局部噪声；

局部假纹理；

局部压缩块。

Spectral Normalization 的作用是稳定 GAN 训练，避免判别器过强导致训练发散。

## 9. Real-ESRGAN 的损失函数

Real-ESRGAN 的训练通常包含三类损失。

9.1 Pixel Loss

Pixel loss 负责保证输出和 GT 在像素层面接近。

它的作用是：

保证结构、颜色、整体内容不要偏离

如果只用 GAN loss，模型可能生成看起来锐利但内容不准确的纹理。Pixel loss 可以约束基本保真度。

9.2 Perceptual Loss

Perceptual loss 通常用 VGG 特征计算，让输出在高层语义和纹理感知上接近真实图像。

它的作用是：

让结果看起来更自然，而不是只追求像素平均
9.3 Adversarial Loss

GAN loss 负责提升真实感和锐利度。

它的作用是：

鼓励生成更像真实高清图像的细节

但是 GAN loss 也带来风险：可能生成假纹理。所以 Real-ESRGAN 的效果通常比 PSNR 模型锐利，但也可能在文字、人脸、规则纹理上产生不真实细节。

## 10. Real-ESRGAN 的训练阶段

Real-ESRGAN 通常可以理解为两阶段训练。

第一阶段训练 RealESRNet：

合成 LR → 生成 SR
使用 pixel / perceptual 等重建损失
输出比较稳

第二阶段训练 RealESRGAN：

在 RealESRNet 基础上加入 GAN loss
使用 U-Net discriminator
输出更锐利

这种训练方式比一开始直接 GAN 训练更稳定。官方模型列表中同时提供 RealESRNet 和 RealESRGAN，也反映了这两类模型的不同用途。

## 11. Real-ESRGAN 为什么适合工程落地

Real-ESRGAN 在工程里很常见，原因不是它理论最新，而是它非常实用。

11.1 不需要真实成对数据

真实世界很难获得严格对齐的 LR-HR 图像对。Real-ESRGAN 使用纯合成数据训练，降低了数据获取难度。论文题目和摘要都强调了 pure synthetic data。

11.2 真实图片效果直观

相比 bicubic SR 模型，Real-ESRGAN 更适合网络图片、老照片、动漫图、视频帧截图等复杂输入。

11.3 工具链成熟

官方仓库提供 PyTorch 推理，也有 ncnn-vulkan 版本，方便在桌面端、无 Python 环境、部分 GPU/CPU 环境下运行。官方仓库明确定位为 general image/video restoration，并提供多个模型。

11.4 视频处理简单

虽然 Real-ESRGAN 本身不是严格的视频时序模型，但可以很容易用于视频：

ffmpeg 拆帧
  ↓
Real-ESRGAN 逐帧增强
  ↓
ffmpeg 合成视频

这使它成为很多视频增强 Demo、动漫修复工具、老视频增强工具的常用选择。

## 12. 官方常见模型

官方仓库列出的常见模型包括：

模型	用途
realesrgan-x4plus	默认通用模型
realesrnet-x4plus	非 GAN 版本，更平滑稳健
realesrgan-x4plus-anime	动漫图像优化，小模型
realesr-animevideov3	动漫视频模型

官方说明中列出了这些模型，并且可以通过 -n 参数选择不同模型。

工程上选择可以简单记：

真实照片 / 普通图片：realesrgan-x4plus
更保守少假纹理：realesrnet-x4plus
动漫图片：realesrgan-x4plus-anime
动漫视频：realesr-animevideov3

## 13. Real-ESRGAN 用于视频时的特点

Real-ESRGAN 可以用于视频，但要注意：它本质上是逐帧模型，不是真正的视频超分模型。

也就是说，它处理视频时通常是：

第 1 帧 → 超分
第 2 帧 → 超分
第 3 帧 → 超分
...

它不会显式利用：

前后帧光流；

时序传播；

跨帧对齐；

时间一致性约束。

所以它的优点是：

部署简单
速度可控
单帧效果强
不需要处理复杂序列输入

缺点是：

可能闪烁
纹理可能帧间跳动
不能利用多帧互补信息
运动区域可能不稳定

这也是 Real-ESRGAN 和 RealBasicVSR 的核心区别：Real-ESRGAN 是真实图像/视频帧增强，RealBasicVSR 是真实视频超分，后者显式利用时序传播。

## 14. Real-ESRGAN 和 RealBasicVSR 对比
对比项	Real-ESRGAN	RealBasicVSR
类型	单图/逐帧增强	视频超分
是否利用时序	不显式利用	利用 BasicVSR 时序传播
输入	单张图像或单帧	视频帧序列
优点	简单、成熟、部署方便	时间一致性更好，可利用多帧信息
缺点	可能闪烁	工程复杂、显存更高
适合	图片、动漫图、快速视频增强 Demo	真实低清视频修复、长视频增强

如果你的目标是快速做一个可展示项目，Real-ESRGAN 最容易跑通；如果你想体现“视频超分算法能力”，RealBasicVSR / BasicVSR++ 更有含金量。

## 15. Real-ESRGAN 的优点
15.1 真实退化适应性强

高阶退化模型让它比普通 ESRGAN 更适合真实图片。论文也强调 Real-ESRGAN 在多种真实数据上比早期方法有更好的视觉表现。

15.2 视觉效果锐利

GAN 和 perceptual loss 让输出更清晰，纹理更强。对于老照片、低清插画、网络图，主观提升明显。

15.3 部署资料多

官方提供模型、代码、ncnn-vulkan 版本和不同应用模型，工程上容易集成。

15.4 适合做快速 Demo

对于面试项目，Real-ESRGAN 可以很快展示：

原图 / 原视频；

增强图 / 增强视频；

局部放大对比；

处理耗时；

部署方式。

## 16. Real-ESRGAN 的局限
16.1 可能产生假细节

GAN 模型常见问题是 hallucination。比如：

文字可能被修成错误字符；

人脸细节可能不真实；

规则纹理可能变形；

细线条可能被强化过度。

所以 Real-ESRGAN 不适合对内容真实性要求极高的场景，比如医学影像、司法取证、精密工业检测等。

16.2 对视频时间一致性不强

逐帧处理视频时，每帧生成的纹理可能略有差异，播放时会出现闪烁。

16.3 对严重退化不一定稳

如果输入太糊、太噪、压缩太重，模型可能无法恢复真实细节，只能生成“看起来像”的纹理。

16.4 不是高 PSNR 模型

Real-ESRGAN 更偏感知质量，不是为了在标准 bicubic benchmark 上拿最高 PSNR。如果面试问指标，你要说明它更关注主观视觉质量，而不是单纯 PSNR。

## 17. 工程部署注意点
17.1 Tile 推理

高分辨率图像直接推理容易爆显存，所以常用 tile：

大图切成小块
每块分别超分
边缘加 overlap
最后拼回去

这样可以降低显存，但如果 overlap 不够，可能出现接缝。

17.2 视频处理流程

常见流程：

ffmpeg 提取帧
  ↓
Real-ESRGAN 批量处理帧
  ↓
ffmpeg 合成视频
  ↓
复制原音频

如果做项目，最好保留音频，并输出对比视频。

17.3 模型选择

真实照片优先 realesrgan-x4plus；

动漫视频优先 realesr-animevideov3；

想减少假纹理可以试 realesrnet-x4plus；

显存小就用 tile 或 ncnn-vulkan 版本。

官方仓库说明可以通过模型名参数选择不同模型。

17.4 ONNX / TensorRT / ncnn

Real-ESRGAN 相比 EDVR、BasicVSR++ 更容易部署，因为它主要是单帧 CNN，没有光流传播和 DCN 对齐这类复杂视频模块。ncnn-vulkan 版本也降低了桌面端部署难度。

## 18. 面试中怎么讲 Real-ESRGAN

Real-ESRGAN 是 ESRGAN 面向真实世界盲超分的改进版本。传统超分通常假设低清图来自 bicubic 下采样，但真实图像往往经历了多次模糊、缩放、噪声、JPEG 压缩、锐化和转码，因此普通 ESRGAN 在真实图像上容易放大噪声或产生伪影。Real-ESRGAN 的核心贡献是提出高阶退化建模，用多阶段 blur、resize、noise、JPEG 等操作模拟真实复杂退化，并且使用 sinc filter 模拟振铃和过冲伪影。同时，它使用带 spectral normalization 的 U-Net 判别器，提高局部细节判别能力并稳定 GAN 训练。生成器仍主要采用 RRDBNet，训练时先得到较稳定的 RealESRNet，再通过感知损失和对抗损失训练 RealESRGAN，使结果更锐利、更符合真实图像感知质量。工程上 Real-ESRGAN 很常用于图片增强和视频逐帧增强，优点是部署简单、效果直观，缺点是可能生成假细节，并且用于视频时缺少显式时间一致性。

最核心的一句话：

Real-ESRGAN 的本质是：用更真实的合成退化训练 ESRGAN，使模型从只会处理理想低清图，变成能处理真实世界复杂退化图像的实用超分模型。