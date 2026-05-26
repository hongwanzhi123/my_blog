# DeepLab

DeepLab 系列语义分割算法。它和 U-Net 一样是图像分割领域非常经典的模型，但两者侧重点不同：U-Net 更强调编码器-解码器和跳跃连接恢复边界细节；DeepLab 更强调空洞卷积、多尺度上下文建模和语义分割精度。

## 一、DeepLab 是什么

DeepLab 是一系列用于语义分割 Semantic Segmentation 的经典深度学习模型。

语义分割任务是：

输入：一张图像
输出：每个像素所属的类别

例如自动驾驶场景中：

道路 / 天空 / 建筑 / 行人 / 车辆 / 树木

DeepLab 的输出不是一个类别，也不是目标框，而是一张和输入图像对应的像素级类别图：

[B, num_classes, H, W]

每个像素位置都有一个类别预测。

DeepLab 系列最核心的贡献包括：空洞卷积 atrous convolution、ASPP 多尺度上下文模块、早期版本中的 DenseCRF 后处理，以及 DeepLabv3+ 中的 Encoder-Decoder 结构和 atrous separable convolution。早期 DeepLab 论文明确提出用空洞卷积控制特征响应分辨率和感受野，用 ASPP 捕获多尺度上下文，并用全连接 CRF 改善边界定位。

## 二、DeepLab 解决了什么问题？

普通 CNN 做图像分类时，会不断下采样：

输入图像：512 × 512
↓
256 × 256
↓
128 × 128
↓
64 × 64
↓
32 × 32

这样有利于图像分类，因为模型可以获得更强的语义信息。

但是分割任务需要输出像素级结果，如果下采样太多，会出现几个问题：

1. 输出特征图分辨率太低
2. 小目标容易丢失
3. 目标边界不清晰
4. 空间定位不准确

例如一个行人的边界、道路边缘、裂缝区域，如果特征图被下采样得太严重，模型很难恢复精细位置。

DeepLab 主要从两个角度解决这个问题：

1. 用空洞卷积扩大感受野，同时尽量保留特征图分辨率
2. 用 ASPP 在多个尺度上提取上下文信息

DeepLabv3 论文也强调，空洞卷积可以显式调整卷积核视野并控制 DCNN 特征响应的分辨率；为了解决多尺度目标分割问题，它设计了串联或并联的空洞卷积模块来捕获多尺度上下文。

## 三、DeepLab 的核心思想

DeepLab 的核心思想可以概括为：

通过空洞卷积在不明显增加参数量的情况下扩大感受野，通过 ASPP 捕获多尺度上下文信息，再结合上采样或解码器恢复像素级分割结果。

其中最重要的两个技术是：

Atrous Convolution / Dilated Convolution：空洞卷积
ASPP：Atrous Spatial Pyramid Pooling，空洞空间金字塔池化

如果只记一句话：

DeepLab = Backbone + Atrous Convolution + ASPP + Segmentation Head

DeepLabv3+ 可以理解为：

DeepLabv3+ = DeepLabv3 Encoder + 简单有效的 Decoder

DeepLabv3+ 论文明确说，它在 DeepLabv3 基础上加入一个 decoder 模块，用来细化分割结果，尤其改善目标边界；同时把 depthwise separable convolution 用到 ASPP 和 decoder 中，以形成更快、更强的 encoder-decoder 网络。

## 四、什么是空洞卷积？

空洞卷积也叫：

Atrous Convolution
Dilated Convolution

它的作用是：

在不增加卷积核参数量的情况下扩大感受野。

普通 3×3 卷积是连续采样：

x x x
x x x
x x x

空洞卷积会在采样点之间插入间隔。

例如 dilation rate = 2：

x . x . x
. . . . .
x . x . x
. . . . .
x . x . x

虽然仍然只有 9 个有效采样点，但覆盖范围变大了。

## 五、空洞卷积为什么重要？

在语义分割中，通常希望同时满足两个要求：

1. 特征图分辨率不能太低，否则边界和小目标会丢失
2. 感受野要足够大，否则模型看不到全局上下文

普通 CNN 如果想扩大感受野，通常会继续下采样：

512 → 256 → 128 → 64 → 32

但这样会损失空间细节。

空洞卷积的好处是：

不需要继续下采样
也能扩大感受野

例如：

普通 3×3 卷积，rate = 1：感受野约 3×3
空洞 3×3 卷积，rate = 2：感受野约 5×5
空洞 3×3 卷积，rate = 4：感受野约 9×9
空洞 3×3 卷积，rate = 8：感受野约 17×17

注意，有效参数还是 3×3，只是采样间隔变大。

所以空洞卷积可以让 DeepLab 在保持较高特征分辨率的同时获取更大上下文。DeepLab 系列论文也把空洞卷积作为 dense prediction 任务中控制特征响应分辨率和扩大卷积视野的重要工具。

## 六、什么是 Output Stride？

理解 DeepLab 时，经常会看到一个概念：

Output Stride

它表示：

输入图像尺寸 / 输出特征图尺寸

例如输入图像是：

512 × 512

如果 backbone 输出特征图是：

32 × 32

那么：

output stride = 512 / 32 = 16

常见 output stride 有：

OS = 32
OS = 16
OS = 8

对于图像分类，OS=32 很常见，因为分类不需要高分辨率输出。

但对于语义分割，OS=32 通常太粗，容易导致边界模糊、小目标丢失。

DeepLab 常用：

output stride = 16
或者
output stride = 8

做法是：

减少后面几层的下采样
用空洞卷积补偿感受野

例如原本 ResNet 后面会继续 stride=2 下采样，DeepLab 可以把 stride 改成 1，然后用 dilation=2 或 dilation=4 的空洞卷积保持感受野。

## 七、什么是 ASPP？

ASPP 全称是：

Atrous Spatial Pyramid Pooling

中文可以叫：

空洞空间金字塔池化

它是 DeepLab 系列最有代表性的模块之一。

ASPP 的核心思想是：

用多个不同 dilation rate 的空洞卷积分支，从不同感受野尺度提取特征，然后把它们融合起来。

比如一个 ASPP 可能包含：

1×1 Conv
3×3 Atrous Conv, rate = 6
3×3 Atrous Conv, rate = 12
3×3 Atrous Conv, rate = 18
Image Pooling

然后把这些分支的输出 concat，再用 1×1 Conv 融合。

结构大致是：

输入特征
 ├── 1×1 Conv
 ├── 3×3 Atrous Conv rate=6
 ├── 3×3 Atrous Conv rate=12
 ├── 3×3 Atrous Conv rate=18
 └── Image-level Pooling
        ↓
     Concat
        ↓
     1×1 Conv
        ↓
     输出特征

ASPP 的作用是捕获多尺度上下文。早期 DeepLab 论文提出 ASPP，用多个采样率和有效感受野的滤波器探测输入特征层，从而分割不同尺度的目标；DeepLabv3 则进一步改进 ASPP，并加入 image-level features 来编码全局上下文

## 八、为什么语义分割需要多尺度上下文？

图像中的目标尺度差异很大。

例如自动驾驶场景：

近处的车：很大
远处的车：很小
近处的人：较大
远处的人：很小
道路：大面积区域
交通灯：小目标

如果只用小感受野，模型能看清局部细节，但理解不了整体。

如果只用大感受野，模型能理解全局，但可能丢失小目标和边界。

ASPP 的设计就是为了解决这个矛盾：

小 dilation rate：关注局部细节
中 dilation rate：关注中等尺度目标
大 dilation rate：关注大范围上下文
image pooling：提供全局语义信息

最后融合多个尺度的信息，让模型同时看到局部和全局。

## 九、DeepLab 的整体结构

以 DeepLabv3+ 为例，整体可以分为：

1. Backbone
2. Atrous Convolution
3. ASPP
4. Decoder
5. Segmentation Head

流程如下：

输入图像
↓
Backbone 提取特征
↓
使用空洞卷积控制 output stride
↓
ASPP 提取多尺度上下文
↓
上采样
↓
与浅层特征融合
↓
Decoder 细化边界
↓
输出语义分割 mask

如果画成结构：

Image
 ↓
Backbone, e.g. ResNet / Xception
 ↓
High-level feature
 ↓
ASPP
 ↓
Upsample
 ↓
Concat with low-level feature
 ↓
Decoder Conv
 ↓
Upsample to original size
 ↓
Pixel-wise prediction

DeepLabv3+ 的关键在于：它把 DeepLabv3 的 ASPP 强上下文建模能力，和 encoder-decoder 的边界恢复能力结合起来。论文中也明确说，spatial pyramid pooling 能编码多尺度上下文，而 encoder-decoder 结构能逐步恢复空间信息并捕获更清晰的边界；DeepLabv3+ 试图结合这两者优势。

## 十、DeepLab 系列版本演进

DeepLab 不是单一模型，而是一个系列。

### 1. DeepLab v1：CNN + DenseCRF

早期 DeepLab 的问题意识是：

DCNN 语义表达强，但输出边界不够精确

因为 CNN 的池化和下采样会提升不变性，但会降低定位精度。

DeepLab v1 的思路是：

CNN 负责语义预测
DenseCRF 负责细化边界

流程大致是：

输入图像
↓
CNN 得到粗分割结果
↓
DenseCRF 根据颜色、位置、边缘信息修正边界
↓
输出更精细 mask

早期 DeepLab 工作指出，DCNN 最后一层响应不够局部化，原因与深层网络的不变性有关；它通过把 DCNN 响应和全连接 CRF 结合，提高了边界定位精度，并在 PASCAL VOC 2012 上取得当时领先结果。

### 2. DeepLab v2：空洞卷积 + ASPP + CRF

DeepLab v2 更系统地引入了：

Atrous Convolution
ASPP
DenseCRF

它的核心提升在于：

用空洞卷积扩大感受野
用 ASPP 处理多尺度目标
用 CRF 改善边界

可以理解为：

DeepLab v2 = CNN + Atrous Conv + ASPP + DenseCRF

这时 DeepLab 的多尺度建模能力已经很强。

### 3. DeepLab v3：重新思考空洞卷积

DeepLabv3 主要改进是：

更系统地研究 atrous convolution
改进 ASPP
加入 image-level feature
不再强依赖 DenseCRF

DeepLabv3 中 ASPP 通常包括多个并行空洞卷积分支，同时加入全局池化分支。

结构大致是：

ASPP:
    1×1 Conv
    3×3 Atrous Conv rate=6
    3×3 Atrous Conv rate=12
    3×3 Atrous Conv rate=18
    Image Pooling

DeepLabv3 论文明确提出，它通过串联或并联不同 atrous rates 的模块捕获多尺度上下文，并在 ASPP 中加入 image-level features 来进一步提升性能；它也指出 DeepLabv3 在没有 DenseCRF 后处理的情况下显著优于前代 DeepLab。

### 4. DeepLab v3+：Encoder-Decoder + Atrous Separable Conv

DeepLabv3+ 是最常见、最常被拿来做项目 baseline 的版本之一。

它在 DeepLabv3 的基础上加入了 decoder。

为什么要加 decoder？

因为 DeepLabv3 的 ASPP 能捕获很强的语义上下文，但输出边界仍然可能不够精细。

DeepLabv3+ 的思路是：

Encoder：Backbone + ASPP，提取高级语义和多尺度上下文
Decoder：融合浅层特征，恢复边界细节

它通常会取 backbone 中较浅层的低级特征，例如：

低级特征：H/4, W/4，边界细节多
高级特征：H/16 或 H/8，语义强

然后：

ASPP 输出高级语义特征
↓
上采样到 H/4
↓
和低级特征 concat
↓
几个 3×3 Conv 细化
↓
上采样到原图大小

DeepLabv3+ 还使用了 atrous separable convolution，也就是把空洞卷积和深度可分离卷积结合起来，以减少计算量并提升效率。论文报告 DeepLabv3+ 在 PASCAL VOC 2012 和 Cityscapes 上分别达到 89.0% 和 82.1% 的测试集性能，且不使用后处理。

## 十一、什么是 Atrous Separable Convolution？

DeepLabv3+ 中经常提到：

Atrous Separable Convolution

它可以理解为：

Atrous Convolution + Depthwise Separable Convolution

普通卷积同时做两件事：

1. 空间特征提取
2. 通道特征融合

Depthwise Separable Convolution 把它拆成两步：

Depthwise Conv：每个通道单独做空间卷积
Pointwise Conv：用 1×1 卷积做通道融合

Atrous Separable Convolution 则是在 depthwise conv 中引入 dilation。

好处是：

扩大感受野
减少参数量
减少计算量
保持较好分割效果

这也是 DeepLabv3+ 能兼顾效果和效率的重要原因之一。

## 十二、DeepLab 和 U-Net 的区别

DeepLab 和 U-Net 都是语义分割常用模型，但设计思想不同。

对比项	U-Net	DeepLab
核心结构	Encoder-Decoder + Skip Connection	Atrous Conv + ASPP
主要优势	边界恢复、小数据、医学/缺陷分割	多尺度上下文、自然场景语义分割
多尺度建模	主要依赖 encoder-decoder 和跳跃连接	ASPP 明确建模多尺度上下文
感受野控制	通过下采样和卷积堆叠	通过 atrous rate 显式控制
边界细节	原生较强	v3+ 通过 decoder 改善
典型任务	医学、细胞、裂缝、缺陷区域	Cityscapes、VOC、场景理解

简单理解：

U-Net 更像是“下采样提语义 + 上采样恢复细节”
DeepLab 更像是“用空洞卷积和 ASPP 强化多尺度语义理解”

如果任务是医学病灶、裂纹、小样本缺陷：

U-Net / U-Net++ / Attention U-Net 通常很合适

如果任务是道路、城市街景、自然场景分割：

DeepLabv3+ 是很经典的选择

当然实际项目中两者都可以试，最终看数据集效果。

## 十三、DeepLab 和 FCN 的区别

FCN 是更早期的语义分割网络。

FCN 的核心思想是：

把分类网络的全连接层改成卷积层
输出空间特征图
再上采样得到分割结果

DeepLab 在 FCN 基础上进一步解决了两个关键问题：

1. 下采样过多导致分辨率低
2. 单尺度上下文不足

DeepLab 的改进是：

用 atrous convolution 保持特征分辨率并扩大感受野
用 ASPP 捕获多尺度上下文
早期版本用 CRF 改善边界
v3+ 用 decoder 恢复边界细节

所以 DeepLab 可以看作是对 FCN 系列语义分割思想的加强版。

## 十四、DeepLab 和 Mask R-CNN 的区别

DeepLab 是语义分割模型。

Mask R-CNN 是实例分割模型。

区别如下：

对比项	DeepLab	Mask R-CNN
任务	语义分割	实例分割
是否区分同类不同个体	不区分	区分
输出	每个像素的类别	每个实例的 bbox + mask
结构	Backbone + ASPP + Decoder	Faster R-CNN + RoIAlign + Mask Head
适合	道路、天空、建筑、缺陷区域	人、车、细胞、苹果等实例目标

例如图中有 3 个人：

DeepLab：
    所有人像素都预测为 person

Mask R-CNN：
    person_1 mask
    person_2 mask
    person_3 mask

所以如果只关心类别区域，用 DeepLab；如果需要区分每一个个体，用 Mask R-CNN。

## 十五、DeepLab 的输出和标签格式

DeepLab 做多类别语义分割时，输出通常是：

outputs: [B, C, H, W]

其中：

B = batch size
C = 类别数
H, W = 输出分辨率

标签通常是：

masks: [B, H, W]

每个像素是类别 ID：

0 = background
1 = road
2 = car
3 = person
...

训练时通常使用：

nn.CrossEntropyLoss()

注意：

CrossEntropyLoss 输入 logits，不需要手动 softmax

如果是二分类分割，也可以输出：

[B, 1, H, W]

使用：

BCEWithLogitsLoss

推理时：

pred = output.argmax(dim=1)

得到每个像素的类别 ID。

## 十六、DeepLab 常用损失函数

DeepLab 常用于语义分割，常见损失包括：

CrossEntropy Loss
Weighted CrossEntropy Loss
Dice Loss
Focal Loss
Lovasz-Softmax Loss
CE + Dice
### 1. CrossEntropy Loss

多类别语义分割最常用：

criterion = nn.CrossEntropyLoss()

适合类别相对均衡的情况。

### 2. Weighted CrossEntropy

如果类别不均衡，比如背景很多、目标很少，可以给少数类更高权重：

criterion = nn.CrossEntropyLoss(weight=class_weights)
### 3. Dice Loss

适合前景区域小、类别不均衡的分割任务。

例如：

裂缝
病灶
缺陷
小目标区域
### 4. Focal Loss

适合难样本多、前景少的情况。

它会降低简单样本权重，提高困难样本权重。

## 十七、DeepLab 常用评价指标

语义分割常见指标包括：

Pixel Accuracy
Mean Pixel Accuracy
IoU
mIoU
Dice
Precision
Recall
F1-score
Boundary F1

其中最常用的是：

mIoU

mIoU 表示每个类别 IoU 的平均值。

IoU = 预测区域和真实区域的交集 / 并集

如果类别不均衡，不可只看 Pixel Accuracy。

例如背景占 95%，模型全部预测背景，也可能有很高 Accuracy。

对于缺陷、病灶、小目标分割，更应该关注：

Dice
IoU
Recall
F1-score

## 十八、DeepLab 的训练流程

训练 DeepLab 的流程一般是：

1. 准备图像和对应 mask
2. 划分 train / val / test
3. 对 image 和 mask 做同步增强
4. 选择 backbone，例如 ResNet / Xception / MobileNet
5. 加载 ImageNet 预训练权重
6. 设置 output stride，例如 16 或 8
7. 构建 ASPP 和 decoder
8. 设置损失函数
9. 训练模型
10. 在验证集上计算 mIoU / Dice
11. 保存最优模型
12. 在测试集上最终评估

数据增强时要注意：

image 和 mask 的几何变换必须同步
mask resize 必须用最近邻插值
颜色增强只作用于 image

例如：

随机裁剪：image 和 mask 同时裁剪
随机翻转：image 和 mask 同时翻转
颜色扰动：只改 image，不改 mask

## 十九、DeepLab 的推理流程

推理流程如下：

1. 读取图像
2. Resize / Normalize
3. 输入 DeepLab
4. 得到 logits: [1, C, H, W]
5. argmax 得到每个像素类别
6. 将类别 ID 映射为颜色
7. 可视化或保存 mask

如果使用滑窗推理：

大图切成小块
每块分别预测
再拼回完整 mask

这种方法常用于：

遥感图像
工业高分辨率图像
医学大图
下水道缺陷图像

## 二十、DeepLab 的优点

DeepLab 的优点主要有：

1. 多尺度上下文建模能力强
2. 空洞卷积可以扩大感受野，同时保持较高特征分辨率
3. ASPP 对不同尺度目标比较友好
4. DeepLabv3+ 加入 decoder 后边界效果更好
5. 可以结合强 backbone，例如 ResNet、Xception、MobileNet
6. 适合自然场景语义分割和道路场景分割
7. 理论结构清晰，适合知识总结和论文项目讲解

尤其是城市街景、道路、建筑、天空、车辆等类别的语义分割，DeepLabv3+ 是非常经典的 baseline。

## 二十一、DeepLab 的缺点

DeepLab 也有一些不足。

### 1. 对极细边界不一定如 U-Net 直接

虽然 DeepLabv3+ 加入 decoder，但标准 DeepLab 的强项仍然是上下文语义。

对于医学边界、裂纹、细小缺陷等任务，U-Net 系列有时更自然。

### 2. Atrous rate 需要设计

不同数据集、不同输入分辨率下，合适的 dilation rate 可能不同。

如果 rate 太小：

感受野不够大

如果 rate 太大：

采样过于稀疏，可能丢失局部细节
### 3. 计算量可能较大

DeepLabv3+ 如果使用 Xception 或 ResNet101，精度高，但计算量也不小。

如果要移动端部署，可以考虑：

MobileNetV2 / MobileNetV3 backbone
轻量 ASPP
减少通道数
降低输入分辨率
模型剪枝 / 量化
### 4. 不区分实例

DeepLab 是语义分割模型。

如果图中有多个同类目标，它不会区分具体实例。

例如多个行人：

DeepLab 只输出 person 类区域
不会输出 person_1、person_2

如果需要实例级结果，要用 Mask R-CNN、YOLO-Seg、Mask2Former 等。

## 二十二、实际项目中怎么选择 DeepLab？

可以这样判断：

适合 DeepLab 的场景
道路场景分割
自然场景语义分割
遥感地物分类
城市街景分割
大目标区域分割
多尺度目标明显的语义分割

例如：

道路 / 车辆 / 行人 / 天空 / 建筑
水体 / 农田 / 建筑物 / 林地
地面 / 墙面 / 障碍物
也可以用于但需调优的场景
工业缺陷分割
医学病灶分割
裂缝检测
小目标分割

这些任务可以尝试 DeepLabv3+，但通常要配合：

Dice / Focal Loss
高分辨率输入
滑窗推理
前景裁剪采样
更强 decoder
边界损失
不适合的场景

如果需要区分每个独立实例，例如：

每个细胞单独分割
每个苹果单独分割
每个人单独分割

DeepLab 不够，需要实例分割模型。

## 二十四、DeepLab 核心总结

核心说明：

DeepLab 是一系列经典的语义分割模型，它的核心思想是利用空洞卷积和 ASPP 模块提升分割效果。空洞卷积可以在不明显增加参数量的情况下扩大卷积感受野，同时保持较高的特征图分辨率，这对于像素级分割非常重要。ASPP 则使用多个不同 dilation rate 的空洞卷积分支，从不同尺度提取上下文信息，从而增强模型对大目标、小目标和复杂场景的适应能力。

DeepLab 早期版本还会结合 DenseCRF 改善边界定位；DeepLabv3 进一步改进 ASPP，并加入 image-level feature 来增强全局上下文；DeepLabv3+ 则在 DeepLabv3 的基础上加入 decoder，将高级语义特征和浅层细节特征融合，从而改善边界分割效果。整体来看，DeepLabv3+ 可以理解为一个结合 ASPP 多尺度上下文和 encoder-decoder 边界恢复能力的语义分割网络。

## 二十五、延伸知识：什么是空洞卷积？

核心说明：

空洞卷积是在普通卷积采样点之间加入间隔，也就是 dilation rate。它可以在不增加卷积核参数量的情况下扩大感受野。比如一个 3×3 卷积，如果 dilation rate 为 2，它实际覆盖的区域相当于 5×5；如果 dilation rate 为 4，覆盖区域会更大。对于语义分割来说，空洞卷积可以减少下采样带来的分辨率损失，同时让模型看到更大范围的上下文。

## 二十六、延伸知识：ASPP 是什么？

核心说明：

ASPP 是 Atrous Spatial Pyramid Pooling，也就是空洞空间金字塔池化。它会使用多个不同 dilation rate 的空洞卷积分支并行处理同一个特征图，比如 rate 为 6、12、18 的卷积分支，再加上 1×1 卷积和全局池化分支，最后把这些多尺度特征拼接融合。这样模型可以同时捕获局部细节、中等尺度目标和全局上下文信息，提高对不同尺度目标的分割能力。

## 二十七、延伸知识：DeepLabv3 和 DeepLabv3+ 有什么区别？

核心说明：

DeepLabv3 的核心是改进 ASPP，通过多个空洞卷积分支和 image-level feature 捕获多尺度上下文信息，但它主要还是一个强语义编码结构。DeepLabv3+ 在 DeepLabv3 的基础上加入了一个简单的 decoder，把 ASPP 输出的高级语义特征上采样后，与 backbone 的低级浅层特征融合，再通过卷积细化分割结果。这样可以更好地恢复边界细节，所以 DeepLabv3+ 在边界分割上通常比 DeepLabv3 更好。

## 二十八、延伸知识：DeepLab 和 U-Net 怎么选？

核心说明：

如果任务更强调小数据、精细边界和局部细节，比如医学图像、细胞、裂纹、缺陷区域分割，可以优先尝试 U-Net 或 U-Net++。如果任务是自然场景、道路场景、城市街景这类多尺度语义分割，DeepLabv3+ 是很经典的选择，因为它的空洞卷积和 ASPP 对多尺度上下文建模能力很强。

简单说，U-Net 更强调通过跳跃连接恢复空间细节，DeepLab 更强调通过空洞卷积和 ASPP 捕获多尺度语义上下文。实际项目中两者都可以作为 baseline 做对比。