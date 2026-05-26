# SAM 
SAM：Segment Anything Model。它是 Meta AI 在 2023 年提出的图像分割基础模型，核心目标不是针对某一个固定类别训练一个分割器，而是做成一个可提示、可迁移、可零样本泛化的通用分割模型。原始 SAM 和 SA-1B 数据集一起发布，SA-1B 包含约 1100 万张图像、10 亿级 mask，这是 SAM 能具备强泛化能力的重要基础之一。

## 一、SAM 是什么？

SAM，全称是：

Segment Anything Model

中文可以理解为：

分割一切模型

它的任务是：

给定一张图像
再给定某种提示 prompt
模型输出对应区域的分割 mask

比如给它一张图，然后：

点一下某个物体
画一个框
给一个已有粗 mask

SAM 就会输出这个目标的分割结果。

原始 SAM 官方仓库说明，它可以根据 点、框等输入提示生成高质量 object masks，也可以用于自动生成图像中所有对象的 masks。

## 二、SAM 和传统分割模型有什么不同？

传统分割模型通常是任务专用的。

例如 U-Net：

输入：医学图像
输出：病灶 mask

DeepLab：

输入：街景图像
输出：道路、天空、建筑、车辆等语义类别

Mask R-CNN：

输入：图像
输出：每个实例的 bbox + mask

这些模型通常需要针对具体数据集和类别训练。

而 SAM 的目标是：

不是训练一个“猫狗分割模型”
也不是训练一个“道路分割模型”
而是训练一个“根据提示分割任意区域”的通用模型

所以 SAM 更像是分割领域的基础模型。

它的核心能力是：

Promptable Segmentation

也就是：

可提示分割

告诉模型“要分割哪里”，它就根据提示输出对应区域。

## 三、SAM 的核心思想

SAM 的核心思想可以概括为一句话：

把图像分割变成一个可提示任务：模型先理解整张图像，再根据用户给出的 prompt 生成对应 mask。

传统语义分割模型通常是：

图像 → 每个像素类别

SAM 是：

图像 + prompt → mask

这个 prompt 可以是：

点 point
框 box
粗 mask

后续 SAM 3 进一步扩展到 文本概念、图像示例 exemplar、视觉提示，可以根据“yellow school bus”这类短语或示例图片检测、分割和跟踪匹配对象。

## 四、SAM 为什么被称为“分割基础模型”？

它和普通模型的区别在于：

普通分割模型：
    针对一个数据集训练
    针对固定类别预测
    泛化能力依赖任务数据

SAM：
    在超大规模 mask 数据上训练
    支持 prompt 输入
    可以迁移到很多新图像和新任务
    不一定需要针对每个任务重新训练

SAM 原论文报告其在多个分割任务上具有较强 zero-shot 表现，很多情况下可以和已有全监督模型竞争，甚至超过它们。

所以 SAM 的意义不只是“一个分割模型”，而是：

让分割任务从“为每个任务训练专用模型”
逐渐转向“使用通用基础模型 + prompt”

## 五、SAM 的整体结构

原始 SAM 主要由三部分组成：

1. Image Encoder
2. Prompt Encoder
3. Mask Decoder

论文中也明确说明 SAM 包含这三个组件：图像编码器、灵活的提示编码器和快速 mask 解码器。

整体流程是：

输入图像
↓
Image Encoder 提取图像 embedding
↓
输入 prompt
↓
Prompt Encoder 编码提示信息
↓
Mask Decoder 融合图像特征和 prompt 特征
↓
输出一个或多个 mask

可以简化为：

Image + Prompt → SAM → Mask

## 六、Image Encoder：图像编码器

Image Encoder 的作用是：

把整张图像编码成高维图像特征

它通常是一个强大的 Vision Transformer。

可以理解为：

输入图像
↓
切成 patch
↓
Transformer 提取全局视觉特征
↓
得到 image embedding

Image Encoder 是 SAM 中计算量最大的部分。

但它有一个很重要的特点：

对同一张图像，只需要编码一次。

也就是说：

图像编码一次
之后可以多次输入不同 prompt
快速得到不同 mask

例如打开一张图片：

第一次点击猫 → 分割猫
第二次点击狗 → 分割狗
第三次框选车 → 分割车

图像特征可以复用，不需要每次都重新完整编码图像。

这也是 SAM 能支持交互式分割的重要原因。

## 七、Prompt Encoder：提示编码器

Prompt Encoder 的作用是：

把用户输入的提示转换成模型能理解的向量表示

原始 SAM 支持的常见 prompt 包括：

点 point
框 box
mask

其中点又可以分为：

正点 positive point
负点 negative point
### 1. 正点 positive point

正点表示：

想要这个区域

例如点在一只猫身上，SAM 会倾向于分割猫。

### 2. 负点 negative point

负点表示：

不要这个区域

例如模型把猫和旁边的狗一起分出来了，可以在狗身上点一个负点，告诉模型：

狗不是要的目标

模型就会调整 mask。

### 3. 框 box

框表示：

目标大概在这个矩形区域里

这类似检测框提示。

例如画一个框框住汽车，SAM 会在这个框内分割汽车轮廓。

### 4. 粗 mask

粗 mask 表示：

已经有一个大致区域，请细化

这可以用于迭代式分割。

比如第一轮模型给出一个粗 mask，第二轮把这个 mask 作为输入，让模型进一步优化。

## 八、Mask Decoder：掩码解码器

Mask Decoder 的作用是：

根据 image embedding 和 prompt embedding 生成最终 mask

它相对 Image Encoder 更轻量。

可以理解为：

图像特征告诉模型：图里有什么、在哪里
prompt 特征告诉模型：用户想要哪一部分
mask decoder 负责把二者结合，输出 mask

SAM 的交互速度很大程度上来自这个设计：

Image Encoder 较重，但只跑一次
Prompt Encoder 和 Mask Decoder 较轻，可以快速响应多次提示

所以在实际交互中，用户不断点击点、画框，模型可以比较快地更新 mask。

## 九、SAM 的输出是什么？

SAM 通常输出：

一个或多个候选 mask
每个 mask 的质量预测分数

为什么会输出多个 mask？

因为有些 prompt 本身是模糊的。

例如在一个人身上点一下，这个点可能表示：

整个人
人的衣服
人的上半身
人的头部

一个点并不总能唯一确定用户意图。

所以 SAM 会输出多个可能 mask，让系统或用户选择最合适的一个。

这叫：

Ambiguity-aware mask prediction

也就是考虑提示歧义的 mask 预测。

## 十、SAM 的 promptable segmentation 怎么理解？

可以把 SAM 想成一个“分割工具”，而不是普通固定类别分割器。

普通语义分割模型像这样工作：

模型：只认识训练过的类别，比如 road、car、person
输入图像后，给每个像素分配类别

SAM 像这样工作：

用户：要这个区域
SAM：根据点/框/mask，把这个区域切出来

所以 SAM 不一定需要知道目标类别名称。

它更关注：

根据 prompt 找到对应的 object / region

这也是它适合交互式标注、自动标注、数据集构建的原因。

## 十一、SA-1B 数据集是什么？

SA-1B 是 SAM 一起发布的大规模分割数据集。

它包含：

约 1100 万张图像
约 10 亿级 mask

Meta 官方介绍 SA-1B 是为训练通用 object segmentation 模型而构建的开放世界图像数据集，SAM 官方论文也说明发布了约 1B masks 和 11M images。

这个数据集的重要意义是：

规模巨大
mask 数量远超传统分割数据集
覆盖开放世界图像
支撑 SAM 的泛化能力

很多传统分割数据集规模相对有限，例如只覆盖固定类别和固定场景。

SA-1B 的目标是让模型看到足够多样的物体、区域、形状和场景。

## 十二、SAM 的数据引擎

SAM 不只是发布了一个模型，也提出了一套数据构建思路。

可以理解为：

模型帮助标注数据
数据继续提升模型
更强模型再帮助标注更多数据

这种方式类似一个数据飞轮：

模型 → 辅助标注 → 更多 mask 数据 → 训练更强模型 → 更高效标注

Meta 官方介绍中提到，他们使用高效模型配合数据收集循环，构建了迄今规模最大的分割数据集。

这也是基础模型时代很典型的思路：

模型能力不只来自结构设计
也来自大规模数据和数据引擎

## 十三、SAM 可以做哪些任务？

SAM 原始模型主要用于图像分割，可以用于很多场景。

### 1. 交互式分割

用户通过点或框选择目标，SAM 输出 mask。

例如：

点一下猫 → 得到猫 mask
框住车 → 得到车 mask
点背景作为负点 → 去掉误分区域

这是 SAM 最典型的使用方式。

### 2. 自动 mask 生成

SAM 可以对整张图生成多个候选 mask。

这适合：

自动标注
数据集预处理
图像编辑素材提取
实例区域候选生成

官方仓库也说明 SAM 可以用于为图像中所有 objects 生成 masks。

### 3. 辅助数据标注

传统像素级分割标注非常耗时。

SAM 可以让标注员：

点几下
画个框
快速得到初始 mask
再人工修正

这能显著减少标注成本。

尤其适合：

医学图像
工业缺陷
遥感图像
自动驾驶
电商商品抠图
### 4. 图像编辑和抠图

SAM 可以把目标区域分割出来，用于：

换背景
删除物体
局部修图
图像合成
生成式图像编辑

比如先用 SAM 得到某个物体 mask，再把 mask 交给图像修复模型或扩散模型做编辑。

### 5. 作为下游任务预处理

SAM 可以为其他视觉任务提供候选区域。

例如：

检测前先生成区域
异常检测中先分割目标主体
医学图像中先得到器官粗 mask
工业图像中先提取产品区域

## 十四、SAM 是语义分割、实例分割还是交互式分割？

严格说，原始 SAM 不是传统意义上的语义分割模型，也不是完整的实例分割模型。

它更准确地说是：

Promptable Object Segmentation Model

也就是：

可提示目标分割模型

它输出的是 mask，但默认不直接输出语义类别。

例如点一只狗，SAM 可以分割出狗的轮廓，但它未必告诉：

这是 dog

它只告诉：

这是提示对应的区域 mask

所以：

任务	是否是 SAM 原始目标
语义分割	不是标准形式
实例分割	可辅助实现，但原始 SAM 不负责类别识别
交互式分割	是核心能力
自动 mask 生成	是重要能力
类别识别	原始 SAM 不擅长，需要结合分类/检测模型

如果要让 SAM 按文本类别找物体，通常需要结合 GroundingDINO、CLIP 或使用后续 SAM 3 这类支持 concept prompt 的模型。SAM 3 论文明确把任务扩展为 Promptable Concept Segmentation，可以用短名词短语、图像示例或二者组合来返回所有匹配对象的 masks 和身份。

## 十五、SAM 和 Grounded-SAM

实际项目中经常会看到：

Grounded-SAM

它通常是：

GroundingDINO + SAM

工作流程是：

文本 prompt：比如 "dog"
↓
GroundingDINO 找到 dog 的检测框
↓
把检测框传给 SAM
↓
SAM 输出 dog 的精确 mask

这样就弥补了原始 SAM 的一个问题：

SAM 擅长根据点/框/mask 分割
但原始 SAM 不负责文本语义检测

GroundingDINO 负责“找什么”，SAM 负责“切出来”。

可以理解为：

GroundingDINO：开放词汇检测
SAM：高质量分割

## 十六、SAM 的优点

SAM 的优点主要有：

1. 泛化能力强
2. 支持 promptable segmentation
3. 可以 zero-shot 应用于很多新场景
4. 交互式分割体验好
5. 适合辅助标注
6. 可用于自动 mask 生成
7. 和检测、分类、生成模型容易组合
8. 降低分割数据标注成本

尤其是辅助标注方面，SAM 非常有价值。

因为像素级分割数据很贵，如果能先用 SAM 生成初始 mask，再人工校正，可以大幅提升效率。

## 十七、SAM 的局限性

SAM 很强，但不是万能的。

### 1. 原始 SAM 不直接识别类别

原始 SAM 主要输出 mask，不直接告诉类别。

比如它可以切出一只狗，但它不一定负责输出：

class = dog

如果需要类别，需要结合：

分类模型
检测模型
CLIP
GroundingDINO
SAM 3
### 2. 对专业领域不一定完美

比如：

医学影像
工业缺陷
遥感图像
显微图像
红外图像
深度图

这些领域和自然图像差异较大，SAM zero-shot 可能能给出不错的粗分割，但未必达到专业模型精度。

此时可能需要：

微调 SAM
使用 MedSAM / domain-specific SAM
结合专业数据训练
### 3. 对语义边界和任务定义可能不稳定

有时候一个区域到底应该怎么分割，取决于任务定义。

例如一辆自行车：

只要车架？
包括轮子？
包括骑车人？

SAM 根据 prompt 可能给出不同粒度的 mask。

这既是优点，也是问题。

### 4. 小目标、透明物体、细长结构可能困难

例如：

很细的裂缝
电线
毛发
透明玻璃
严重遮挡目标
低对比度病灶

这些区域即使对人类也不容易标注，SAM 可能出现漏分或边界不准。

### 5. 模型较重

原始 SAM 的 Image Encoder 通常比较重。

如果要在移动端、实时视频、低算力设备上使用，需要考虑：

MobileSAM
FastSAM
EdgeSAM
SAM2 tiny
模型蒸馏
量化
裁剪

## 十八、SAM 和 U-Net 的区别
对比项	U-Net	SAM
任务定位	专用语义/医学/缺陷分割	通用 promptable segmentation
输入	图像	图像 + prompt
输出	固定类别 mask	prompt 对应的 mask
是否需要训练特定数据	通常需要	可 zero-shot 使用
是否输出类别	可以输出固定类别	原始 SAM 不直接输出类别
适合场景	医学、缺陷、小样本任务	交互式分割、自动标注、通用区域分割
部署成本	可以很轻	原始 SAM 较重

简单说：

U-Net 是为具体任务训练的专业分割模型
SAM 是根据提示进行通用分割的基础模型

如果存在大量专业标注数据，要在固定场景做到极致精度，U-Net 可能更合适。

如果想快速获得 mask、辅助标注、处理开放场景，SAM 更有优势。

## 十九、SAM 和 Mask R-CNN 的区别
对比项	Mask R-CNN	SAM
任务	实例分割	Promptable segmentation
是否输出类别	输出类别	原始 SAM 不直接输出类别
是否需要候选框	RPN 生成 proposals	用户 prompt 或自动 mask
是否固定类别	通常固定训练类别	更开放
输出	class + bbox + mask	mask
训练方式	监督训练在具体数据集	大规模 mask 数据训练
适合	已知类别实例分割	任意区域交互式分割、辅助标注

Mask R-CNN 更像：

训练好识别 person、car、dog，然后自动检测并分割它们

SAM 更像：

告诉要哪里，就尽量把那里切出来
## 二十、SAM 和 Mask2Former 的区别
对比项	Mask2Former	SAM
任务	语义/实例/全景统一分割	可提示分割
输出	多个 query 的 class + mask	prompt 对应 mask
是否输出类别	输出类别	原始 SAM 不直接输出类别
是否需要 prompt	通常不需要用户 prompt	需要点/框/mask 或自动模式
训练目标	数据集监督的通用分割框架	大规模 promptable segmentation
适合	标准分割 benchmark	交互式分割、数据标注、开放场景

Mask2Former 更像一个强大的通用分割架构，用于训练后直接输出语义/实例/全景结果。

SAM 更像一个交互式基础模型，用 prompt 控制分割对象。

## 二十一、SAM 和 DeepLab 的区别
对比项	DeepLab	SAM
任务	语义分割	Promptable segmentation
核心技术	空洞卷积 + ASPP	ViT image encoder + prompt encoder + mask decoder
输出	每个像素的类别	prompt 对应 mask
类别	固定类别	原始 SAM 不固定类别但也不直接分类
是否需要 prompt	不需要	需要
适合	道路、街景、自然场景语义分割	通用交互式分割、标注、抠图
## 二十二、SAM 和 YOLO-Seg 的区别
对比项	YOLO-Seg	SAM
任务	实时实例分割	Promptable segmentation
输出	class + bbox + mask	mask
是否识别类别	是	原始 SAM 不直接识别类别
速度	通常更适合实时部署	原始 SAM 较重
泛化	依赖训练类别	开放场景泛化更强
交互	通常不需要交互	强调 prompt 交互

如果需要做实时视频中固定类别分割：

YOLO-Seg 更合适

如果需要做开放场景中任意目标抠图和辅助标注：

SAM 更合适

## 二十三、SAM2：从图像扩展到视频

SAM 2 是 Meta 在 2024 年发布的下一代 Segment Anything 模型，核心变化是：

从图像分割扩展到图像 + 视频分割

Meta 官方介绍 SAM 2 是首个统一处理图像和视频对象分割的模型，可以用点击、框或 mask 在图像或视频帧中选择对象。

SAM2 的关键点包括：

1. 支持图像和视频
2. 支持实时 promptable video segmentation
3. 引入 streaming memory
4. 能在视频中持续跟踪目标 mask

SAM2 论文摘要说明，它是面向图像和视频 promptable visual segmentation 的基础模型，采用带 streaming memory 的简单 Transformer 架构，用于实时视频处理。

SAM2 为什么需要 memory？

图像分割只处理一张图。

视频分割需要处理连续帧：

frame 1
frame 2
frame 3
...

如果用户在第一帧点了一个目标，模型需要在后续帧中持续追踪这个目标。

这就需要记住：

目标之前长什么样
之前在哪里
mask 如何变化
是否被遮挡
是否重新出现

所以 SAM2 引入 memory 机制，让模型可以利用历史帧信息。

可以理解为：

SAM：
    当前图像 + 当前 prompt → mask

SAM2：
    当前帧 + prompt + 历史 memory → 当前帧 mask

## 二十四、SAM3：从“分割任意区域”到“分割任意概念”

截至 2026 年，Meta 已经发布了 SAM 3。SAM 3 的重点是：

Promptable Concept Segmentation

也就是：

根据概念提示进行检测、分割和跟踪

SAM 3 支持用：

短文本概念，例如 "yellow school bus"
图像示例 exemplar
文本 + 示例组合
视觉提示

来寻找图像或视频中所有匹配对象，并输出 mask 和 identity。SAM 3 论文摘要说明，它是一个统一模型，能够基于概念提示在图像和视频中检测、分割和跟踪对象。

这和原始 SAM 的区别很大：

SAM：
    点哪里 / 框哪里，分割哪里

SAM3：
    告诉概念，比如 yellow school bus
    找到所有匹配对象，并分割、跟踪它们

Meta 官方知识总结也将 SAM 3 描述为用于图像和视频中对象检测、分割和跟踪的统一模型，支持 text、exemplar 和 visual prompts。

## 二十五、SAM 系列演进总结

可以这样理解 SAM 系列：

SAM / SAM1：
    图像 promptable segmentation
    点、框、mask 提示
    输出对应 mask

SAM2：
    图像 + 视频 promptable segmentation
    引入 streaming memory
    支持视频中目标持续分割和跟踪

SAM3：
    Promptable Concept Segmentation
    支持文本概念、示例图片等 prompt
    统一检测、分割、跟踪

也可以总结为：

SAM1：点哪里，分哪里
SAM2：视频里点一次，持续分割和跟踪
SAM3：说出概念，找出并分割/跟踪所有匹配对象

## 二十六、SAM 在项目里怎么用？

如果做图像分割项目，SAM 通常有几种用法。

### 1. 直接作为交互式分割工具

适合：

抠图
图像编辑
目标区域提取
人工标注辅助

流程：

上传图像
用户点击/框选目标
SAM 输出 mask
用户微调
保存结果
### 2. 作为自动标注工具

适合需要构建数据集。

流程：

SAM 自动生成候选 masks
人工筛选和修正
生成训练集
再训练专用模型，如 U-Net / DeepLab / YOLO-Seg

这在工业缺陷、医学图像、遥感图像中很实用。

### 3. 和检测模型组合

比如：

YOLO / GroundingDINO 负责检测目标框
SAM 负责从框中分割精细 mask

流程：

输入图像
↓
检测模型输出 bbox
↓
bbox 作为 SAM prompt
↓
SAM 输出精细 mask

适合：

开放词汇实例分割
目标检测 + 精细轮廓
自动数据标注
### 4. 用 SAM 生成伪标签

如果没有足够 mask 标注，可以用 SAM 生成伪标签：

少量人工 prompt
↓
SAM 生成 mask
↓
人工检查
↓
作为训练标签
↓
训练轻量专用分割模型

例如想训练一个轻量化 U-Net 部署到移动端，可以用 SAM 辅助生成训练数据。

## 二十七、SAM 是否适合直接训练业务模型？

这要看任务。

如果目标是：

快速交互式分割
辅助标注
开放场景抠图

可以直接用 SAM。

如果目标是：

固定类别
高实时性
低算力部署
强业务指标

更常见做法是：

用 SAM 辅助构建数据
再训练专用模型

例如：

SAM 生成缺陷 mask
人工修正
训练轻量 U-Net / YOLO-Seg / DeepLab
部署到产线设备

这样可以兼顾：

SAM 的标注效率
专用模型的速度和稳定性

## 二十八、SAM 在医学图像中的使用

SAM 可以用于医学图像，但要谨慎。

医学图像和自然图像差异很大：

灰度图多
器官边界复杂
病灶区域低对比度
标注规则专业
细微结构重要

所以原始 SAM zero-shot 可能能给出不错初始 mask，但不一定达到临床或专业任务标准。

常见做法是：

使用医学领域微调版本
例如 MedSAM 类方法
或者用医学数据微调 SAM

在医学项目中，更稳妥的策略是：

SAM 辅助标注 + 专用医学分割模型训练

而不是完全依赖原始 SAM。

## 二十九、SAM 在工业缺陷检测中的使用

工业缺陷场景也类似。

缺陷通常有这些特点：

小
细
低对比度
形状不规则
背景纹理复杂
类别极不均衡

SAM 对明显目标区域可能表现很好，但对微小裂纹、划痕、针孔等缺陷可能不稳定。

常用用法：

1. 用 SAM 辅助生成初始 mask
2. 人工修正缺陷边界
3. 训练专用缺陷分割模型
4. 部署专用模型

如果直接用 SAM 做线上缺陷分割，可能会遇到：

速度慢
边界不稳定
误分背景纹理
无法输出缺陷类别

## 三十、SAM 的评价指标

如果把 SAM 用于分割任务，可以用常见分割指标评价：

IoU
mIoU
Dice
Boundary F1
Precision
Recall

如果是交互式分割，还可以评价：

点击次数达到某个 IoU 所需数量
NoC@85
NoC@90

其中 NoC 表示：

Number of Clicks

也就是达到目标分割质量所需的点击次数。

因为 SAM 是 promptable model，评价时不仅要看 mask 质量，还要看：

给什么 prompt
需要多少 prompt
用户交互成本多高

## 三十一、SAM 的典型推理流程

以点提示为例：

1. 输入图像
2. Image Encoder 提取 image embedding
3. 用户点击一个 positive point
4. Prompt Encoder 编码点坐标和点类型
5. Mask Decoder 融合 image embedding 和 prompt embedding
6. 输出多个候选 mask 和质量分数
7. 选择最佳 mask
8. 用户可继续添加正点/负点修正

以框提示为例：

1. 输入图像
2. 框选目标区域
3. box 作为 prompt
4. SAM 输出框内目标 mask

以自动 mask 生成模式为例：

1. 在图像上生成大量采样点
2. 对每个点运行 SAM
3. 得到大量候选 mask
4. 根据稳定性、质量分数、NMS 等策略筛选
5. 输出整图 object mask 集合

## 三十三、SAM 核心总结

核心说明：

SAM，也就是 Segment Anything Model，是 Meta 提出的一个通用图像分割基础模型。它的核心思想是把分割建模成 promptable segmentation，也就是输入图像和提示，输出对应区域的 mask。提示可以是点、框或者已有 mask。相比传统 U-Net、DeepLab 这类针对固定类别训练的分割模型，SAM 不直接预测固定语义类别，而是根据用户提示分割对应目标，因此具有很强的交互性和 zero-shot 泛化能力。

SAM 的结构主要包括三个部分：Image Encoder、Prompt Encoder 和 Mask Decoder。Image Encoder 通常是较大的 ViT，用来提取整张图像的 embedding；Prompt Encoder 把点、框、mask 等提示编码成向量；Mask Decoder 则融合图像特征和提示特征，输出一个或多个候选 mask 以及对应质量分数。由于图像特征可以预先计算，后续用户多次交互时只需要快速运行 prompt encoder 和 mask decoder，因此适合交互式分割和辅助标注。

SAM 的优势是泛化能力强、交互方便、适合自动标注和开放场景分割；局限是原始 SAM 不直接输出类别，专业领域如医学和工业缺陷可能需要微调或人工校正，而且原始模型较重，不一定适合低算力实时部署。

## 三十四、延伸知识：SAM 和传统语义分割有什么区别？

核心说明：

传统语义分割模型通常是对每个像素预测固定类别，比如道路、天空、车辆等，模型输出是 [B, C, H, W] 的类别 logits。而 SAM 不是固定类别分割模型，它输入的是图像和 prompt，比如点、框或 mask，然后输出该 prompt 对应的区域 mask。也就是说，传统语义分割回答的是“每个像素属于哪个类别”，SAM 回答的是“用户指定的这个区域具体轮廓是什么”。

## 三十五、延伸知识：SAM 为什么适合辅助标注？

核心说明：

因为像素级标注非常耗时，而 SAM 可以根据点或框快速生成高质量初始 mask。标注员不需要从零开始一点点画轮廓，而是可以先用 SAM 得到候选 mask，再进行修正。这样可以显著降低分割数据集构建成本。特别是在医学图像、工业缺陷、遥感图像等需要精细 mask 的任务中，SAM 可以作为半自动标注工具，提高数据生产效率。

## 三十六、延伸知识：SAM 的三个模块分别做什么？

核心说明：

Image Encoder 负责对整张图像提取视觉特征，通常比较重，但同一张图只需要计算一次。Prompt Encoder 负责把用户输入的点、框或 mask 编码成 prompt embedding。Mask Decoder 负责融合 image embedding 和 prompt embedding，生成最终分割 mask。这样的设计使得 SAM 可以预先计算图像特征，然后对不同 prompt 快速输出不同分割结果，适合交互式场景。

## 三十七、延伸知识：SAM、SAM2、SAM3 的区别？

核心说明：

SAM1 主要面向图像中的 promptable segmentation，用户通过点、框或 mask 提示模型分割对应区域。SAM2 把这种能力扩展到视频，引入 memory 机制，让模型可以在视频中根据用户提示持续分割和跟踪目标。SAM3 则进一步扩展到 promptable concept segmentation，支持文本概念和图像示例等提示，可以根据概念在图像和视频中检测、分割和跟踪所有匹配对象。

简单说，SAM1 是图像交互式分割，SAM2 是图像和视频交互式分割，SAM3 是基于概念提示的检测、分割和跟踪统一模型。