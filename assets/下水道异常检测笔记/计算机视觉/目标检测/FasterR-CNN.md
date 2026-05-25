# Faster R-CNN

## 一、Faster R-CNN 是什么？

Faster R-CNN 是一个经典的两阶段目标检测算法。

目标检测要解决两个问题：

1. 图中有什么物体？
2. 物体在图中的哪个位置？

所以模型最终输出的是：

类别 class
边界框 bounding box
置信度 score

例如：

person: 0.95, bbox = [x1, y1, x2, y2]
car:    0.88, bbox = [x1, y1, x2, y2]

Faster R-CNN 的核心贡献是提出了 **RPN，Region Proposal Network，区域候选网络**。它不再依赖 Selective Search 这类传统候选框算法，而是让神经网络自己学习“哪里可能有目标”，并且和后续检测网络共享卷积特征，从而显著提高检测效率。原论文指出，RPN 可以和 Fast R-CNN 共享整图卷积特征，使候选区域生成几乎没有额外成本，并在 VGG-16 上达到约 5 fps，同时只使用约 300 个 proposal 就能取得很好的检测效果.

## 二、Faster R-CNN 在 R-CNN 系列中的位置

要理解 Faster R-CNN，最好先看 R-CNN 系列的演进。

R-CNN
↓
Fast R-CNN
↓
Faster R-CNN
↓
Mask R-CNN
### 1. R-CNN：先提候选框，再逐个分类

R-CNN 的流程大致是：

输入图像
↓
Selective Search 生成约 2000 个候选区域
↓
每个候选区域裁剪出来
↓
分别送入 CNN 提取特征
↓
SVM 分类
↓
边界框回归

R-CNN 的重要意义是把 CNN 引入目标检测，通过“区域候选框 + CNN 特征”的方式大幅提升了检测效果；R-CNN 论文报告其在 PASCAL VOC 2012 上达到 53.3% mAP，相比之前最佳结果提升超过 30%。

但 R-CNN 的问题很明显：

每个候选框都要单独跑一次 CNN
计算量很大
训练流程复杂
推理速度慢
### 2. Fast R-CNN：整张图只跑一次 CNN

Fast R-CNN 做了一个重要改进：

先对整张图做一次 CNN 特征提取
再在共享特征图上处理每个 RoI

流程是：

输入图像
↓
CNN 提取整图 feature map
↓
Selective Search 生成 RoI
↓
RoI Pooling 把不同大小 RoI 变成固定大小特征
↓
全连接层
↓
分类 + 边界框回归

Fast R-CNN 相比 R-CNN 的关键提升是：不再对每个候选框重复跑 CNN，而是共享整张图的卷积特征；它还把分类和边界框回归放在同一个网络中训练，提升了训练和推理效率。

但 Fast R-CNN 还有一个瓶颈：

候选框仍然依赖 Selective Search
Selective Search 不是神经网络
速度慢
无法端到端训练
### 3. Faster R-CNN：用 RPN 替代 Selective Search

Faster R-CNN 的核心改进就是：

用 RPN 网络生成候选框

也就是说，候选区域不再由传统算法生成，而是由 CNN 学出来。

整体结构变成：

输入图像
↓
Backbone 提取共享特征图
↓
RPN 生成候选区域 proposals
↓
RoI Pooling / RoI Align
↓
检测头分类 + 边界框回归
↓
输出最终检测结果

Faster R-CNN 本质上是：

RPN + Fast R-CNN

RPN 负责“找哪里可能有物体”，Fast R-CNN 检测头负责“判断是什么物体，并进一步修正框的位置”。

## 三、Faster R-CNN 的整体结构

Faster R-CNN 主要由四个部分组成：

1. Backbone 特征提取网络
2. RPN 区域候选网络
3. RoI Pooling / RoI Align
4. Detection Head 检测头

## 四、Backbone：特征提取网络

Backbone 的作用是从原始图像中提取视觉特征。

常见 backbone：

VGG16
ResNet-50
ResNet-101
ResNeXt
Swin Transformer

原始 Faster R-CNN 论文中常用的是 VGG-16；后来的工程实现中，ResNet + FPN 版本非常常见。Faster R-CNN 论文强调 RPN 和检测网络共享整图卷积特征，这是它相对于 Fast R-CNN 的关键效率来源。

Backbone 输出的是特征图，例如输入图像：

输入：[3, H, W]

经过 CNN 后得到：

feature map：[C, H', W']

这个 feature map 会同时送给：

RPN
Detection Head

也就是说，RPN 和检测头共享同一份卷积特征。

## 五、RPN 是什么？

RPN，全称：

Region Proposal Network

中文可以叫：

区域候选网络

它的作用是：

在特征图上快速预测哪些位置可能存在目标，并生成一批候选框 proposals。

RPN 不负责判断具体类别，比如“猫”“狗”“车”。

它只判断：

这个区域有没有物体？
这个候选框应该怎么调整？

所以 RPN 是一个类别无关的目标候选框生成器。

## 六、RPN 的输入和输出

RPN 的输入是 backbone 输出的 feature map：

feature map: [C, H', W']

RPN 在 feature map 的每个位置上滑动一个小窗口，通常可以理解为一个 3×3 卷积。

在每个位置，RPN 会预测若干个 anchor。

如果每个位置有 k 个 anchor，那么 RPN 输出：

1. objectness 分类分数：每个 anchor 是前景 / 背景
2. bbox regression：每个 anchor 的边界框修正量

输出可以理解为：

objectness: 2k 个分数
bbox regression: 4k 个偏移量

其中：

2k：每个 anchor 有 foreground / background 两类
4k：每个 anchor 预测 dx, dy, dw, dh 四个回归参数

RPN 论文描述它是一个全卷积网络，会在每个位置同时预测 objectness score 和 object bounds。

## 七、Anchor 是什么？

Anchor 可以理解为：

在特征图每个位置预先放置的一组参考框。

比如在某个特征图位置上，放置 9 个 anchor：

3 种尺度 × 3 种长宽比 = 9 个 anchor

常见长宽比：

1:1
1:2
2:1

常见尺度：

小尺度
中尺度
大尺度

这样做的原因是：图像中的物体大小和形状不一样。

例如：

人：细长
车：扁宽
球：接近正方形

Anchor 就是给模型一个初始参考框，模型只需要学习如何修正这个框，而不是完全从零预测。

## 八、RPN 如何判断正负样本？

训练 RPN 时，需要告诉模型哪些 anchor 是正样本，哪些是负样本。

通常根据 anchor 和真实框 ground truth 的 IoU 来判断。

一般思想是：

IoU 高的 anchor → 正样本
IoU 低的 anchor → 负样本
中间区域 → 忽略

例如：

anchor 与某个 GT 的 IoU ≥ 0.7 → 正样本
anchor 与所有 GT 的 IoU ≤ 0.3 → 负样本
其他 anchor → 不参与训练

此外，为了保证每个真实目标至少有一个正样本，和某个 GT IoU 最大的 anchor 通常也会被设为正样本。

这个正负样本分配机制让 RPN 学会区分：

哪些区域像物体
哪些区域只是背景

## 九、RPN 的两个任务

RPN 有两个任务：

1. Objectness classification
2. Bounding box regression
   
### 1. Objectness classification

判断 anchor 是前景还是背景。

也就是：

这个 anchor 里有没有物体？

注意，这里不是判断具体类别。

例如：

有物体：foreground
没物体：background

如果一张图里有猫、狗、车，RPN 不关心它们分别是什么，只关心：

这里可能有一个目标
那里可能没有目标
### 2. Bounding box regression

修正 anchor 的位置和大小。

模型预测的是：

dx, dy, dw, dh

它表示：

anchor 的中心点应该移动多少
anchor 的宽高应该缩放多少

通过这些偏移量，可以把 anchor 调整成更接近真实目标的 proposal。

## 十、边界框回归怎么理解？

假设 anchor 是：

A = (xa, ya, wa, ha)

真实框是：

G = (xg, yg, wg, hg)

其中：

x, y：中心点坐标
w, h：宽高

模型学习的不是直接输出真实框坐标，而是学习相对变换：

tx = (xg - xa) / wa
ty = (yg - ya) / ha
tw = log(wg / wa)
th = log(hg / ha)

预测时，模型输出：

dx, dy, dw, dh

再把 anchor 转换成 proposal：

xp = dx * wa + xa
yp = dy * ha + ya
wp = exp(dw) * wa
hp = exp(dh) * ha

这样做的好处是：

学习相对偏移比直接学习绝对坐标更稳定
不同尺度目标可以统一处理

## 十一、RPN 如何生成 proposals？

RPN 生成 proposals 的流程大致是：

1. 在特征图每个位置生成 anchors
2. 对每个 anchor 预测 objectness 和 bbox offset
3. 根据 bbox offset 修正 anchors
4. 得到大量 proposal boxes
5. 裁剪超出图像边界的框
6. 去除太小的框
7. 按 objectness 分数排序
8. 使用 NMS 去除高度重叠框
9. 保留 top-N proposals

最后输出一批候选框，例如：

训练阶段：保留 2000 个 proposals
测试阶段：保留 300 个 proposals

不同实现会有不同超参数，但整体思想一致。

## 十二、NMS 是什么？

NMS 是：

Non-Maximum Suppression
非极大值抑制

它的作用是去掉重复框。

RPN 可能会对同一个物体生成很多相似框：

box1 score = 0.96
box2 score = 0.92
box3 score = 0.89
box4 score = 0.75

这些框可能都围着同一个人。

NMS 的流程是：

1. 按 score 从高到低排序
2. 保留分数最高的框
3. 删除与它 IoU 过高的其他框
4. 继续处理下一个框
5. 最终得到较少的非重复 proposals

这样可以避免后续检测头处理大量重复区域。

## 十三、RoI Pooling 是什么？

RPN 输出的 proposals 大小不同。

例如：

proposal 1: 40 × 60
proposal 2: 120 × 80
proposal 3: 300 × 200

但是后面的全连接层需要固定大小输入。

所以需要 RoI Pooling。

RoI Pooling 的作用是：

把不同大小的 proposal 区域，转换成固定大小的特征。

例如统一变成：

7 × 7 × C

流程是：

1. 将 proposal 映射到 feature map 上
2. 把该区域划分成固定数量的小格子，比如 7×7
3. 对每个格子做 max pooling
4. 得到固定大小特征

Fast R-CNN 使用 RoI Pooling 来把不同尺度的 RoI 转换为固定长度特征，然后送入全连接层进行分类和边界框回归。

## 十四、RoI Pooling 的问题

RoI Pooling 有一个问题：

量化误差

因为 proposal 在原图上的坐标映射到 feature map 时，可能不是整数。

RoI Pooling 会进行取整操作，例如：

3.7 → 4
8.2 → 8

这会导致特征和原图区域之间产生轻微错位。

对于目标检测，这个问题通常还能接受。

但对于实例分割，像素级精度要求更高，这种错位会影响 mask 质量。

所以后来的 Mask R-CNN 提出了 RoI Align，用双线性插值代替粗暴取整，从而更好地保留空间对齐信息。Mask R-CNN 论文明确说明它是在 Faster R-CNN 基础上增加 mask 分支，并用 RoIAlign 改善 RoI 特征对齐。

## 十五、Detection Head 检测头

经过 RoI Pooling / RoI Align 后，每个 proposal 都变成固定大小的特征。

然后进入检测头。

检测头通常有两个分支：

1. 分类分支 classification
2. 边界框回归分支 bbox regression
### 1. 分类分支

分类分支负责判断 proposal 属于哪个类别。

例如：

background
person
car
dog
cat
...

如果有 C 个目标类别，通常还会加一个背景类：

C + 1

例如 COCO 有 80 类，则分类输出可能是：

81 类 = 80 个目标类别 + 1 个背景类
### 2. 边界框回归分支

检测头还会进一步修正 proposal 的位置。

注意，RPN 已经修正过一次框了。

但 RPN 只做类别无关的粗定位。

检测头会做更精细的框回归：

RPN：这个地方可能有物体，先给一个大致框
Detection Head：这个物体是 car，并把框修得更准

所以 Faster R-CNN 的定位过程可以理解为两步：

第一步：RPN 粗定位
第二步：检测头精定位

## 十六、Faster R-CNN 的训练损失

Faster R-CNN 的损失可以分成两大部分：

1. RPN loss
2. Fast R-CNN detection loss
### 1. RPN Loss

RPN loss 包括：

objectness classification loss
bbox regression loss

也就是：

L_rpn = L_cls_rpn + L_reg_rpn

其中：

L_cls_rpn：判断 anchor 是前景还是背景
L_reg_rpn：修正 anchor 到真实框
### 2. Detection Head Loss

检测头 loss 包括：

proposal classification loss
proposal bbox regression loss

也就是：

L_det = L_cls_det + L_reg_det

其中：

L_cls_det：判断 proposal 属于哪个类别
L_reg_det：进一步修正 proposal 的位置
### 3. 总损失

整体可以写成：

L = L_cls_rpn + L_reg_rpn + L_cls_det + L_reg_det

更直观地说：

RPN 学会找目标候选区域
检测头学会识别目标类别并精修边界框

## 十七、Faster R-CNN 的训练流程

Faster R-CNN 原论文中使用过交替训练方式，让 RPN 和 Fast R-CNN 逐步共享特征；现代框架中通常可以端到端联合训练。

训练流程可以理解为：

1. 输入图像和真实标注框
2. Backbone 提取 feature map
3. RPN 在 feature map 上生成 anchors
4. 根据 GT 给 anchors 分配正负样本
5. RPN 预测 objectness 和 bbox offset
6. 计算 RPN loss
7. RPN 生成 proposals
8. 对 proposals 和 GT 做匹配，分配类别标签
9. RoI Pooling / RoI Align 提取 proposal 特征
10. Detection Head 预测类别和 bbox offset
11. 计算检测头 loss
12. 反向传播，更新网络参数

可以总结为：

先训练模型找到可能有目标的区域
再训练模型判断这些区域是什么类别以及边界在哪里
## 十八、Faster R-CNN 的推理流程

推理时没有真实标签，流程是：

1. 输入图片
2. Backbone 提取 feature map
3. RPN 生成大量 proposals
4. 对 proposals 做排序、裁剪、NMS
5. 保留 top proposals
6. RoI Pooling / RoI Align 提取每个 proposal 的特征
7. 检测头输出类别概率和边界框修正
8. 过滤低分框
9. 对每个类别做 NMS
10. 输出最终检测结果

最终输出：

[class_id, score, x1, y1, x2, y2]

## 二十六、面试中如何介绍 Faster R-CNN？

如果面试官问：

你了解 Faster R-CNN 吗？

可以这样回答：

Faster R-CNN 是 R-CNN 系列中的经典两阶段目标检测算法。它主要由 backbone、RPN、RoI Pooling 和检测头组成。首先，backbone 对整张图像提取共享特征图；然后 RPN 在特征图上基于 anchor 预测 objectness 分数和边界框偏移量，生成一批可能包含目标的 region proposals；接着通过 RoI Pooling 或 RoI Align 将不同大小的 proposals 转换成固定尺寸特征；最后检测头对每个 proposal 进行具体类别分类和边界框回归，得到最终检测结果。

Faster R-CNN 相比 Fast R-CNN 的核心改进是用 RPN 替代了 Selective Search，使候选框生成也由神经网络完成，并且 RPN 与检测网络共享卷积特征，因此检测效率和候选框质量都有明显提升。它的优点是检测精度高、定位稳定，缺点是两阶段结构相对复杂，推理速度通常不如 YOLO 这类单阶段检测器。

## 二十七、如果面试官追问：RPN 是怎么工作的？

可以回答：

RPN 是一个全卷积网络，它在 backbone 输出的 feature map 上滑动小窗口，并在每个位置生成多个不同尺度和长宽比的 anchor。对于每个 anchor，RPN 会预测两个东西：第一是 objectness 分数，用来判断该 anchor 是前景还是背景；第二是 bbox regression 偏移量，用来修正 anchor 的位置和大小。经过边界框解码、排序、裁剪和 NMS 后，RPN 输出一批高质量 proposals，供第二阶段检测头进一步分类和回归。

## 二十八、如果面试官追问：为什么 Faster R-CNN 比 Fast R-CNN 快？

可以回答：

Fast R-CNN 虽然共享了整张图像的 CNN 特征，但候选区域仍然依赖 Selective Search 这类传统算法，proposal 生成速度较慢，而且不能和检测网络端到端联合优化。Faster R-CNN 用 RPN 替代 Selective Search，让 proposal 生成也变成神经网络的一部分，并且 RPN 和检测头共享 backbone 特征，因此候选框生成的额外成本很低，整体检测速度更快。

## 二十九、如果面试官追问：Faster R-CNN 的损失函数有哪些？

可以回答：

Faster R-CNN 的损失主要包括 RPN 阶段的损失和检测头阶段的损失。RPN 阶段包括 objectness 分类损失和 anchor 的边界框回归损失；检测头阶段包括 proposal 的多类别分类损失和边界框回归损失。所以整体可以看成四部分：RPN 分类损失、RPN 回归损失、检测头分类损失和检测头回归损失。

## 三十、如果面试官追问：Faster R-CNN 和 YOLO 怎么选？

可以回答：

如果任务更关注检测精度和定位稳定性，并且对实时性要求不是特别高，我会优先考虑 Faster R-CNN 这类两阶段检测器。它通过 RPN 先筛选候选区域，再对候选区域进行精细分类和回归，检测结果通常比较稳定。

如果任务要求实时推理，比如视频流检测、移动端部署、边缘设备部署，我会更倾向于 YOLO 这类单阶段检测器，因为 YOLO 直接一次前向传播输出检测结果，速度更快，部署链路也更成熟。