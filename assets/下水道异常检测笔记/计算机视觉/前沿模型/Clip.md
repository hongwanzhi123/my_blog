# Clip

CLIP 是一种图文对齐模型，它把图像和文本映射到同一个语义向量空间中，让模型能够判断“一张图”和“一段文字”是否匹配。

它最重要的能力是：不需要针对每个新类别重新训练分类头，也可以通过文字描述完成图像分类、检索、开放词表识别等任务。

CLIP 是一种通过对比学习训练的图文对齐模型，它使用图像编码器和文本编码器将图像和自然语言描述映射到同一语义空间，通过相似度实现 zero-shot 分类、图文检索和开放词表视觉理解。它擅长语义匹配，但不直接做精确检测和分割，实际项目中常与 YOLO、SAM、Mask R-CNN 等模型结合使用。

## 一、CLIP 是什么？

CLIP 全称是：

Contrastive Language-Image Pre-training

中文可以理解为：

对比式语言-图像预训练

它的核心思想是：

给模型看大量“图片-文本”配对数据，让模型学习哪些图片和哪些文字语义匹配，哪些不匹配。

例如一张狗的图片和文本：

a photo of a dog

应该匹配。

而和：

a photo of a car

不应该匹配。

训练完成后，CLIP 就能做到：

输入图片
↓
输入若干候选文本类别
↓
计算图片和每个文本的相似度
↓
选择相似度最高的文本作为预测结果

比如给一张猫的图片，再给文本：

a photo of a cat
a photo of a dog
a photo of a car

CLIP 会把图片和这三句话分别编码成向量，然后比较相似度，最终选择：

a photo of a cat

## 二、CLIP 解决了什么问题？

传统图像分类模型通常是这样训练的：

图片 → CNN / ViT → 分类头 → 固定类别

比如训练一个猫狗分类模型，它只能识别：

cat
dog

如果想让它识别牙齿病灶、工业缺陷、车辆、螺丝、裂纹，就需要重新收集数据、标注类别、训练模型。

CLIP 的思路不一样。

CLIP 不直接训练一个固定类别分类器，而是训练：

图片编码器
文本编码器

让图片和文本可以在同一个语义空间中比较。

所以它可以做：

zero-shot classification
image-text retrieval
text-image retrieval
open-vocabulary recognition
多模态特征提取

也就是说，CLIP 不是只会识别训练时固定的几个类别，而是可以根据提供的文字描述进行判断。

## 三、CLIP 的整体结构

CLIP 主要有两个编码器：

1. Image Encoder：图像编码器
2. Text Encoder：文本编码器

结构可以表示为：

图像
↓
Image Encoder
↓
图像特征向量

文本
↓
Text Encoder
↓
文本特征向量

图像特征向量 · 文本特征向量
↓
相似度

更具体一点：

image → image_encoder → image_embedding
text  → text_encoder  → text_embedding

similarity = cosine_similarity(image_embedding, text_embedding)

CLIP 的目标是：

匹配的图文对，相似度高
不匹配的图文对，相似度低

## 四、Image Encoder 图像编码器

CLIP 的图像编码器可以有不同版本。

常见有：

ResNet
ViT
### 1. ResNet 版本

早期 CLIP 可以使用 ResNet 作为图像编码器。

流程类似：

输入图片
↓
卷积层
↓
残差模块
↓
全局特征
↓
投影到 CLIP embedding 空间

ResNet 的优势是：

卷积结构稳定
局部特征提取能力强
工程上成熟
### 2. ViT 版本

CLIP 也常使用 Vision Transformer 作为图像编码器。

ViT 处理图像的方式是：

输入图像
↓
切成 patch
↓
patch embedding
↓
Transformer Encoder
↓
得到图像全局特征

例如：

CLIP ViT-B/32
CLIP ViT-B/16
CLIP ViT-L/14

其中：

B = Base
L = Large
/32 = patch size 32
/16 = patch size 16
/14 = patch size 14

patch 越小，图像细节保留越多，但计算量也越大。

## 五、Text Encoder 文本编码器

CLIP 的文本编码器通常是 Transformer。

输入文本，例如：

a photo of a dog

会先被 tokenizer 分成 token，然后输入 Transformer：

文本
↓
Tokenization
↓
Token Embedding
↓
Position Embedding
↓
Transformer
↓
文本特征向量

最后输出一个文本 embedding。

这个文本 embedding 和图像 embedding 维度相同，这样二者才能计算相似度。

## 六、CLIP 的关键：图像和文本在同一个向量空间

CLIP 最核心的地方是：

图像和文本虽然模态不同，但最终都会被映射成同一维度的向量。

例如：

image_embedding: [512]
text_embedding:  [512]

然后计算余弦相似度：

cosine_similarity(image_embedding, text_embedding)

如果图像和文本匹配：

相似度高

如果不匹配：

相似度低

这就让模型具备了“用文本理解图像”的能力。

## 七、CLIP 是怎么训练的？

CLIP 使用的是 对比学习。

假设一个 batch 中有 N 对图文数据：

(image_1, text_1)
(image_2, text_2)
(image_3, text_3)
...
(image_N, text_N)

其中：

image_i 和 text_i 是匹配的
image_i 和 text_j，i ≠ j，是不匹配的

CLIP 会分别编码所有图片和文本：

image_1 → I1
image_2 → I2
...
image_N → IN

text_1 → T1
text_2 → T2
...
text_N → TN

然后计算所有图像和所有文本之间的相似度矩阵：

          T1     T2     T3   ...   TN
I1       s11    s12    s13        s1N
I2       s21    s22    s23        s2N
I3       s31    s32    s33        s3N
...
IN       sN1    sN2    sN3        sNN

正确匹配在对角线上：

I1 ↔ T1
I2 ↔ T2
I3 ↔ T3
...
IN ↔ TN

训练目标是：

让对角线上的相似度尽可能高
让非对角线上的相似度尽可能低

这就是 CLIP 的对比学习核心。

## 八、CLIP 的损失函数

CLIP 使用的是双向对比损失。

它不仅要求：

给定图像，找到正确文本

还要求：

给定文本，找到正确图像

所以有两个方向：

image-to-text loss
text-to-image loss

最后取平均：

loss = (loss_i2t + loss_t2i) / 2

更具体地说，模型会把相似度矩阵当作分类 logits。

对于 image-to-text：

每一张图片都要在 N 个文本里选出正确文本

对于 text-to-image：

每一段文本都要在 N 张图片里选出正确图片

这和普通分类任务很像，只不过类别不是固定类别，而是 batch 内其他样本。

## 九、为什么叫 Contrastive？

因为它通过“正样本”和“负样本”对比来学习。

正样本：

匹配的 image-text pair

负样本：

不匹配的 image-text pair

例如：

正样本：
狗的图片 ↔ "a photo of a dog"

负样本：
狗的图片 ↔ "a photo of a car"
狗的图片 ↔ "a photo of a chair"
狗的图片 ↔ "a photo of a flower"

模型不断学习：

让正确图文靠近
让错误图文远离

最终就学到了图像和语言之间的语义对应关系。

## 十、CLIP 的 Zero-Shot 分类

CLIP 最经典的用法是 zero-shot image classification。

传统分类模型需要训练分类头。

比如存在 10 个类别：

cat
dog
car
bird
horse
...

传统做法是训练一个：

Linear(feature_dim, 10)

CLIP 不需要这样。

CLIP 只需要把类别变成文本 prompt：

a photo of a cat
a photo of a dog
a photo of a car
a photo of a bird
a photo of a horse

然后：

图片 → 图像特征
每个文本 → 文本特征
计算图片和每个文本的相似度
选择相似度最高的类别

流程：

image
↓
image encoder
↓
image embedding

class names
↓
prompt template
↓
text encoder
↓
text embeddings

image embedding vs text embeddings
↓
similarity scores
↓
argmax
↓
predicted class

这就是 zero-shot 分类。

## 十一、Prompt 为什么重要？

CLIP 不是直接用类别名效果最好。

例如类别是：

dog

直接输入：

dog

可能不如输入：

a photo of a dog

因为 CLIP 训练时见到的大量文本更像自然语言描述，而不是孤立类别词。

所以常用 prompt template：

a photo of a {}
a blurry photo of a {}
a close-up photo of a {}
a photo of the small {}
a photo of the large {}

对于医学图像或工业图像，可以改成：

a dental image of {}
a close-up intraoral photo showing {}
an industrial inspection image of {}
a defect image of {}

这就叫 prompt engineering。

## 十二、Prompt Ensembling

为了让 CLIP 更稳定，可以使用多个 prompt 模板。

例如类别是 dog，构造多个句子：

a photo of a dog
a blurry photo of a dog
a close-up photo of a dog
a cropped photo of a dog

分别编码成文本向量，然后取平均。

这样可以减少某一个 prompt 表达不好的影响。

这叫：

prompt ensembling

它在 zero-shot 分类中经常有提升。

## 十三、CLIP 的图像检索能力

因为 CLIP 可以把图像和文本放在同一个空间，所以它很适合做检索。

1. 文本搜图

输入：

a red car on the street

CLIP 会把文本编码成向量，然后和数据库中所有图片向量比较，找相似度最高的图片。

应用：

以文搜图
图片数据库检索
素材搜索
电商搜索
医学图像检索
工业缺陷案例检索
2. 图搜文

输入一张图片，找最匹配的描述。

例如图片是猫，CLIP 会更接近：

a photo of a cat

而不是：

a photo of a truck
3. 图搜图

也可以把图片都编码成图像向量，然后做图像之间的相似度检索。

应用：

相似图片搜索
重复图片检测
缺陷样本聚类
以图找图

## 十四、CLIP 和普通图像分类模型的区别
对比项	普通分类模型	CLIP
训练目标	固定类别分类	图文对齐
输出	固定类别概率	图文相似度
新类别	通常要重新训练	可以通过文本 prompt 指定
泛化能力	依赖训练类别	开放词表能力更强
输入	图像	图像 + 文本
典型用法	分类	zero-shot、检索、多模态理解

一句话：

普通分类模型学的是“类别边界”，CLIP 学的是“图像和语言的语义对齐”。

## 十五、CLIP 和 ViT 的关系

CLIP 不是单纯的 ViT。

CLIP 是一个训练框架和多模态模型。

它可以使用 ViT 作为图像编码器。

也就是说：

ViT 是视觉 backbone
CLIP 是图文对齐模型

CLIP-ViT 的结构大概是：

图像 → ViT → image embedding
文本 → Transformer → text embedding

CLIP 的贡献不只是用了 ViT，而是通过对比学习把图像和文本对齐。

## 十六、CLIP 和 CNN 的关系

CLIP 的图像编码器可以是 CNN，也可以是 ViT。

如果是 ResNet 版 CLIP：

image encoder = ResNet
text encoder = Transformer

如果是 ViT 版 CLIP：

image encoder = ViT
text encoder = Transformer

所以 CLIP 和 CNN 不是互斥关系。

它的本质是：

图像编码器 + 文本编码器 + 对比学习

## 十七、CLIP 和 BLIP 的区别

CLIP 主要做：

图文匹配
图文检索
zero-shot 分类

BLIP 更偏：

图像描述生成
视觉问答
图文理解与生成

区别可以简单理解：

CLIP：判断图像和文本是否匹配
BLIP：可以生成或理解更复杂的图文内容

CLIP 的输出通常是 embedding 和相似度。

BLIP 可以生成 caption：

a dog running on the grass

## 十八、CLIP 和 SAM 的区别

SAM 是 Segment Anything Model，主要做分割。

CLIP 主要做图文语义对齐。

模型	主要能力
CLIP	图文匹配、zero-shot 分类、检索
SAM	根据点、框、mask prompt 分割物体

CLIP 不擅长像素级精确分割。

SAM 不擅长理解类别语义。

所以很多开放词表分割系统会组合：

CLIP 负责理解“是什么”
SAM 负责分割“在哪里”

例如：

用户输入：car
CLIP / Grounding 模型找到语义目标
SAM 生成精细 mask

## 十九、CLIP 在目标检测中的作用

原始 CLIP 本身不是目标检测模型。

它不能直接输出：

bbox
mask

但它可以帮助做开放词表检测。

例如传统检测模型只能检测固定类别：

person
car
dog
cat

开放词表检测希望模型能检测任意文本类别：

rust on metal
dental caries
crack on surface
missing screw

CLIP 可以提供文本语义特征，用于和视觉区域特征匹配。

常见思路：

候选区域 proposal
↓
区域图像特征
↓
CLIP 文本特征
↓
区域-文本相似度
↓
判断区域类别

一些开放词表检测/分割模型会使用 CLIP 或 CLIP 风格的图文对齐思想。

## 二十、CLIP 在图像分割中的作用

原始 CLIP 是图像级模型，它输出的是整张图片的 embedding。

所以它不直接适合语义分割，因为分割需要：

每个像素属于什么类别

但可以通过改造用于分割：

CLIPSeg
DenseCLIP
MaskCLIP
Open-vocabulary segmentation
SAM + CLIP

大致思路是：

用 CLIP 理解文本类别
用分割模型提供区域或像素 mask
用图文相似度判断 mask 类别

比如工业缺陷场景：

文本 prompt:
"a scratch defect"
"a stain defect"
"a crack defect"
"a normal surface"

然后用 CLIP 判断某个区域更像哪种描述。

## 二十一、CLIP 在工业视觉中的应用

如果需要走工业视觉检测方向，CLIP 可以这样用。

1. 缺陷图像检索

把所有历史缺陷图片编码成 CLIP 向量。

新来一张缺陷图，找最相似的历史缺陷。

应用：

缺陷案例库检索
相似异常样本查询
辅助工程师判断缺陷类型
2. 缺陷类别 zero-shot 初筛

构造 prompt：

a photo of a normal product surface
a photo of a scratch defect
a photo of a crack defect
a photo of a stain defect
a photo of a dent defect

输入一张图，看它和哪个 prompt 相似度最高。

注意：这只能作为辅助，不如专门训练的工业缺陷模型稳定。

3. 标注辅助

CLIP 可以帮助从大量未标注图片中筛选可能的缺陷样本。

例如用文本：

scratch defect
dirty surface
broken edge

去检索相似图像，让标注人员优先检查。

4. 与分割模型结合

可以先用 SAM 或传统算法生成候选区域，再用 CLIP 判断区域类别。

流程：

图像
↓
候选区域生成
↓
裁剪 ROI
↓
CLIP 计算 ROI 和文本 prompt 相似度
↓
确定缺陷类别

这种方式适合做开放词表或弱监督实验。

## 二十二、CLIP 在医学图像中的应用

CLIP 的思想也可以用于医学图像，但有一个问题：

通用 CLIP 主要在自然图文数据上训练，对医学图像存在 domain gap。

医学图像中的概念和自然图像差异很大，例如：

dental caries
periodontal lesion
periapical radiolucency
oral ulcer

通用 CLIP 不一定理解得好。

所以医学领域通常会使用：

医学图文预训练模型
领域微调 CLIP
医学报告-图像对齐

如果做牙齿检测/分割项目，可以这样理解：

YOLO / Mask R-CNN / U-Net：负责精确检测和分割
CLIP：更适合做图文语义检索、辅助分类、开放词表识别

不要直接指望通用 CLIP 完成牙齿病灶精细分割。

## 二十三、CLIP 的优点

CLIP 的主要优点：

1. 具备 zero-shot 分类能力
2. 可以通过文本 prompt 定义类别
3. 图像和文本统一到同一个语义空间
4. 适合图文检索
5. 开放词表能力强
6. 可以作为多模态特征提取器
7. 能和检测、分割、检索系统结合

它的最大价值是：

把视觉任务从“固定类别分类”扩展到“自然语言描述驱动的视觉理解”。
## 二十四、CLIP 的缺点

CLIP 也有明显局限。

1. 不擅长精确定位

原始 CLIP 主要是图像级模型。

它知道图片大概是什么，但不直接输出：

bbox
mask
关键点

所以不能直接替代 YOLO、Mask R-CNN、U-Net。

2. 对 prompt 敏感

不同文本模板可能带来不同结果。

例如：

dog
a photo of a dog
a blurry photo of a dog
a close-up photo of a dog

结果可能不一样。

所以实际使用时要做 prompt engineering。

3. 领域迁移有问题

通用 CLIP 对自然图像效果好，但对专业领域可能不稳定：

医学图像
工业缺陷
遥感图像
显微图像
牙科影像

这些领域最好做微调或使用领域 CLIP。

4. 容易学到数据偏见

CLIP 来自大规模图文数据，训练数据中可能包含偏差。

所以在高风险领域，例如医疗、司法、安防，不应无验证直接使用。

5. 对细粒度差异不一定可靠

CLIP 可以识别大类语义，但对于非常细粒度的差异，比如：

轻微龋齿 vs 牙釉质阴影
细小划痕 vs 表面纹理
早期病灶 vs 正常结构

可能不如专门训练的监督模型。

## 二十五、CLIP 的常见使用方式
1. Zero-shot 分类
没有训练分类器，只用文本 prompt 分类

适合：

快速 baseline
开放类别分类
小样本类别探索
2. Linear Probe

固定 CLIP 图像编码器，只训练一个线性分类头。

流程：

图像
↓
CLIP image encoder 冻结
↓
提取 image embedding
↓
训练 Linear classifier

优点：

训练快
小数据上比较稳定
3. Fine-tuning

解冻部分或全部 CLIP 参数，在目标数据集上训练。

风险：

数据少时容易过拟合
训练成本更高
可能破坏原来的通用语义能力
4. Adapter / LoRA

在 CLIP 上加轻量模块，只训练少量参数。

适合：

小数据领域适配
医学/工业领域迁移
节省显存
5. Prompt Tuning

不改模型参数，而是学习 prompt embedding。

典型思路：

learnable prompt + class name

比如：

[learnable tokens] + "caries"

适合在少量标注数据上提升 CLIP 分类效果。

## 二十八、CLIP 在项目里怎么用？

结合当前关注的几个方向：

Kaggle
目标检测
图像分割
工业视觉
医学牙齿分割
资料项目

可这样使用 CLIP。

1. 不要用 CLIP 替代 YOLO segmentation

当前的 AlphaDent 牙齿病理分割项目，核心模型还是：

YOLO-seg
Mask R-CNN
U-Net
DeepLabv3+
Mask2Former

CLIP 不适合直接输出牙齿病灶多边形。

2. 可以用 CLIP 做辅助分类

例如 YOLO 分割出一个病灶区域后，裁剪这个 ROI：

原图
↓
YOLO 分割 mask
↓
裁剪病灶区域
↓
CLIP 判断更像哪种病理描述

prompt 可以写：

a dental image of caries
a dental image of calculus
a dental image of tooth discoloration
a dental image of gum inflammation

但要注意，通用 CLIP 对医学术语可能不稳定，最好只作为实验或辅助。

3. 可以用 CLIP 做图像检索 Demo

这对适合作为项目扩展方向。

比如做一个工业缺陷检索系统：

输入文本：scratch defect
返回最相似的缺陷图片

或者：

输入一张缺陷图
返回历史相似缺陷案例

这能体现对多模态模型的理解。

## 二十九、CLIP 核心总结

核心说明：

CLIP 是一种图文对齐模型，全称是 Contrastive Language-Image Pre-training。它由图像编码器和文本编码器组成，分别把图像和文本映射到同一个向量空间中，然后通过余弦相似度判断图文是否匹配。训练时，CLIP 使用大规模图文配对数据，并采用对比学习损失，让同一对图文的相似度尽可能高，不匹配的图文相似度尽可能低。

CLIP 最典型的能力是 zero-shot 图像分类。比如不需要重新训练分类头，只需要把类别写成 prompt，例如 a photo of a cat、a photo of a dog，然后计算图片特征和这些文本特征的相似度，选择相似度最高的类别作为预测结果。

它和普通分类模型不同，普通模型只能识别固定类别，而 CLIP 可以通过自然语言描述扩展类别空间，所以很适合图文检索、开放词表分类和多模态理解。不过原始 CLIP 主要是图像级语义模型，不直接输出 bbox 或 mask，所以在检测和分割任务中通常需要和 YOLO、SAM、Mask R-CNN 等模型结合使用。

## 三十、延伸知识：CLIP 为什么能 zero-shot？

核心说明：

因为 CLIP 训练时不是学习固定类别标签，而是学习图像和文本之间的语义对应关系。训练完成后，文本编码器可以把任意类别描述编码成向量，图像编码器也能把图片编码成向量。只要某个类别可以用文本描述出来，模型就可以通过图像向量和文本向量的相似度进行分类。因此它不需要针对新类别重新训练分类头，就能做 zero-shot 分类。

## 三十一、延伸知识：CLIP 的 loss 怎么设计？

核心说明：

CLIP 在一个 batch 中同时输入 N 对图文样本，分别得到 N 个图像特征和 N 个文本特征，然后计算 N×N 的相似度矩阵。矩阵对角线表示正确匹配的图文对，非对角线表示负样本。CLIP 分别做 image-to-text 和 text-to-image 两个方向的交叉熵损失，让每张图片匹配正确文本，也让每段文本匹配正确图片，最后把两个方向的 loss 求平均。

## 三十二、延伸知识：CLIP 有什么局限？

核心说明：

CLIP 的局限主要有三点。第一，原始 CLIP 是图像级模型，不直接提供精确定位能力，所以不能直接用于目标检测或像素级分割。第二，它对 prompt 比较敏感，不同文本模板会影响分类效果。第三，通用 CLIP 对专业领域可能存在 domain gap，比如医学影像、工业缺陷、遥感图像等，往往需要领域数据微调或使用领域版 CLIP 才能稳定应用。