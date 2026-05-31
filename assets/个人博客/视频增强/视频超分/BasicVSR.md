# BasicVSR

BasicVSR 可以理解成视频超分里的一个“强基线”：它不靠复杂注意力、不靠很重的对齐模块，而是把视频超分拆成四件事：时序传播 Propagation、帧间对齐 Alignment、特征聚合 Aggregation、上采样 Upsampling，然后用最基础但有效的组件把它们组合起来。论文的目标不是提出一个特别花哨的模块，而是回答：视频超分到底哪些组件是最关键的？ 最终结论是：双向传播 + 光流特征对齐 + 简单特征拼接 + PixelShuffle，就能得到一个又快又强的 VSR baseline。

## 1. BasicVSR 解决的是什么问题

视频超分，Video Super-Resolution，输入是一段低分辨率视频：

LR: x1, x2, x3, ..., xt

输出是一段高分辨率视频：

HR: y1, y2, y3, ..., yt

和单图超分不同，视频超分可以利用相邻帧的信息。比如第 5 帧某个纹理很模糊，但第 4 帧或者第 6 帧由于运动位置不同，可能保留了更清楚的细节。BasicVSR 的核心就是：把整段视频的信息沿时间方向传递起来，让每一帧都能利用过去和未来的信息。

但是视频帧之间有运动，不能直接把不同帧特征相加。所以 BasicVSR 同时要解决两个问题：

第一，信息怎么从远处帧传过来？
这就是 propagation。

第二，传过来的特征怎么和当前帧对齐？
这就是 alignment。

论文认为，很多复杂 VSR 方法都可以拆成 propagation、alignment、aggregation、upsampling 四个基本功能；BasicVSR 的贡献就是把这四个功能用非常简洁的方式重新组合成一个强基线。

## 2. BasicVSR 的整体结构

BasicVSR 的整体流程可以写成：

低分辨率视频序列
        ↓
估计相邻帧光流
        ↓
Backward propagation：从后往前传播未来信息
        ↓
Forward propagation：从前往后传播过去信息
        ↓
融合当前帧的 forward hidden feature 和 backward hidden feature
        ↓
卷积 + PixelShuffle 上采样
        ↓
输出高分辨率视频帧

论文 Figure 2 里把 BasicVSR 画成一个典型的 bidirectional recurrent network：红色是 backward propagation，蓝色是 forward propagation；传播分支只包含三个通用组件：光流估计模块 S、空间 warping 模块 W、残差块 R；最后的上采样模块 U 由卷积和 PixelShuffle 组成。

一句话概括：

BasicVSR = 双向循环传播 + 光流特征对齐 + 残差块融合 + PixelShuffle 重建。

## 3. 为什么 BasicVSR 强调“传播”

视频超分不是只看当前帧，而是要利用整段视频。传播方式决定了模型能看到多少时间信息。BasicVSR 论文把传播分成三类：局部传播、单向传播、双向传播。

3.1 局部传播的问题

很多滑动窗口方法只看当前帧附近几帧，比如：

预测第 5 帧，只看第 3、4、5、6、7 帧

这种方法的问题是：时间感受野有限。如果第 1 帧或第 10 帧里有对当前帧有用的信息，滑动窗口方法就用不到。论文实验发现，时间感受野越大，PSNR 越好，说明远距离帧确实能提供有效信息。

3.2 单向传播的问题

单向传播类似这样：

1 → 2 → 3 → 4 → 5 → 6

第 6 帧能收到前面所有帧的信息，但第 1 帧只能看到自己，第 2 帧只能看到第 1 帧和第 2 帧。这样会导致不同时间位置的信息不平衡。论文指出，单向传播在早期时间步明显比双向传播差，并观察到大约 0.5 dB 的性能下降。

3.3 双向传播的优势

BasicVSR 使用双向传播：

Forward:  1 → 2 → 3 → 4 → 5
Backward: 5 → 4 → 3 → 2 → 1

这样第 i 帧既能获得过去帧的信息，也能获得未来帧的信息。

比如第 5 帧：

过去信息：1, 2, 3, 4 → 5
未来信息：9, 8, 7, 6 → 5

所以它可以利用整段视频的长期信息，而不是只看局部窗口。

BasicVSR 的传播公式可以理解成：

backward hidden:
h_i^b = F_b(x_i, x_{i+1}, h_{i+1}^b)

forward hidden:
h_i^f = F_f(x_i, x_{i-1}, h_{i-1}^f)

其中 h_i^b 是从未来传播到第 i 帧的隐藏特征，h_i^f 是从过去传播到第 i 帧的隐藏特征。论文正是用这种 forward / backward 两个传播分支来实现全局时序信息聚合。

## 4. BasicVSR 的对齐：光流 + 特征 warping

传播过来的特征不能直接用，因为视频中物体会运动。

比如第 t-1 帧中的车在左边，第 t 帧中的车在右边。如果你不对齐，直接把两个特征拼起来，就会出现重影、模糊、边缘错位。

BasicVSR 的做法是：

用光流估计相邻帧运动
        ↓
用光流把上一帧传播来的 hidden feature warp 到当前帧坐标系
        ↓
再用残差块融合当前帧和对齐后的传播特征

重点是：BasicVSR 不是在图像级别 warping，而是在特征级别 warping。

论文比较了不对齐、图像对齐、特征对齐三种情况。不做对齐会明显降低效果；图像级 warping 会因为光流误差导致图像模糊；BasicVSR 采用光流进行特征级 warping，再用残差块做后续 refinement。

论文中的对齐过程可以简化成：

s_i = S(x_i, x_{i±1})

这里 S 是光流估计模块，输入当前帧和相邻帧，输出光流。

h_bar_i = W(h_{i±1}, s_i)

这里 W 是 warping，用光流把上一帧或下一帧的隐藏特征对齐到当前帧。

h_i = R(x_i, h_bar_i)

这里 R 是残差块，用当前帧和对齐后的隐藏特征生成新的传播特征。论文中也明确说明，BasicVSR 的传播分支由光流估计 S、空间 warping W 和残差块 R 组成。

## 5. BasicVSR 为什么不用复杂注意力或 DCN

BasicVSR 的一个核心观点是：复杂模块不一定是视频超分最核心的东西，传播和对齐的选择更关键。

当时很多方法会用复杂模块，比如 EDVR 使用多尺度可变形对齐和 attention，RBPN 使用多个 projection 模块。这些方法有效，但模型复杂、推理慢、复现难。BasicVSR 刻意使用通用组件，比如 optical flow、residual blocks、feature concatenation、PixelShuffle，证明只要传播和对齐设计合理，简单结构也能很强。论文也报告，BasicVSR 在多个数据集上超过已有方法，并且具有效率优势。

所以 BasicVSR 的价值不是“模块很新”，而是：

它证明了视频超分里最关键的不是堆复杂模块，而是让信息在时间维度上充分、稳定地传播。

## 6. Aggregation：BasicVSR 如何融合特征

BasicVSR 的聚合非常简单：把当前帧对应的 forward feature 和 backward feature 拼接起来，然后交给上采样模块。

可以理解成：

当前帧 i 的信息来源：
1. h_i^f：从过去传播过来的信息
2. h_i^b：从未来传播过来的信息

融合：
concat(h_i^f, h_i^b)
        ↓
卷积融合
        ↓
上采样输出 y_i

论文中说 BasicVSR 在 aggregation 和 upsampling 上采用常见选择即可：aggregation 用特征拼接，upsampling 用 PixelShuffle。也就是说，它没有在聚合阶段设计复杂的 attention，而是把设计重点放在传播和对齐上。

## 7. Upsampling：PixelShuffle 重建高分辨率帧

BasicVSR 的上采样模块 U 由多层卷积和 PixelShuffle 组成。PixelShuffle 是超分里很常见的操作，它会把通道维度的信息重新排列到空间维度。

比如 2 倍超分时，可以把：

H × W × 4C

重排成：

2H × 2W × C

BasicVSR 最终输出：

y_i = U(h_i^f, h_i^b)

也就是根据过去传播特征和未来传播特征重建第 i 帧的高分辨率图像。论文中 Figure 2 和方法部分都说明，上采样模块 U 包含多个 PixelShuffle 和卷积。

## 8. BasicVSR 的完整前向过程

假设输入 7 帧：

x1, x2, x3, x4, x5, x6, x7
第一步：反向传播

从最后一帧往前：

x7 → x6 → x5 → x4 → x3 → x2 → x1

每一步做：

估计当前帧和后一帧的光流
用光流 warp 后一帧传播来的 hidden feature
把当前帧和 warped feature 输入残差块
得到当前帧的 backward feature

这样每个位置都拿到了“未来帧信息”。

第二步：正向传播

从第一帧往后：

x1 → x2 → x3 → x4 → x5 → x6 → x7

每一步做：

估计当前帧和前一帧的光流
用光流 warp 前一帧传播来的 hidden feature
把当前帧和 warped feature 输入残差块
得到当前帧的 forward feature

这样每个位置都拿到了“过去帧信息”。

第三步：融合与上采样

对每一帧：

concat(forward feature, backward feature)
        ↓
卷积融合
        ↓
PixelShuffle
        ↓
输出 HR 帧

所以第 4 帧最终既能用到：

x1, x2, x3 的过去信息

也能用到：

x5, x6, x7 的未来信息

这就是 BasicVSR 的核心优势。

## 9. BasicVSR 的训练细节

论文实验中，BasicVSR 使用 REDS 和 Vimeo-90K 作为训练数据，并在 REDS4、Vid4、UDM10、Vimeo-90K-T 等测试集上评估；任务主要是 4× 视频超分，并测试了 Bicubic 和 Blur Downsampling 两种退化。训练时使用预训练 SPyNet 作为光流估计模块，采用 Adam 优化器和 Cosine Annealing；输入 LR patch 大小为 64×64，loss 使用 Charbonnier loss，因为它相比普通 L2 对异常值更稳健。

从工程角度看，BasicVSR 的关键依赖是 光流估计。论文中 BasicVSR 和 IconVSR 的参数量统计也包含了 SPyNet，因此和其他方法比较时比较公平。

## 10. BasicVSR 的实验结论

论文比较了 BasicVSR、IconVSR 和 14 个已有 VSR 方法，包括 VESPCN、SPMC、TOFlow、FRVSR、DUF、RBPN、EDVR、MuCAN、PFNL、RSDN、RRN 等。BasicVSR 在 REDS4、UDM10、Vid4 等数据集上超过已有方法，并且推理效率较高。论文中特别提到，BasicVSR 在 UDM10 上相比 RSDN 提升 0.61 dB，在 REDS4 上相比复杂度更高的 EDVR 提升 0.33 dB。

但是论文也指出，BasicVSR 在 Vimeo-90K-T 上相比 EDVR、TGA 这类滑动窗口方法略低一些。原因是 Vimeo-90K-T 每个序列只有 7 帧，而 BasicVSR 的优势部分来自长时序信息聚合；序列太短时，它的长时序传播优势不能完全发挥。

这点很重要，说明 BasicVSR 不是所有场景都绝对最强，它更适合能够利用较长序列信息的视频恢复任务。

## 11. BasicVSR 的优点
11.1 结构简单，容易理解

BasicVSR 没有堆很多复杂模块，核心就是：

光流估计
特征 warping
残差块
双向传播
PixelShuffle

这使它非常适合作为视频增强项目的 baseline。

11.2 能利用长时序信息

相比滑动窗口方法，BasicVSR 可以通过 recurrent propagation 利用更长视频序列。论文实验也说明，增加时间感受野能提升恢复效果。

11.3 参数和速度效率好

BasicVSR 的项目页也强调，它由 residual blocks 和 optical flow 等通用组件组成，但能以更少参数和更快速度超过已有方法，因此适合作为 VSR backbone。

11.4 可扩展性强

BasicVSR 后续可以自然扩展成 IconVSR、BasicVSR++、RealBasicVSR 等。BasicVSR 论文自己也提出了 IconVSR，用信息补充和耦合传播继续增强性能。

## 12. BasicVSR 的局限
12.1 依赖光流质量

BasicVSR 的对齐靠 optical flow。如果光流估计错了，传播特征也会错。比如大运动、遮挡、反光、低纹理区域，光流都容易不准。

12.2 只是相邻帧一阶传播

BasicVSR 的每次传播主要来自相邻帧的 hidden state。它没有 BasicVSR++ 里的二阶传播，也没有 flow-guided deformable alignment。所以在复杂运动或遮挡区域，BasicVSR 可能不如 BasicVSR++。

12.3 特征 warping 会丢失边界信息

Warping 在图像边界、遮挡区域会出现无对应位置的情况，导致信息缺失。IconVSR 的 information-refill 就是为了解决这类问题：它在稀疏关键帧上提取额外深层特征，用来补充 warping 中丢失或错误对齐的信息。

12.4 对短序列优势不明显

如果视频片段很短，比如只有 7 帧，BasicVSR 的长时序传播优势不一定能充分发挥。论文中 Vimeo-90K-T 的结果就说明了这一点。

## 13. BasicVSR 和 IconVSR 的关系

IconVSR 是论文在 BasicVSR 基础上做的扩展，主要加了两个东西：

13.1 Information-Refill 信息补充

BasicVSR 的问题是：如果光流 warping 在遮挡或边界区域失败，错误特征会沿时间传播，导致误差累积。

IconVSR 的做法是：选一些关键帧 keyframes，用额外特征提取器从这些关键帧及其邻近帧提取深层特征，然后插入主传播网络，补充那些对齐不准或信息缺失的区域。论文指出，这个机制只在稀疏关键帧上使用，因此额外计算量不大。

13.2 Coupled Propagation 耦合传播

BasicVSR 的 forward 和 backward 两个分支是相对独立的：

forward 分支只看过去
backward 分支只看未来

IconVSR 让两个分支发生信息交换，把 backward propagation 的结果输入 forward propagation。这样 forward 分支不只看过去，也能间接获得未来信息。论文指出，coupled propagation 只改变分支连接方式，不引入额外计算开销。

所以关系是：

BasicVSR：双向传播，但两个方向相对独立
IconVSR：在 BasicVSR 上加入关键帧信息补充 + 前后向耦合传播

## 