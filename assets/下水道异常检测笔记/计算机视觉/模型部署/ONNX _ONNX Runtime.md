# ONNX / ONNX Runtime 模型部署

熟悉 ONNX 模型导出
熟悉 ONNX Runtime 推理部署
了解模型前处理、后处理
了解 CPU/GPU/TensorRT/OpenVINO 等推理后端
了解模型部署优化和常见问题排查

## 一、先明确：招聘要求到底想要你会什么？

招聘写：

掌握 ONNX / ONNX Runtime 进行模型部署

一般不是让你只会一句：

torch.onnx.export(...)

而是希望你能完成完整流程：

PyTorch / TensorFlow 训练模型
↓
导出 ONNX
↓
检查 ONNX 模型结构
↓
使用 ONNX Runtime 加载模型
↓
完成输入预处理
↓
执行推理
↓
完成输出后处理
↓
验证 ONNX 输出和原框架输出一致
↓
部署到实际业务环境

对于视觉算法岗位，尤其是目标检测、分割、工业视觉岗位，实际更看重：

1. 会把 PyTorch 模型导出为 ONNX
2. 会用 ONNX Runtime 写独立推理脚本
3. 会处理图像预处理和后处理
4. 会处理 YOLO / U-Net / 分类模型的部署
5. 会做 PyTorch 和 ONNX 输出一致性验证
6. 会优化推理速度和内存占用
7. 知道常见部署坑怎么排查

## 二、ONNX 是什么？

ONNX 全称是：

Open Neural Network Exchange

中文可以理解为：

开放神经网络交换格式

它本质上是一个模型中间表示格式。

你用 PyTorch 训练出来的模型通常是：

.pth
.pt

这类文件主要适合 PyTorch 使用。

但是公司实际部署时，可能希望模型运行在：

C++
C#
Java
Python 服务
工控机
边缘设备
服务器
TensorRT
OpenVINO
ONNX Runtime
移动端

这时候直接使用 PyTorch 不一定方便。

所以我们会把模型导出成：

.onnx

ONNX 文件里保存的是：

计算图 Graph
模型参数 Weights
算子节点 Nodes
输入输出信息
张量形状 Shape
数据类型 dtype
算子版本 opset

简单说：

ONNX 是一种把不同深度学习框架模型统一起来的中间格式。

## 三、ONNX Runtime 是什么？

ONNX Runtime 简称：

ORT

它是一个运行 ONNX 模型的推理引擎。

你可以这样区分：

ONNX：模型文件格式
ONNX Runtime：执行 ONNX 模型的推理框架

类比一下：

.jpg 是图片格式
图片查看器负责打开 jpg

.onnx 是模型格式
ONNX Runtime 负责运行 onnx

ONNX Runtime 可以在多种硬件上运行模型：

CPU
NVIDIA GPU
TensorRT
OpenVINO
DirectML
CoreML
NNAPI
ROCm

在实际部署里，ONNX Runtime 的作用是：

加载 onnx 模型
↓
根据硬件选择执行后端
↓
优化计算图
↓
执行模型前向推理
↓
返回输出结果

## 四、为什么不用 PyTorch 直接部署？

不是不能用 PyTorch 部署，而是很多公司更喜欢 ONNX / ONNX Runtime。

原因有几个。

### 1. 减少环境依赖

PyTorch 部署需要安装完整 PyTorch 环境。

而 ONNX Runtime 通常更轻量，部署环境更简洁。

### 2. 跨平台

ONNX 模型可以被很多推理框架使用：

ONNX Runtime
TensorRT
OpenVINO
NCNN
MNN
Tengine
OpenCV DNN

这对工业视觉、边缘设备、C++ 项目很重要。

### 3. 跨语言

PyTorch 主要以 Python 为主。

ONNX Runtime 支持：

Python
C++
C#
Java
JavaScript

很多工业软件是 C++ / C# 写的，所以 ONNX Runtime 更容易集成。

### 4. 推理优化更方便

ONNX Runtime 可以做：

图优化
算子融合
常量折叠
CPU/GPU 加速
TensorRT 加速
OpenVINO 加速
INT8 量化
FP16 推理

这些都是实际部署中很常见的优化方式。

## 五、ONNX 模型里有什么？

一个 ONNX 模型可以理解为一张计算图：

input
↓
Conv
↓
BatchNorm
↓
Relu
↓
MaxPool
↓
Flatten
↓
Gemm / MatMul
↓
output

ONNX 模型主要包含：

### 1. Graph 计算图

计算图描述了模型从输入到输出的计算流程。

### 2. Node 节点

每个节点是一个算子，例如：

Conv
Relu
Add
Mul
MatMul
Gemm
Reshape
Transpose
Softmax
Sigmoid
NonMaxSuppression
### 3. Initializer 参数

模型权重保存在 initializer 里。

例如：

卷积核权重
BatchNorm 参数
Linear 层权重
bias
### 4. Input / Output

模型输入输出信息。

例如分类模型输入：

input: [batch, 3, 224, 224]
output: [batch, 1000]

检测模型输入：

images: [batch, 3, 640, 640]
output: [batch, num_predictions, num_attrs]
### 5. Opset

ONNX 的算子集合版本。

比如：

opset 11
opset 12
opset 13
opset 17
opset 18

不同 opset 支持的算子能力不完全一样。

如果模型里有比较新的 PyTorch 算子，可能需要较新的 opset。

常见导出可以用：

opset_version=17

或者：

opset_version=18

如果部署环境比较老，有时需要降低到：

opset_version=11
opset_version=12

## 六、ONNX Runtime 的 Execution Provider

ONNX Runtime 通过 Execution Provider 使用不同硬件。

你可以理解成：

Execution Provider 是 ONNX Runtime 的硬件后端。

常见有：

CPUExecutionProvider
CUDAExecutionProvider
TensorrtExecutionProvider
OpenVINOExecutionProvider
DirectMLExecutionProvider
CoreMLExecutionProvider

例如 CPU 推理：

providers=["CPUExecutionProvider"]

NVIDIA GPU 推理：

providers=["CUDAExecutionProvider", "CPUExecutionProvider"]

TensorRT 推理：

providers=["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

为什么后面通常还要加 CPU？

因为有些算子可能 GPU provider 不支持，这时候可以 fallback 到 CPU。

## 七、最完整的模型部署流程

你要把这个流程背熟：

1. 训练模型
2. 保存权重
3. 加载模型
4. model.eval()
5. 准备 dummy input
6. torch.onnx.export 导出 ONNX
7. onnx.checker 检查模型
8. onnxruntime 加载模型
9. 对比 PyTorch 和 ONNX 输出
10. 编写预处理代码
11. 编写后处理代码
12. 测试单张图片推理
13. 测试批量推理
14. 测试 CPU / GPU 推理速度
15. 集成到业务系统

## 八、PyTorch 导出 ONNX：分类模型示例

下面用手写数字识别 CNN 举例。

### 1. 定义模型
import torch
import torch.nn as nn


class DigitCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

这个模型输入：

[B, 1, 28, 28]

输出：

[B, 10]
### 2. 加载模型权重
model = DigitCNN(num_classes=10)
model.load_state_dict(torch.load("best_digit_cnn.pth", map_location="cpu"))

model.eval()

这里非常重要：

model.eval()

因为 BatchNorm 和 Dropout 在训练和推理时行为不同。

导出 ONNX 前必须切换到推理模式。

### 3. 准备 dummy input
dummy_input = torch.randn(1, 1, 28, 28)

这个 dummy input 不是训练数据，只是给导出器一个示例输入，让它知道模型计算图怎么走。

### 4. 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "digit_cnn.onnx",
    input_names=["input"],
    output_names=["logits"],
    opset_version=17,
    dynamic_axes={
        "input": {0: "batch_size"},
        "logits": {0: "batch_size"}
    }
)

参数解释：

model：PyTorch 模型
dummy_input：示例输入
digit_cnn.onnx：导出的 ONNX 文件
input_names：输入节点名字
output_names：输出节点名字
opset_version：ONNX 算子版本
dynamic_axes：动态维度

## 九、dynamic_axes 是什么？

如果你不设置 dynamic axes，模型可能只能接受固定 batch size。

比如 dummy input 是：

[1, 1, 28, 28]

导出的模型可能固定只能输入 batch=1。

但是部署时你可能想输入：

[1, 1, 28, 28]
[8, 1, 28, 28]
[16, 1, 28, 28]

所以要设置：

dynamic_axes={
    "input": {0: "batch_size"},
    "logits": {0: "batch_size"}
}

含义是：

input 的第 0 维是动态 batch
logits 的第 0 维也是动态 batch

对于图像尺寸固定的模型，通常只让 batch 动态即可。

## 十、检查 ONNX 模型是否合法

导出后要检查：

import onnx

onnx_model = onnx.load("digit_cnn.onnx")
onnx.checker.check_model(onnx_model)

print("ONNX model is valid.")

如果这里报错，说明 ONNX 图可能有问题。

## 十一、查看 ONNX 模型输入输出
import onnxruntime as ort

session = ort.InferenceSession(
    "digit_cnn.onnx",
    providers=["CPUExecutionProvider"]
)

print("Inputs:")
for inp in session.get_inputs():
    print(inp.name, inp.shape, inp.type)

print("Outputs:")
for out in session.get_outputs():
    print(out.name, out.shape, out.type)

输出可能类似：

input ['batch_size', 1, 28, 28] tensor(float)
logits ['batch_size', 10] tensor(float)

面试时你要知道如何获取输入输出名：

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

## 十二、ONNX Runtime 推理
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession(
    "digit_cnn.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

x = np.random.randn(1, 1, 28, 28).astype(np.float32)

outputs = session.run(
    [output_name],
    {input_name: x}
)

logits = outputs[0]
pred = np.argmax(logits, axis=1)

print(pred)

ONNX Runtime 输入必须是：

numpy.ndarray

而且类型通常要是：

np.float32

如果你传 float64，可能报错。

## 十三、PyTorch 和 ONNX 输出一致性验证

这是部署中非常重要的一步。

import torch
import numpy as np
import onnxruntime as ort

model.eval()

x = torch.randn(4, 1, 28, 28)

with torch.no_grad():
    torch_out = model(x).cpu().numpy()

session = ort.InferenceSession(
    "digit_cnn.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
onnx_out = session.run(None, {input_name: x.cpu().numpy()})[0]

diff = np.abs(torch_out - onnx_out)

print("max diff:", diff.max())
print("mean diff:", diff.mean())

一般误差应该很小，比如：

max diff: 1e-5
mean diff: 1e-6

如果误差很大，说明有问题。

常见原因：

1. 模型没有 model.eval()
2. 输入预处理不一致
3. 输入 dtype 不一致
4. 输入 shape 不一致
5. 导出 opset 不合适
6. 模型里有 ONNX 不支持的算子
7. PyTorch 代码里有动态控制流

## 十四、完整 ONNX 分类推理类

这是比较像实际项目的写法：

import cv2
import numpy as np
import onnxruntime as ort


class ONNXDigitClassifier:
    def __init__(self, model_path, providers=None):
        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(
            model_path,
            providers=providers
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def preprocess(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        img = cv2.resize(img, (28, 28))
        img = img.astype(np.float32) / 255.0

        img = img[None, None, :, :]  # [1, 1, 28, 28]

        return img

    def predict(self, image_path):
        x = self.preprocess(image_path)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: x}
        )

        logits = outputs[0]
        pred = int(np.argmax(logits, axis=1)[0])

        return pred


classifier = ONNXDigitClassifier("digit_cnn.onnx")
pred = classifier.predict("test.png")
print("prediction:", pred)

这就是一个完整的分类部署流程。

## 十五、目标检测模型部署为什么更复杂？

图像分类部署比较简单：

输入图像
↓
resize
↓
归一化
↓
模型推理
↓
argmax
↓
类别

目标检测部署复杂很多：

输入图像
↓
letterbox resize
↓
BGR → RGB
↓
归一化
↓
HWC → CHW
↓
增加 batch
↓
ONNX Runtime 推理
↓
解析输出张量
↓
置信度过滤
↓
NMS
↓
坐标还原到原图
↓
画框 / 输出 bbox

所以招聘要求里写 ONNX Runtime，面试很可能会问：

YOLO 导出 ONNX 后怎么推理？
NMS 在模型里还是在外面？
检测框怎么映射回原图？
letterbox 的 pad 怎么处理？

这些才是目标检测部署的重点。

## 十六、YOLO 导出 ONNX

如果你用 Ultralytics YOLO：

from ultralytics import YOLO

model = YOLO("best.pt")

model.export(
    format="onnx",
    imgsz=640,
    opset=17,
    dynamic=True,
    simplify=True
)

通常会生成：

best.onnx

如果是 YOLO-seg，也可以导出：

model.export(
    format="onnx",
    imgsz=640,
    opset=17,
    dynamic=True,
    simplify=True
)

## 十七、YOLO ONNX 推理流程

下面给你一个目标检测部署骨架。

1. letterbox 预处理

YOLO 通常不是简单 resize，而是 letterbox。

import cv2
import numpy as np


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = img.shape[:2]
    new_h, new_w = new_shape

    scale = min(new_w / w, new_h / h)

    resized_w = int(round(w * scale))
    resized_h = int(round(h * scale))

    resized = cv2.resize(img, (resized_w, resized_h))

    pad_w = new_w - resized_w
    pad_h = new_h - resized_h

    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, scale, left, top

为什么要 letterbox？

因为直接 resize 会拉伸图像，改变目标形状。

letterbox 是保持比例缩放，再补边。

2. YOLO ONNX Detector 骨架
import cv2
import numpy as np
import onnxruntime as ort


class YOLOONNXDetector:
    def __init__(self, model_path, input_size=640, conf_thres=0.25, iou_thres=0.45):
        self.input_size = input_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]

        print("input:", self.input_name)
        print("outputs:", self.output_names)

    def preprocess(self, image_path):
        img = cv2.imread(image_path)

        if img is None:
            raise ValueError(f"Cannot read image: {image_path}")

        original = img.copy()

        img, scale, pad_x, pad_y = letterbox(
            img,
            new_shape=(self.input_size, self.input_size)
        )

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0

        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0)   # [1, 3, H, W]

        return img, original, scale, pad_x, pad_y

    def infer(self, image_path):
        x, original, scale, pad_x, pad_y = self.preprocess(image_path)

        outputs = self.session.run(
            self.output_names,
            {self.input_name: x}
        )

        for i, out in enumerate(outputs):
            print(f"output {i} shape:", out.shape)

        return outputs, original, scale, pad_x, pad_y

注意：不同 YOLO 版本导出的 ONNX 输出格式不同。

有的输出：

[1, 84, 8400]

有的输出：

[1, 8400, 84]

有的导出时已经带 NMS，有的不带。

所以你要根据输出 shape 写后处理。

## 十八、NMS 是什么？

NMS 全称：

Non-Maximum Suppression

中文叫：

非极大值抑制

目标检测模型经常会对同一个物体预测多个框。

例如一个人被预测出 5 个相近 bbox：

box1 score 0.95
box2 score 0.91
box3 score 0.88
...

NMS 的作用是：

保留最高分框
删除与它重叠太高的其他框

流程：

按置信度排序
↓
取最高分框
↓
删除 IoU 超过阈值的其他框
↓
继续处理剩余框

部署检测模型时，NMS 很重要。

## 十九、坐标还原为什么重要？

YOLO 输入前做了 letterbox：

原图尺寸：1280 × 720
↓
缩放 + padding
↓
模型输入：640 × 640

模型输出的 bbox 是相对于 640×640 的。

你要把它还原到原图：

x1 = (x1 - pad_x) / scale
y1 = (y1 - pad_y) / scale
x2 = (x2 - pad_x) / scale
y2 = (y2 - pad_y) / scale

否则画出来的框会偏。

这是检测部署里很常见的坑。

## 二十、图像分割模型部署

分割模型常见输出是 mask。

例如 U-Net 输出：

[B, C, H, W]

或者二分类分割：

[B, 1, H, W]

后处理一般是：

sigmoid / softmax
↓
threshold / argmax
↓
resize 回原图大小
↓
形态学后处理
↓
连通域分析
↓
轮廓提取

二分类分割示例：

mask_prob = 1 / (1 + np.exp(-logits))
mask = (mask_prob > 0.5).astype(np.uint8)

多分类分割：

pred = np.argmax(logits, axis=1)

如果部署工业缺陷分割，还经常会加：

开运算去小噪声
闭运算填洞
连通域过滤
面积阈值筛选
轮廓提取
缺陷面积统计

## 二十一、ONNX Runtime SessionOptions

ONNX Runtime 可以设置 SessionOptions 进行优化：

import onnxruntime as ort

session_options = ort.SessionOptions()

session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session_options.intra_op_num_threads = 4
session_options.inter_op_num_threads = 1

session = ort.InferenceSession(
    "model.onnx",
    sess_options=session_options,
    providers=["CPUExecutionProvider"]
)

几个参数：

graph_optimization_level：图优化等级
intra_op_num_threads：单个算子内部线程数
inter_op_num_threads：算子之间并行线程数

CPU 部署时，线程数对性能影响较大。

## 二十二、ONNX Runtime GPU 推理

安装 GPU 版本：

pip install onnxruntime-gpu

加载模型：

session = ort.InferenceSession(
    "model.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

检查 provider：

print(session.get_providers())

如果输出：

['CUDAExecutionProvider', 'CPUExecutionProvider']

说明 GPU provider 可用。

如果只有：

['CPUExecutionProvider']

说明 GPU 没用上。

常见原因：

1. 没安装 onnxruntime-gpu
2. CUDA 版本不匹配
3. cuDNN 不匹配
4. provider 写错
5. 环境没有 NVIDIA GPU

## 二十三、TensorRT 加速

在 NVIDIA GPU 上，ONNX 可以进一步用 TensorRT 加速。

流程：

PyTorch
↓
ONNX
↓
TensorRT Engine
↓
高性能推理

ONNX Runtime 也可以使用 TensorRTExecutionProvider：

providers = [
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "CPUExecutionProvider"
]

session = ort.InferenceSession(
    "model.onnx",
    providers=providers
)

TensorRT 常用于：

YOLO 检测部署
实时工业视觉
自动驾驶感知
边缘 GPU 设备
高吞吐服务器推理

优点：

速度快
支持 FP16 / INT8
算子融合优化强

缺点：

环境配置复杂
动态 shape 处理更麻烦
某些算子不支持
首次构建 engine 可能较慢

## 二十四、OpenVINO 加速

OpenVINO 是 Intel 的推理优化工具链，常用于：

Intel CPU
Intel 核显
Intel VPU
工控机
边缘设备

很多工业视觉工控机没有 NVIDIA GPU，而是 Intel CPU。

这时候可以：

ONNX → OpenVINO

或者用 ONNX Runtime 的 OpenVINOExecutionProvider。

适合：

工业相机检测
CPU 实时推理
边缘缺陷检测

## 二十五、量化 Quantization

量化是部署中非常重要的优化手段。

普通模型一般是：

FP32

量化后可以变成：

FP16
INT8
1. FP16

FP16 是半精度浮点。

优点：

速度更快
显存更低
精度损失通常较小
适合 NVIDIA GPU

YOLO 导出时可以指定：

model.export(format="onnx", half=True)

但注意 CPU 通常不适合 FP16。

2. INT8

INT8 是 8 位整数。

优点：

模型更小
速度更快
内存占用更低
适合边缘设备

缺点：

可能有精度损失
需要校准数据
某些模型量化后精度下降明显

ONNX Runtime 支持动态量化和静态量化。

动态量化示例：

from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input="model.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8
)

动态量化通常更适合 Linear / Transformer 类模型，对 CNN 加速不一定明显。

CNN / 检测模型更常用静态量化，需要校准数据。

## 二十六、ONNX Simplifier

有时候导出的 ONNX 模型会有很多冗余节点。

可以使用：

pip install onnxsim

然后：

from onnxsim import simplify
import onnx

model = onnx.load("model.onnx")
model_simp, check = simplify(model)

assert check

onnx.save(model_simp, "model_simplified.onnx")

Ultralytics 导出时也可以：

model.export(format="onnx", simplify=True)

简化后可能：

推理更快
图结构更清晰
兼容性更好

## 二十七、常见部署坑
1. 没有 model.eval()

表现：

PyTorch 和 ONNX 输出不一致
BatchNorm / Dropout 行为异常

解决：

model.eval()
2. 输入 dtype 错误

ONNX Runtime 通常要：

np.float32

如果你传：

np.float64

可能报错：

Unexpected input data type
3. 输入维度顺序错了

OpenCV 读取是：

HWC

PyTorch / ONNX 模型通常要：

NCHW

所以要：

img = np.transpose(img, (2, 0, 1))
img = np.expand_dims(img, axis=0)
4. BGR / RGB 搞反

OpenCV 默认读取：

BGR

模型训练时通常使用：

RGB

所以需要：

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
5. 归一化不一致

训练时如果用了：

/255.0
mean/std normalize

部署时也必须一致。

例如 ImageNet 模型通常需要：

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

如果忘了，精度会明显下降。

6. 检测框坐标没还原

YOLO letterbox 后必须把坐标映射回原图。

否则框会偏移。

7. 动态输入没设置

导出时没设置 dynamic axes，部署时换 batch 或分辨率就报错。

8. Opset 不兼容

模型里某些算子在低 opset 不支持。

可以尝试提高：

opset_version=17

或者根据部署环境降低。

9. 算子不支持

有些 PyTorch 操作不能很好导出 ONNX。

解决方法：

替换模型中的特殊算子
避免 Python 控制流
使用 torch.onnx.export 支持的算子
自定义 symbolic
或者改模型结构

## 二十八、工业视觉部署中的典型 ONNX Runtime 架构

假设你做工业缺陷检测，系统可能是：

工业相机
↓
图像采集 SDK
↓
OpenCV 图像预处理
↓
ONNX Runtime 推理
↓
检测/分割后处理
↓
缺陷判断
↓
保存 NG 图像
↓
生成检测日志
↓
PLC / 上位机通信

## 二十九、面试问答准备
问题 1：ONNX 和 ONNX Runtime 区别？

回答：

ONNX 是一种开放的模型交换格式，用于保存模型的计算图、参数、输入输出和算子信息；ONNX Runtime 是执行 ONNX 模型的推理引擎，可以根据不同 Execution Provider 在 CPU、CUDA、TensorRT、OpenVINO 等硬件后端上运行模型。

问题 2：PyTorch 模型怎么导出 ONNX？

回答：

首先加载训练好的模型并切换到 model.eval()，然后准备一个和真实输入形状一致的 dummy input，使用 torch.onnx.export 导出模型。导出时需要指定 input_names、output_names、opset_version。如果需要支持动态 batch，则设置 dynamic_axes。导出后用 onnx.checker.check_model 检查模型合法性，再用 ONNX Runtime 推理并和 PyTorch 输出对比误差。

问题 3：为什么要验证 PyTorch 和 ONNX 输出一致性？

回答：

因为模型导出过程中可能出现算子转换差异、输入预处理不一致、BatchNorm/Dropout 模式错误等问题。如果 ONNX 输出和 PyTorch 输出差异很大，说明部署模型可能不可靠。通常会用同一个输入分别跑 PyTorch 和 ONNX Runtime，比较 max abs diff 和 mean abs diff，确保误差在可接受范围内。

问题 4：ONNX Runtime 如何使用 GPU？

回答：

需要安装 onnxruntime-gpu，然后创建 InferenceSession 时指定 CUDAExecutionProvider，通常写成 providers=["CUDAExecutionProvider", "CPUExecutionProvider"]。可以通过 session.get_providers() 检查实际使用的 provider。如果没有 CUDA provider，可能是 CUDA/cuDNN 版本不匹配或安装的是 CPU 版 onnxruntime。

问题 5：YOLO 部署的难点是什么？

回答：

YOLO 部署难点主要在预处理和后处理。预处理要保证和训练一致，包括 letterbox resize、BGR 转 RGB、归一化、HWC 转 CHW。后处理要解析模型输出，进行置信度过滤、NMS，然后把 bbox 从模型输入尺寸映射回原图尺寸。不同 YOLO 版本导出的 ONNX 输出格式可能不同，所以需要根据输出 shape 编写对应解析逻辑。

问题 6：ONNX 模型怎么优化？

回答：

可以从几个方向优化。第一是使用 ONNX Runtime 的图优化；第二是选择合适的 Execution Provider，例如 CPU、CUDA、TensorRT、OpenVINO；第三是使用 FP16 或 INT8 量化减少计算和内存；第四是固定输入尺寸或合理设置动态 batch；第五是减少预处理和后处理耗时，比如优化 NMS、使用批量推理或异步流水线。

## 三十、你应该怎么练才能达到招聘要求？

我建议你按 4 个小项目练。

项目 1：手写数字 CNN 导出 ONNX

目标：

训练 CNN
导出 ONNX
ONNX Runtime 推理
生成 Kaggle submission.csv

你要掌握：

torch.onnx.export
dynamic_axes
onnx.checker
onnxruntime.InferenceSession
输出一致性验证
项目 2：图像分类模型部署

用 ResNet / ConvNeXt 做分类。

流程：

训练分类模型
↓
导出 ONNX
↓
OpenCV 读取图片
↓
resize + normalize
↓
ORT 推理
↓
输出类别
项目 3：YOLO 目标检测 ONNX 部署

流程：

YOLO best.pt
↓
export best.onnx
↓
OpenCV letterbox
↓
ORT inference
↓
NMS
↓
坐标还原
↓
画检测框

这是最适合视觉岗位的。

项目 4：U-Net / YOLO-seg 分割部署

流程：

分割模型
↓
导出 ONNX
↓
ORT 推理 mask
↓
threshold / argmax
↓
resize 回原图
↓
OpenCV 连通域过滤
↓
缺陷面积统计

这对工业视觉很有用。

## 三十、你应该怎么练才能达到招聘要求？

我建议你按 4 个小项目练。

项目 1：手写数字 CNN 导出 ONNX

目标：

训练 CNN
导出 ONNX
ONNX Runtime 推理
生成 Kaggle submission.csv

你要掌握：

torch.onnx.export
dynamic_axes
onnx.checker
onnxruntime.InferenceSession
输出一致性验证
项目 2：图像分类模型部署

用 ResNet / ConvNeXt 做分类。

流程：

训练分类模型
↓
导出 ONNX
↓
OpenCV 读取图片
↓
resize + normalize
↓
ORT 推理
↓
输出类别
项目 3：YOLO 目标检测 ONNX 部署

流程：

YOLO best.pt
↓
export best.onnx
↓
OpenCV letterbox
↓
ORT inference
↓
NMS
↓
坐标还原
↓
画检测框

这是最适合视觉岗位的。

项目 4：U-Net / YOLO-seg 分割部署

流程：

分割模型
↓
导出 ONNX
↓
ORT 推理 mask
↓
threshold / argmax
↓
resize 回原图
↓
OpenCV 连通域过滤
↓
缺陷面积统计

这对工业视觉很有用。