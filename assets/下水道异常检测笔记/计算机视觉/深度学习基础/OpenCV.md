# OpenCV

下面按知识总结中常见的 OpenCV 能力要求来系统介绍。可以把 OpenCV 理解成一个图像/视频处理工具箱，主要覆盖：

图像读取与显示
图像基础操作
颜色空间转换
图像滤波与增强
阈值分割
形态学处理
边缘与轮廓分析
几何变换
特征点检测与匹配
目标检测辅助处理
视频读取与处理
相机标定与透视变换
传统视觉 + 深度学习部署

如果技术呈现要求写“熟练掌握 OpenCV”，通常不是要求记住所有 API，而是要求能用 OpenCV 完成常见视觉任务，比如：

读取图像和视频
预处理图像
提取目标区域
做边缘、轮廓、形状分析
做简单目标检测和定位
做图像增强
做透视矫正
做相机标定
处理摄像头视频流
配合深度学习模型做输入输出处理

## 一、OpenCV 是什么？

OpenCV，全称是：

Open Source Computer Vision Library

中文可以理解为：

开源计算机视觉库

它最常用于：

图像处理
视频处理
传统机器视觉
摄像头采集
目标检测前后处理
工业视觉
自动驾驶感知预处理
AR / 相机标定
OCR 图像预处理
医学图像处理

在实际项目中，OpenCV 常常承担两类角色：

1. 传统视觉算法主体
2. 深度学习模型的前处理和后处理工具

比如 YOLO 检测项目里，OpenCV 可能负责：

读取图像
resize
BGR 转 RGB
归一化
画检测框
NMS
保存结果
读取视频逐帧推理

## 二、OpenCV 的常见模块

OpenCV 常见模块可以这样分：

模块	作用
core	矩阵、数组、基础数据结构
imgcodecs	图像读取和保存
highgui	图像窗口显示、鼠标键盘交互
imgproc	图像处理核心模块
videoio	摄像头和视频读取保存
video	光流、背景建模等视频分析
features2d	特征点检测与匹配
calib3d	相机标定、三维重建、几何估计
objdetect	Haar、HOG 等传统目标检测
dnn	深度学习模型加载和推理
ml	传统机器学习
stitching	图像拼接
photo	图像修复、去噪、HDR 等

其中知识总结最常问的是：

imgcodecs
highgui
imgproc
videoio
features2d
calib3d
dnn

## 三、图像读取、显示、保存

这是 OpenCV 最基础的部分。

### 1. 读取图像：cv2.imread
import cv2

img = cv2.imread("image.jpg")

作用：

从磁盘读取图像

注意：

OpenCV 默认读取的是 BGR 格式，不是 RGB

这点非常重要。

如果用 matplotlib 显示 OpenCV 读取的图像，颜色可能会不对，因为 matplotlib 习惯 RGB。

### 2. 显示图像：cv2.imshow
cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

作用：

弹出窗口显示图像

解释：

cv2.imshow("image", img)

显示图像。

cv2.waitKey(0)

等待键盘输入。参数为 0 表示一直等待。

cv2.destroyAllWindows()

关闭所有窗口。

### 3. 保存图像：cv2.imwrite
cv2.imwrite("result.jpg", img)

作用：

把图像保存到磁盘

常见应用：

保存检测结果图
保存预处理后的图
保存分割 mask
保存标注可视化结果
### 4. 知识总结注意点

核心问题：
OpenCV 读取图像是什么颜色格式？

核心说明：
OpenCV 的 cv2.imread 默认读取的是 BGR 格式，而不是 RGB。如果要和 matplotlib、PIL、深度学习模型配合，通常需要用 cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 转换。

## 四、颜色空间转换

常用函数：

cv2.cvtColor()
### 1. BGR 转 RGB
rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

应用场景：

OpenCV 图像转给 matplotlib 显示
OpenCV 图像转给深度学习模型
和 PIL / torchvision 配合
### 2. BGR 转灰度图
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

应用场景：

边缘检测
阈值分割
轮廓检测
角点检测
模板匹配
传统图像处理

很多传统算法只需要单通道灰度图。

### 3. BGR 转 HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

HSV 包含：

H：色相
S：饱和度
V：亮度

常用于颜色分割。

例如检测红色目标、蓝色目标、绿色目标。

lower = (35, 50, 50)
upper = (85, 255, 255)

mask = cv2.inRange(hsv, lower, upper)

这个例子可以提取绿色区域。

### 4. BGR 转 LAB
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

LAB 常用于：

亮度增强
色彩校正
CLAHE 处理亮度通道

比如低光增强时，可以只增强 L 通道。

## 五、图像尺寸、通道和基础操作
### 1. 查看图像尺寸
h, w, c = img.shape

含义：

h：高度
w：宽度
c：通道数

灰度图只有二维：

h, w = gray.shape
### 2. 裁剪 ROI
roi = img[y1:y2, x1:x2]

注意顺序是：

图像数组索引：img[y, x]
不是 img[x, y]

应用场景：

裁剪目标区域
检测框后处理
局部图像分析
OCR 区域提取
人脸区域提取
### 3. 图像复制
copy_img = img.copy()

如果需要在图上画框，可复制一份，避免修改原图。

### 4. 拆分和合并通道
b, g, r = cv2.split(img)
merged = cv2.merge([b, g, r])

应用场景：

单独处理某个颜色通道
图像增强
颜色分析

## 六、图像缩放、翻转、旋转
### 1. resize 缩放
resized = cv2.resize(img, (640, 640))

注意参数顺序：

cv2.resize(img, (width, height))

不是 (height, width)。

常见应用：

深度学习模型输入尺寸统一
图像预处理
视频帧缩放
加快处理速度

例如 YOLO 常需要：

640 × 640
### 2. 翻转 cv2.flip
flip_horizontal = cv2.flip(img, 1)
flip_vertical = cv2.flip(img, 0)
flip_both = cv2.flip(img, -1)

参数含义：

1：水平翻转
0：垂直翻转
-1：水平 + 垂直翻转

应用场景：

数据增强
镜像摄像头画面
图像修正
### 3. 旋转

简单旋转：

rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

自定义角度旋转：

h, w = img.shape[:2]
center = (w // 2, h // 2)

M = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated = cv2.warpAffine(img, M, (w, h))

应用场景：

图像矫正
数据增强
文档方向校正
目标姿态归一化

## 七、图像滤波和平滑

滤波主要用于：

去噪
平滑
减少细节干扰
增强后续边缘/轮廓稳定性
### 1. 均值滤波 cv2.blur
blur = cv2.blur(img, (5, 5))

作用：

对邻域像素取平均

优点：

简单快速

缺点：

容易模糊边缘
### 2. 高斯滤波 cv2.GaussianBlur
gaussian = cv2.GaussianBlur(img, (5, 5), 0)

作用：

用高斯核进行平滑

应用场景：

边缘检测前去噪
图像预处理
减少高频噪声

Canny 边缘检测前经常会先做高斯滤波。

### 3. 中值滤波 cv2.medianBlur
median = cv2.medianBlur(img, 5)

作用：

用邻域中值替代当前像素

特别适合去除：

椒盐噪声
黑白点噪声
### 4. 双边滤波 cv2.bilateralFilter
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

特点：

平滑图像的同时保留边缘

应用场景：

美颜磨皮
边缘保留去噪
图像风格化预处理

缺点：

速度比均值/高斯滤波慢

## 八、图像锐化

锐化用于增强边缘和细节。

可以用卷积核：

import numpy as np

kernel = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])

sharp = cv2.filter2D(img, -1, kernel)

应用场景：

增强边缘
增强文字清晰度
工业缺陷图像增强
OCR 前处理

但要注意：

锐化会放大噪声

## 九、阈值分割

阈值分割用于把灰度图变成二值图。

### 1. 固定阈值 cv2.threshold
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

含义：

像素值 > 127 → 255
像素值 <= 127 → 0

应用场景：

前景背景分割
文字分割
二值 mask 生成
轮廓检测前处理
### 2. 反向二值化
ret, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

含义：

像素值 > 127 → 0
像素值 <= 127 → 255
### 3. Otsu 自动阈值
ret, otsu = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

Otsu 会自动寻找一个合适阈值。

适合：

前景和背景灰度分布比较明显的图像

应用：

文档二值化
工业零件分割
显微图像目标提取
### 4. 自适应阈值 cv2.adaptiveThreshold
adaptive = cv2.adaptiveThreshold(
    gray,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

适合光照不均匀的图像。

比如：

一张文档左边暗右边亮
普通固定阈值效果差
自适应阈值更稳定

应用场景：

文档扫描
票据识别
车牌字符分割
OCR 前处理

## 十、形态学操作

形态学主要处理二值图，用于：

去噪
填洞
连接断裂区域
分离粘连区域
提取结构

常用函数：

cv2.erode()
cv2.dilate()
cv2.morphologyEx()
### 1. 腐蚀 erode
kernel = np.ones((3, 3), np.uint8)
eroded = cv2.erode(binary, kernel, iterations=1)

作用：

让白色区域变小

应用：

去除小白点噪声
分离粘连目标
缩小前景
### 2. 膨胀 dilate
dilated = cv2.dilate(binary, kernel, iterations=1)

作用：

让白色区域变大

应用：

填补小黑洞
连接断裂目标
扩大前景区域
### 3. 开运算
opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

开运算：

先腐蚀，再膨胀

作用：

去除小白点噪声
### 4. 闭运算
closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

闭运算：

先膨胀，再腐蚀

作用：

填补小黑洞
连接小裂缝
### 5. 形态学梯度
gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

作用：

提取边界
### 6. 顶帽和黑帽

顶帽：

tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

用于提取亮的小结构。

黑帽：

blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

用于提取暗的小结构。

应用场景：

缺陷检测
文字增强
不均匀光照校正
工业表面检测

## 十一、边缘检测
### 1. Sobel 算子
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

作用：

检测水平或垂直方向梯度

参数：

dx=1, dy=0：检测 x 方向梯度
dx=0, dy=1：检测 y 方向梯度

应用：

边缘增强
梯度分析
图像纹理分析
### 2. Laplacian 算子
lap = cv2.Laplacian(gray, cv2.CV_64F)

作用：

二阶导数边缘检测

对噪声敏感，所以常先高斯滤波。

### 3. Canny 边缘检测
edges = cv2.Canny(gray, 100, 200)

Canny 是最常见边缘检测函数。

流程大致是：

高斯滤波去噪
计算梯度
非极大值抑制
双阈值检测
边缘连接

参数：

100：低阈值
200：高阈值

应用场景：

轮廓检测前处理
车道线检测
文档边缘检测
工业零件边缘检测
目标形状提取

知识总结常问：

Canny 两个阈值有什么作用？

核心说明：

高阈值用于确定强边缘，低阈值用于保留和强边缘相连的弱边缘。这样可以减少噪声边缘，同时保留连续边界。

## 十二、轮廓检测

轮廓是 OpenCV 知识总结非常常见内容。

### 1. 查找轮廓 cv2.findContours
contours, hierarchy = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

作用：

从二值图中提取目标轮廓

常见参数：

cv2.RETR_EXTERNAL：只检测最外层轮廓
cv2.RETR_TREE：检测所有轮廓，并建立层级关系
cv2.CHAIN_APPROX_SIMPLE：压缩轮廓点
cv2.CHAIN_APPROX_NONE：保留所有轮廓点

应用场景：

目标计数
尺寸测量
形状分析
缺陷检测
字符分割
零件定位
### 2. 绘制轮廓 cv2.drawContours
result = img.copy()
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

参数：

-1：绘制所有轮廓
(0,255,0)：绿色
2：线宽
### 3. 轮廓面积 cv2.contourArea
area = cv2.contourArea(contour)

应用：

过滤小噪声
筛选目标
计算目标面积

例如：

filtered = [c for c in contours if cv2.contourArea(c) > 100]
### 4. 轮廓周长 cv2.arcLength
perimeter = cv2.arcLength(contour, True)

第二个参数：

True：闭合轮廓
False：非闭合曲线
### 5. 外接矩形 cv2.boundingRect
x, y, w, h = cv2.boundingRect(contour)
cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 2)

应用：

目标框定位
字符框提取
检测结果可视化
### 6. 最小外接旋转矩形 cv2.minAreaRect
rect = cv2.minAreaRect(contour)
box = cv2.boxPoints(rect)
box = np.intp(box)

作用：

得到可以旋转的最小外接矩形

应用场景：

倾斜物体检测
工业零件角度测量
文本框倾斜矫正
### 7. 最小外接圆 cv2.minEnclosingCircle
(x, y), radius = cv2.minEnclosingCircle(contour)

应用：

圆形目标检测
颗粒检测
细胞检测
球状物体定位
### 8. 多边形逼近 cv2.approxPolyDP
epsilon = 0.02 * cv2.arcLength(contour, True)
approx = cv2.approxPolyDP(contour, epsilon, True)

作用：

把复杂轮廓近似成多边形

应用：

检测三角形
检测矩形
文档四边形检测
二维码定位

如果 len(approx) == 4，可能是矩形或四边形。

### 9. 凸包 cv2.convexHull
hull = cv2.convexHull(contour)

作用：

得到包围轮廓的最小凸多边形

应用：

手势识别
形状分析
缺陷检测
目标外形分析

## 十三、连通域分析

轮廓检测之外，连通域分析也很常用。

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

返回：

num_labels：连通域数量
labels：每个像素属于哪个连通域
stats：每个连通域的 bbox、面积等
centroids：每个连通域中心点

应用场景：

目标计数
小区域过滤
分割 mask 后处理
去除小噪声
连通区域统计

例如分割模型输出 mask 后，可以去除小连通域：

num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)

for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    if area > 100:
        # 保留该区域
        pass

这在医学分割、缺陷分割、工业视觉中非常常见。

## 十四、几何变换
### 1. 仿射变换 cv2.warpAffine

仿射变换包括：

平移
旋转
缩放
剪切

示例：

M = np.float32([
    [1, 0, 50],
    [0, 1, 30]
])

shifted = cv2.warpAffine(img, M, (w, h))

这个表示平移：

x 方向移动 50
y 方向移动 30
### 2. 透视变换 cv2.warpPerspective

透视变换用于把一个四边形区域拉正。

src = np.float32([
    [100, 100],
    [400, 100],
    [450, 500],
    [80, 500]
])

dst = np.float32([
    [0, 0],
    [300, 0],
    [300, 400],
    [0, 400]
])

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, (300, 400))

应用场景：

文档扫描矫正
车牌透视矫正
棋盘格矫正
鸟瞰图转换
自动驾驶 BEV 视角变换

知识总结可以这样说：

仿射变换保持平行关系，透视变换可以处理近大远小的投影变化，常用于文档矫正和视角变换。

## 十五、直方图与图像增强
### 1. 灰度直方图 cv2.calcHist
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

作用：

统计灰度值分布

应用：

分析亮度
判断曝光
图像增强
阈值选择
### 2. 直方图均衡化 cv2.equalizeHist
equalized = cv2.equalizeHist(gray)

作用：

增强图像对比度

适合：

灰度图对比度低
光照偏暗

缺点：

可能放大噪声
可能导致局部过增强
### 3. CLAHE 自适应直方图均衡
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)

CLAHE 比全局直方图均衡更稳定。

应用：

医学图像增强
低光图像增强
工业表面缺陷增强
OCR 图像增强

彩色图一般不直接对 BGR 三通道做 CLAHE，而是转到 LAB，只增强 L 通道：

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l2 = clahe.apply(l)

lab2 = cv2.merge([l2, a, b])
enhanced = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

## 十六、模板匹配

函数：

cv2.matchTemplate()

示例：

result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

作用：

在大图中查找和模板最相似的位置

应用场景：

UI 自动化
固定图案查找
工业零件定位
简单目标检测
游戏图像识别

缺点：

对尺度变化、旋转变化、光照变化敏感

如果目标大小和方向变化很大，模板匹配效果会变差。

## 十七、霍夫变换
### 1. 直线检测 cv2.HoughLinesP
lines = cv2.HoughLinesP(
    edges,
    rho=1,
    theta=np.pi / 180,
    threshold=80,
    minLineLength=50,
    maxLineGap=10
)

应用：

车道线检测
表格线检测
文档边缘检测
工业直线结构检测
### 2. 圆检测 cv2.HoughCircles
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50,
    param1=100,
    param2=30,
    minRadius=10,
    maxRadius=100
)

应用：

圆形零件检测
硬币检测
瞳孔检测
细胞圆形区域检测

## 十八、特征点检测与匹配

常见模块：

SIFT
ORB
AKAZE
BRISK
BFMatcher
FLANN
### 1. ORB 特征
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)

返回：

kp：关键点
des：描述子

ORB 优点：

速度快
免费
适合实时任务

应用：

图像匹配
目标定位
SLAM 前端
全景拼接
视觉定位
### 2. SIFT 特征
sift = cv2.SIFT_create()
kp, des = sift.detectAndCompute(gray, None)

SIFT 特点：

尺度不变
旋转不变
鲁棒性强

应用：

图像配准
物体识别
三维重建
全景拼接

缺点：

比 ORB 慢
### 3. 特征匹配 BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

如果是 ORB，通常用：

NORM_HAMMING

如果是 SIFT，通常用：

NORM_L2
### 4. FLANN 匹配
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

常配合 Lowe's ratio test：

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

作用：

过滤不可靠匹配点
### 5. 单应性矩阵 cv2.findHomography
H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

应用：

图像配准
平面目标定位
全景拼接
文档矫正
AR 标记定位

RANSAC 作用：

剔除错误匹配点
鲁棒估计几何变换

## 十九、视频读取与保存
### 1. 读取摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

应用：

摄像头实时检测
人脸识别
工业相机采集
视频流处理
### 2. 读取视频文件
cap = cv2.VideoCapture("video.mp4")

逐帧读取：

ret, frame = cap.read()

ret 表示是否成功读取。

### 3. 获取视频属性
fps = cap.get(cv2.CAP_PROP_FPS)
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

应用：

视频处理
帧率控制
视频保存
推理速度统计
### 4. 保存视频 cv2.VideoWriter
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps, (width, height))

out.write(frame)
out.release()

应用：

保存检测结果视频
保存跟踪结果视频
保存处理后的视频流

## 二十、背景建模与运动检测
### 1. MOG2 背景建模
fgbg = cv2.createBackgroundSubtractorMOG2()

fgmask = fgbg.apply(frame)

作用：

分离运动前景和背景

应用场景：

监控运动检测
入侵检测
车辆检测
行人移动检测
### 2. KNN 背景建模
fgbg = cv2.createBackgroundSubtractorKNN()
fgmask = fgbg.apply(frame)

和 MOG2 类似，用于前景检测。

## 二十一、光流 Optical Flow

光流用于估计相邻帧之间像素运动。

### 1. Lucas-Kanade 稀疏光流
next_pts, status, err = cv2.calcOpticalFlowPyrLK(
    prev_gray,
    next_gray,
    prev_pts,
    None
)

应用：

目标跟踪
运动估计
视频稳定
特征点跟踪
SLAM
### 2. Farneback 稠密光流
flow = cv2.calcOpticalFlowFarneback(
    prev_gray,
    next_gray,
    None,
    0.5,
    3,
    15,
    3,
    5,
    1.2,
    0
)

输出每个像素的运动向量。

应用：

运动分析
视频增强
动作检测
视频稳定
光流可视化

## 二十二、相机标定与畸变矫正

这是工业视觉、机器人、AR、自动驾驶中很重要的模块。

### 1. 棋盘格角点检测
ret, corners = cv2.findChessboardCorners(gray, (9, 6))

作用：

检测棋盘格内角点

用于相机标定。

### 2. 相机标定 cv2.calibrateCamera
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

输出：

camera_matrix：相机内参矩阵
dist_coeffs：畸变参数
rvecs：旋转向量
tvecs：平移向量

应用：

相机内参估计
畸变矫正
三维测量
机器人视觉
AR 定位
### 3. 畸变矫正 cv2.undistort
undistorted = cv2.undistort(img, camera_matrix, dist_coeffs)

作用：

矫正镜头畸变

例如广角镜头边缘会弯曲，可以用这个矫正。

### 4. solvePnP 位姿估计
ret, rvec, tvec = cv2.solvePnP(
    object_points,
    image_points,
    camera_matrix,
    dist_coeffs
)

作用：

根据 3D 点和 2D 图像点，估计相机或物体位姿

应用：

AR 标记定位
机器人抓取
相机姿态估计
三维测量

## 二十三、OpenCV DNN 模块

OpenCV 可以加载一些深度学习模型进行推理。

常用函数：

cv2.dnn.readNet()
cv2.dnn.blobFromImage()
cv2.dnn.NMSBoxes()
### 1. blobFromImage
blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1/255.0,
    size=(640, 640),
    mean=(0, 0, 0),
    swapRB=True,
    crop=False
)

作用：

把图像转换成神经网络输入格式

常见处理包括：

resize
归一化
减均值
BGR 转 RGB
HWC 转 NCHW
增加 batch 维度
### 2. 加载模型
net = cv2.dnn.readNet("model.onnx")

可以加载：

ONNX
Caffe
TensorFlow
Darknet
### 3. 前向推理
net.setInput(blob)
outputs = net.forward()

应用：

部署 YOLO
部署分类模型
部署分割模型
轻量推理
### 4. NMSBoxes
indices = cv2.dnn.NMSBoxes(
    boxes,
    confidences,
    score_threshold=0.25,
    nms_threshold=0.45
)

作用：

非极大值抑制，去除重复检测框

这是目标检测后处理中很常见的函数。

## 二十四、绘图函数

OpenCV 常用于结果可视化。

### 1. 画矩形框
cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

应用：

目标检测框
ROI 标记
人脸框
缺陷框
### 2. 画圆
cv2.circle(img, (x, y), 5, (0, 0, 255), -1)

应用：

关键点显示
中心点显示
角点显示
### 3. 画线
cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

应用：

车道线
测量线
轨迹线
### 4. 写文字
cv2.putText(
    img,
    "person 0.95",
    (x, y),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

应用：

检测类别
置信度
目标数量
FPS 显示

## 二十五、OpenCV 在深度学习项目中的典型作用

即使做的是 YOLO、U-Net、SAM、Mask R-CNN，OpenCV 仍然很常用。

### 1. 图像分类项目

OpenCV 用于：

读取图像
resize
颜色转换
归一化
数据增强
可视化 Grad-CAM
### 2. 目标检测项目

OpenCV 用于：

读取图片和视频
resize / letterbox
画 bbox
NMS
裁剪目标区域
保存检测结果
计算 FPS
视频逐帧推理
### 3. 图像分割项目

OpenCV 用于：

读取 mask
resize mask
阈值化 mask
连通域过滤
轮廓提取
mask 可视化
边界绘制
形态学后处理

例如分割模型输出 mask 后：

mask = (pred > 0.5).astype(np.uint8) * 255

kernel = np.ones((3, 3), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

这就是典型后处理。

### 4. OCR 项目

OpenCV 用于：

灰度化
去噪
二值化
形态学处理
文字区域检测
透视矫正
字符切分
### 5. 工业视觉项目

OpenCV 用于：

缺陷增强
阈值分割
轮廓检测
面积筛选
尺寸测量
相机标定
畸变矫正
模板匹配
缺陷框可视化

## 二十六、一个典型 OpenCV 工业检测流程

假设需要检测零件表面缺陷，传统 OpenCV 流程可能是：

读取图像
↓
灰度化
↓
高斯滤波去噪
↓
CLAHE 增强对比度
↓
阈值分割
↓
形态学开闭运算
↓
轮廓检测
↓
按面积/长宽比过滤
↓
画出缺陷框
↓
输出缺陷数量和位置

代码结构：

import cv2
import numpy as np

img = cv2.imread("defect.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(blur)

_, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

result = img.copy()

for c in contours:
    area = cv2.contourArea(c)
    if area < 100:
        continue

    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(result, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imwrite("result.jpg", result)

知识总结中能讲出这个流程，就说明不是只会调包，而是理解传统视觉 pipeline。

## 二十七、知识总结中 OpenCV 高频问题
问题 1：OpenCV 读取图像默认是什么格式？

核心说明：
OpenCV 使用 cv2.imread 读取图像时默认是 BGR 格式，而不是 RGB。如果要用 matplotlib 显示或者送入某些深度学习模型，通常需要用 cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 转换。

问题 2：阈值分割有哪些方式？

核心说明：
常见有固定阈值、Otsu 自动阈值和自适应阈值。固定阈值适合光照稳定的场景，Otsu 适合前景背景灰度分布比较明显的场景，自适应阈值适合光照不均匀的图像，比如文档扫描和票据识别。

问题 3：开运算和闭运算有什么区别？

核心说明：
开运算是先腐蚀再膨胀，主要用于去除小的白色噪声；闭运算是先膨胀再腐蚀，主要用于填补前景中的小黑洞或连接断裂区域。

问题 4：findContours 输入应该是什么图？

核心说明：
findContours 通常输入二值图，而不是彩色图。一般会先灰度化，再阈值分割或 Canny 边缘检测，然后再找轮廓。

问题 5：Canny 边缘检测流程是什么？

核心说明：
Canny 通常包括高斯滤波去噪、计算梯度幅值和方向、非极大值抑制、双阈值检测和边缘连接。两个阈值分别用于确定强边缘和弱边缘，弱边缘只有和强边缘连接时才会保留。

问题 6：仿射变换和透视变换有什么区别？

核心说明：
仿射变换可以表示平移、旋转、缩放、剪切，并保持直线和平行关系；透视变换更一般，可以模拟近大远小的投影变化，不一定保持平行关系，常用于文档矫正、车牌矫正和鸟瞰图变换。

问题 7：传统 OpenCV 和深度学习怎么结合？

核心说明：
OpenCV 常用于深度学习前处理和后处理。前处理包括读取图像、resize、颜色空间转换、归一化、构造 blob；后处理包括绘制检测框、NMS、mask 阈值化、连通域过滤、轮廓提取和结果可视化。在实际项目里，即使模型是 YOLO 或 U-Net，OpenCV 仍然非常常用。

## 二十九、知识总结

核心问题：
熟悉 OpenCV 吗？常用哪些模块？

核心说明：

常用 OpenCV 做图像读取、预处理、传统图像分析和深度学习后处理。基础部分包括 imread、imshow、imwrite、resize、cvtColor 等；图像处理部分包括高斯滤波、中值滤波、阈值分割、Otsu、自适应阈值、形态学开闭运算、Canny 边缘检测、轮廓检测和连通域分析；几何变换部分包括仿射变换、透视变换、旋转、裁剪；视频处理部分包括 VideoCapture 和 VideoWriter；在深度学习项目中，可以用 OpenCV 做模型输入前处理、检测框绘制、NMS、mask 后处理和视频结果保存。

比如在分割项目中，可以对模型输出的 mask 做阈值化，然后用形态学操作去除噪声，再通过连通域分析过滤小区域误检；在目标检测项目中，可以用 OpenCV 读取视频帧、送入模型推理、绘制 bbox、类别和置信度，并统计 FPS 后保存结果视频。

## 三十、需要重点掌握的函数清单

最后给一个知识总结复习清单。

基础 IO
cv2.imread
cv2.imwrite
cv2.imshow
cv2.waitKey
cv2.destroyAllWindows
颜色和尺寸
cv2.cvtColor
cv2.resize
cv2.flip
cv2.rotate
cv2.split
cv2.merge
滤波增强
cv2.blur
cv2.GaussianBlur
cv2.medianBlur
cv2.bilateralFilter
cv2.filter2D
cv2.equalizeHist
cv2.createCLAHE
阈值和形态学
cv2.threshold
cv2.adaptiveThreshold
cv2.inRange
cv2.erode
cv2.dilate
cv2.morphologyEx
边缘和轮廓
cv2.Canny
cv2.Sobel
cv2.Laplacian
cv2.findContours
cv2.drawContours
cv2.contourArea
cv2.arcLength
cv2.boundingRect
cv2.minAreaRect
cv2.approxPolyDP
cv2.convexHull
cv2.connectedComponentsWithStats
几何变换
cv2.getRotationMatrix2D
cv2.warpAffine
cv2.getPerspectiveTransform
cv2.warpPerspective
特征匹配
cv2.ORB_create
cv2.SIFT_create
cv2.BFMatcher
cv2.FlannBasedMatcher
cv2.findHomography
视频
cv2.VideoCapture
cv2.VideoWriter
cv2.VideoWriter_fourcc
cv2.CAP_PROP_FPS
相机标定
cv2.findChessboardCorners
cv2.calibrateCamera
cv2.undistort
cv2.solvePnP
深度学习 DNN
cv2.dnn.blobFromImage
cv2.dnn.readNet
net.setInput
net.forward
cv2.dnn.NMSBoxes