# Docker

1. 解释 Docker 是什么、为什么要用
2. 理解镜像、容器、Dockerfile、数据卷、端口映射
3. 能把一个 Python / FastAPI / 深度学习推理项目打包成镜像
4. 能用 docker run 启动服务
5. 能用 docker-compose 编排前后端、数据库、模型服务
6. 知道模型部署场景里 Docker 怎么用

## 一、Docker 是什么？

Docker 是一种 容器化技术。

可以简单理解为：

Docker 可以把你的程序、依赖库、运行环境、启动命令一起打包成一个标准化运行单元，让它在不同机器上都能以相同方式运行。

比如本地训练了一个模型推理服务：

Python 3.10
PyTorch
OpenCV
ONNX Runtime
FastAPI
模型文件 best.onnx
推理代码 app.py

如果直接拿到别人电脑或者服务器上运行，可能会遇到：

Python 版本不一致
OpenCV 没装
CUDA 版本不匹配
缺少依赖包
路径不一致
系统环境不一致

Docker 的作用就是把这些环境封装起来：

你的代码
+
Python 环境
+
依赖包
+
模型文件
+
启动命令
↓
Docker 镜像
↓
在任何安装 Docker 的机器上运行

一句话：

Docker 解决的是“我的程序在我电脑上能跑，但到别人电脑或服务器上跑不起来”的问题。

## 二、Docker 和虚拟机有什么区别？

很多人刚学 Docker，会把它和虚拟机混在一起。

### 1. 虚拟机

虚拟机是：

宿主机操作系统
↓
虚拟机软件
↓
完整的虚拟操作系统
↓
应用程序

比如你在 Windows 上装一个 Ubuntu 虚拟机。

它有完整的系统内核、系统服务、文件系统，所以比较重。

特点：

隔离性强
启动慢
占用资源多
体积大
### 2. Docker 容器

Docker 是：

宿主机操作系统
↓
Docker Engine
↓
容器
↓
应用程序

容器共享宿主机内核，不需要启动完整操作系统。

特点：

启动快
体积小
资源占用低
适合部署服务
适合微服务和模型推理
### 3. 对比
对比项	虚拟机	Docker 容器
是否包含完整 OS	是	否
启动速度	慢，分钟级	快，秒级
资源占用	高	低
部署方式	镜像 + 虚拟系统	镜像 + 容器
适合场景	完整系统隔离	应用部署、服务部署、模型部署

## 三、Docker 的几个核心概念

必须掌握这几个词：

Image 镜像
Container 容器
Dockerfile
Docker Hub
Volume 数据卷
Port Mapping 端口映射
Network 网络
docker-compose

## 四、Image 镜像是什么？

镜像可以理解成：

一个打包好的运行环境模板。

比如：

python:3.10
ubuntu:22.04
nginx:latest
mysql:8.0
pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

这些都是镜像。

镜像里面可以包含：

操作系统基础环境
Python
依赖库
项目代码
模型文件
启动命令

镜像本身是静态的，不会运行。

可以类比：

镜像 = 类 class / 安装包 / 模板
容器 = 实例对象 / 正在运行的程序

## 五、Container 容器是什么？

容器是镜像运行起来后的实例。

比如你有一个镜像：

my-fastapi-app:1.0

运行它：

docker run my-fastapi-app:1.0

就会创建一个容器。

容器是动态运行的，里面有你的服务进程。

可以类比：

镜像 Image：模型文件 / 程序模板
容器 Container：真正运行中的服务

一个镜像可以启动多个容器：

my-app 镜像
↓
container 1
container 2
container 3

## 六、Dockerfile 是什么？

Dockerfile 是用来构建镜像的说明书。

它告诉 Docker：

基于哪个基础镜像
复制哪些代码
安装哪些依赖
暴露哪个端口
运行什么命令

一个最简单的 Python 项目 Dockerfile：

FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main.py"]

解释：

FROM：指定基础镜像
WORKDIR：设置容器内工作目录
COPY：复制本地文件到容器
RUN：构建镜像时执行命令
CMD：容器启动时执行命令

## 七、Docker 部署的基本流程

完整流程是：

1. 写代码
2. 写 requirements.txt
3. 写 Dockerfile
4. docker build 构建镜像
5. docker run 启动容器
6. 测试服务是否正常
7. 推送镜像到服务器或镜像仓库
8. 服务器拉取镜像并运行

对应命令：

docker build -t my-app:1.0 .
docker run -d -p 8000:8000 my-app:1.0

## 八、最常用 Docker 命令
1. 查看 Docker 版本
docker --version
2. 查看本地镜像
docker images
3. 拉取镜像
docker pull python:3.10-slim
4. 构建镜像
docker build -t my-app:1.0 .

含义：

-t my-app:1.0：给镜像起名和版本
.：使用当前目录下的 Dockerfile 构建
5. 运行容器
docker run my-app:1.0

后台运行：

docker run -d my-app:1.0
6. 端口映射
docker run -d -p 8000:8000 my-app:1.0

含义：

宿主机 8000 端口 → 容器 8000 端口

如果 FastAPI 在容器里监听 8000，那么你可以在宿主机访问：

http://localhost:8000
7. 查看运行中的容器
docker ps

查看所有容器：

docker ps -a
8. 停止容器
docker stop 容器ID
9. 删除容器
docker rm 容器ID
10. 删除镜像
docker rmi 镜像ID
11. 查看容器日志
docker logs 容器ID

持续查看：

docker logs -f 容器ID
12. 进入容器内部
docker exec -it 容器ID bash

如果镜像没有 bash，可以用：

docker exec -it 容器ID sh

## 九、以 FastAPI 模型推理服务为例部署

假设有一个深度学习模型推理项目：
```
model_service/
  app.py
  requirements.txt
  best.onnx
  Dockerfile
1. app.py

这是一个简单的 FastAPI 推理服务：

from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
import onnxruntime as ort

app = FastAPI(title="Defect Detection API")

session = ort.InferenceSession(
    "best.onnx",
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name


def preprocess(image_bytes):
    file_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0

    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)

    return img


@app.get("/")
def root():
    return {"message": "Model service is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    x = preprocess(image_bytes)

    outputs = session.run([output_name], {input_name: x})
    logits = outputs[0]

    pred = int(np.argmax(logits, axis=1)[0])

    return {
        "filename": file.filename,
        "prediction": pred
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

注意这里：

host="0.0.0.0"

非常重要。

如果你写：

host="127.0.0.1"

容器外部可能访问不到。

2. requirements.txt
fastapi
uvicorn
python-multipart
numpy
opencv-python-headless
onnxruntime

注意：

Docker 里建议用 opencv-python-headless

因为它不依赖 GUI，比 opencv-python 更适合服务器环境。

3. Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
4. 构建镜像

在 model_service 目录下执行：

docker build -t defect-api:1.0 .
5. 运行容器
docker run -d -p 8000:8000 --name defect-api-container defect-api:1.0

然后访问：

http://localhost:8000

如果看到：

{"message": "Model service is running"}

说明服务跑起来了。
```

## 十、Docker 部署机器学习模型时要注意什么？

模型部署和普通 Web 项目不一样，主要注意这些：

模型文件路径
依赖库版本
OpenCV 无 GUI 环境问题
GPU / CUDA 支持
推理端口
输入输出预处理
模型文件体积
日志和结果保存
### 1. 模型文件要复制进镜像

Dockerfile 中：

COPY . .

会把 best.onnx 一起复制进去。

如果模型文件很大，也可以挂载数据卷：

docker run -v D:/models:/models my-app

代码里加载：

session = ort.InferenceSession("/models/best.onnx")
### 2. OpenCV 用 headless 版本

服务器或容器里一般没有图形界面，所以用：

opencv-python-headless

不要用：

opencv-python

否则可能报：

libGL.so.1: cannot open shared object file
### 3. GPU 推理要用 NVIDIA Docker

如果你的模型需要 GPU，例如 PyTorch / CUDA / onnxruntime-gpu，需要：

docker run --gpus all ...

例如：

docker run --gpus all -d -p 8000:8000 my-gpu-app:1.0

并且基础镜像要支持 CUDA，例如：

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

或者 PyTorch 官方 CUDA 镜像：

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

## 十一、CPU 版 ONNX Runtime Dockerfile

如果是 CPU 推理：
```
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

requirements.txt：

fastapi
uvicorn
python-multipart
numpy
opencv-python-headless
onnxruntime
```

## 十二、GPU 版 PyTorch Dockerfile

如果要部署 PyTorch GPU 模型：
```
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

requirements.txt：

fastapi
uvicorn
python-multipart
opencv-python-headless
numpy
Pillow

运行：

docker run --gpus all -d -p 8000:8000 my-pytorch-api:1.0
```

## 十三、GPU 版 ONNX Runtime Dockerfile

如果用 ONNX Runtime GPU：
```
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

requirements.txt：

fastapi
uvicorn
python-multipart
numpy
opencv-python-headless
onnxruntime-gpu

运行：

docker run --gpus all -d -p 8000:8000 onnx-gpu-api:1.0
```

## 十四、docker-compose 是什么？

如果项目只有一个服务，用 docker run 就够了。

但真实项目经常有多个服务：

前端 React
后端 FastAPI
数据库 MySQL / PostgreSQL
Redis
Nginx
模型推理服务

这时一个个 docker run 很麻烦。

docker-compose 用一个 YAML 文件统一管理多个容器。

## 十五、docker-compose 示例：模型服务 + 前端

项目结构：
```
project/
  docker-compose.yml
  backend/
    app.py
    requirements.txt
    Dockerfile
  frontend/
    package.json
    Dockerfile
1. backend/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
2. frontend/Dockerfile

React 项目可以这样：

FROM node:20-alpine AS build

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

RUN npm run build


FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
3. docker-compose.yml
services:
  backend:
    build: ./backend
    container_name: model-backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend/uploads:/app/uploads
    restart: always

  frontend:
    build: ./frontend
    container_name: blog-frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: always

启动：

docker compose up -d

查看日志：

docker compose logs -f

停止：

docker compose down

重新构建：

docker compose up -d --build
```

## 十六、Volume 数据卷是什么？

容器默认是临时的。

如果在容器里保存了文件，容器删掉后，文件也可能没了。

比如模型服务生成：

检测结果图片
日志
上传的 PDF
CSV 报告

这些不能只存在容器内部。

所以要用 volume，把宿主机目录挂载到容器目录。

示例：

docker run -v D:/data/uploads:/app/uploads my-app

含义：

宿主机 D:/data/uploads
映射到容器 /app/uploads

容器内写入：

/app/uploads/result.jpg

实际会保存到宿主机：

D:/data/uploads/result.jpg

## 十七、端口映射是什么？

容器内部有自己的网络环境。

FastAPI 监听容器内：

8000

但宿主机不能直接访问容器内部端口。

所以要映射：

-p 8000:8000

格式：

宿主机端口:容器端口

例如：

docker run -p 9000:8000 my-app

表示：

访问宿主机 localhost:9000
↓
转发到容器内部 8000

## 十八、Docker 网络是什么？

如果多个容器之间要通信，例如：

frontend 容器
backend 容器
mysql 容器

它们在 docker-compose 里可以通过服务名访问。

比如 backend 服务名叫：

backend

前端或 Nginx 可以访问：

http://backend:8000

数据库服务名叫：

mysql

后端可以连接：

mysql:3306

这就是 Docker 内部网络。

## 十九、Docker 部署中的常见问题
1. 容器启动后访问不到服务

常见原因：

服务监听的是 127.0.0.1
没有监听 0.0.0.0
没有做端口映射
防火墙拦截

解决：

FastAPI 启动要写：

uvicorn app:app --host 0.0.0.0 --port 8000

运行容器要写：

docker run -p 8000:8000 my-app
2. 容器里找不到模型文件

常见原因：

Dockerfile 没 COPY 模型文件
路径写错
运行目录不对
模型文件太大没放进去

解决：

检查：

docker exec -it 容器ID bash
ls /app

或者用 volume 挂载模型目录：

docker run -v D:/models:/models my-app
3. OpenCV 报 libGL 错误

报错：

ImportError: libGL.so.1: cannot open shared object file

解决一：

opencv-python-headless

解决二：

Dockerfile 安装：

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0
4. pip 安装太慢

可以用国内镜像：

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
5. 镜像太大

优化方式：

使用 slim / alpine 基础镜像
删除缓存
不要复制无关文件
使用 .dockerignore
多阶段构建
模型文件按需挂载

.dockerignore 示例：

.git
__pycache__
*.pyc
.vscode
node_modules
dist
output
runs
*.pth
*.pt

注意：如果模型文件要打进镜像，就不要忽略它。

## 二十、.dockerignore 是什么？

类似 .gitignore。

它告诉 Docker 构建镜像时不要复制哪些文件。

例如：

.git
__pycache__
*.pyc
.ipynb_checkpoints
node_modules
runs
output
dataset

否则 Docker 构建上下文会很大，构建会很慢。