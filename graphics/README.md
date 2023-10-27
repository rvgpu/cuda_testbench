# Graphics

## 编译和运行

* 设置环境变量

```
export CUDA_HOME=/usr/local/cuda-11
export PATH=/usr/local/cuda-11/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:${LD_LIBRARY_PATH}
```

* CUDA 11不支持devtoolset-11，所以使用devtoolset-9

```
source /opt/rh/devtoolset-9/enable
```

* 使用NVCC编译，运行后生成渲染的图片，比如

```
nvcc triangle.cu
./a.out
```

## 渲染的例子

* triangle，一个三角形，具有线性插值的颜色
* games101_hw1，一个白色的线框三角形
* games101_hw2，一个黄色的三角形，遮挡一个蓝色的三角形
* games101_hw4，一条黄色的Bazier曲线