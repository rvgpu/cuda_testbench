# BasicTest

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

* 使用NVCC编译，运行测试

```
nvcc basic_test.cu -lgtest -lgtest_main
./a.out
```

## Kernel和测试用例

* array_add
* multi_array_add
* multi_array_mul
* multi_array_muladd
* multi_array_fmuladd
* branch_if
* branch_ifelse
* branch_ifif
* branch_for
* math_sin
* math_cos
* math_pow