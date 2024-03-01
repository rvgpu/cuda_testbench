# Graphics

## Environment Variables

```
export CUDA_HOME=/usr/local/cuda-11
export PATH=/usr/local/cuda-11/bin:${PATH}
export LD_LIBRARY_PATH=/usr/local/cuda-11/lib64:${LD_LIBRARY_PATH}
```

CUDA 11 does not support devtoolset-11, so we use devtoolset-9

```
source /opt/rh/devtoolset-9/enable
```

## Compile and Run

CUDA is required in CMakeLists.txt

```
mkdir build
cd build
cmake ..
make -j8

./games101_hw5
```
