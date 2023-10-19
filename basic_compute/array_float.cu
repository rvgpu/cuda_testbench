#include <cstdio>
#include <gtest/gtest.h>

template <typename T> __global__ void add(T *x, T *y, T *out) { out[threadIdx.x] = x[threadIdx.x] + y[threadIdx.x]; }
template <typename T> __global__ void sub(T *x, T *y, T *out) { out[threadIdx.x] = x[threadIdx.x] - y[threadIdx.x]; }
template <typename T> __global__ void mul(T *x, T *y, T *out) { out[threadIdx.x] = x[threadIdx.x] * y[threadIdx.x]; }
template <typename T> __global__ void div(T *x, T *y, T *out) { out[threadIdx.x] = x[threadIdx.x] / y[threadIdx.x]; }

template<typename T>
class CUDAArrayTest {
public:
    CUDAArrayTest() {
        kDataLen    = 1024;
        in1         = new T[kDataLen];
        in2         = new T[kDataLen];
        result      = new T[kDataLen];
        reference   = new T[kDataLen];

        for (int i=0; i<kDataLen; ++i) {
            in1[i] = i * 11.0f;
            in2[i] = i + 100.0f;
            result[i] = 12;
            reference[i] = 34;
        }

        cudaMalloc(&device_in1, kDataLen * sizeof(T));
        cudaMalloc(&device_in2, kDataLen * sizeof(T));
        cudaMalloc(&device_out, kDataLen * sizeof(T));

        cudaMemcpy(device_in1, in1, kDataLen * sizeof(T), cudaMemcpyHostToDevice);
        cudaMemcpy(device_in2, in2, kDataLen * sizeof(T), cudaMemcpyHostToDevice);
    }

    ~CUDAArrayTest() {
        delete in1;
        delete in2;
        delete result;
        delete reference;

        cudaDeviceReset();
    }

    void getResultData() {
        cudaDeviceSynchronize();
        cudaMemcpy(result, device_out, kDataLen * sizeof(T), cudaMemcpyDeviceToHost);
    }

    void runKernelAdd(void) { 
        add<T><<<1, kDataLen>>>(device_in1, device_in2, device_out); 
        getResultData();

        // print the results
        for (int i=0; i<kDataLen; ++i) {
            reference[i] = in1[i] + in2[i];
            EXPECT_DOUBLE_EQ(result[i], reference[i]);
        }
    }
    
    void runKernelSub(void) { 
        sub<T><<<1, kDataLen>>>(device_in1, device_in2, device_out); 
        getResultData();

        // print the results
        for (int i=0; i<kDataLen; ++i) {
            reference[i] = in1[i] - in2[i];
            EXPECT_DOUBLE_EQ(result[i], reference[i]);
        }
    }
    
    void runKernelMul(void) {
        mul<T><<<1, kDataLen>>>(device_in1, device_in2, device_out); 
        getResultData();

        // print the results
        for (int i=0; i<kDataLen; ++i) {
            reference[i] = in1[i] * in2[i];
            EXPECT_DOUBLE_EQ(result[i], reference[i]);
        }
    }

    void runKernelDiv(void) {
        cudaMemcpy(device_in2, in2, kDataLen * sizeof(T), cudaMemcpyHostToDevice);

        div<T><<<1, kDataLen>>>(device_in1, device_in2, device_out);
        getResultData();

        // print the results
        for (int i=0; i<kDataLen; ++i) {
            reference[i] = in1[i] / in2[i];
            EXPECT_DOUBLE_EQ(result[i], reference[i]);
        }
    }

    int kDataLen;
    T *in1;
    T *in2;
    T *result;
    T *reference;

    T *device_in1;
    T *device_in2;
    T *device_out;
};

TEST(CUDAArrayTest_F, Add_float)  { auto test = new CUDAArrayTest<float>;  test->runKernelAdd(); delete test; }
TEST(CUDAArrayTest_F, Add_double) { auto test = new CUDAArrayTest<double>; test->runKernelAdd(); delete test; }

TEST(CUDAArrayTest_F, Sub_float)  { auto test = new CUDAArrayTest<float>;  test->runKernelSub(); delete test; }
TEST(CUDAArrayTest_F, Sub_double) { auto test = new CUDAArrayTest<double>; test->runKernelSub(); delete test; }

TEST(CUDAArrayTest_F, Mul_float)  { auto test = new CUDAArrayTest<float>;  test->runKernelMul(); delete test; }
TEST(CUDAArrayTest_F, Mul_double) { auto test = new CUDAArrayTest<double>; test->runKernelMul(); delete test; }

TEST(CUDAArrayTest_F, Div_float)  { auto test = new CUDAArrayTest<float>;  test->runKernelDiv(); delete test; }
TEST(CUDAArrayTest_F, Div_double) { auto test = new CUDAArrayTest<double>; test->runKernelDiv(); delete test; }