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
            in1[i] = i * 100;
            in2[i] = i + 100;
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
            EXPECT_EQ(result[i], reference[i]);
        }
    }
    
    void runKernelSub(void) { 
        sub<T><<<1, kDataLen>>>(device_in1, device_in2, device_out); 
        getResultData();

        // print the results
        for (int i=0; i<kDataLen; ++i) {
            reference[i] = in1[i] - in2[i];
            EXPECT_EQ(result[i], reference[i]);
        }
    }
    
    void runKernelMul(void) {
        mul<T><<<1, kDataLen>>>(device_in1, device_in2, device_out); 
        getResultData();

        // print the results
        for (int i=0; i<kDataLen; ++i) {
            reference[i] = in1[i] * in2[i];
            EXPECT_EQ(result[i], reference[i]);
        }
    }

    void runKernelDiv(void) {
        // check is div zero
        for (int i=0; i<kDataLen; i++) {
            if (in2[i] == 0) {
                in2[i] = 13;
            }
        }
        cudaMemcpy(device_in2, in2, kDataLen * sizeof(T), cudaMemcpyHostToDevice);

        div<T><<<1, kDataLen>>>(device_in1, device_in2, device_out);
        getResultData();

        // print the results
        for (int i=0; i<kDataLen; ++i) {
            reference[i] = in1[i] / in2[i];
            EXPECT_EQ(result[i], reference[i]);
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

TEST(CUDAArrayTest_F, Add_ui64) { auto test = new CUDAArrayTest<uint64_t>; test->runKernelAdd(); delete test; }
TEST(CUDAArrayTest_F, Add_ui32) { auto test = new CUDAArrayTest<uint32_t>; test->runKernelAdd(); delete test; }
TEST(CUDAArrayTest_F, Add_ui16) { auto test = new CUDAArrayTest<uint16_t>; test->runKernelAdd(); delete test; }
TEST(CUDAArrayTest_F, Add_ui8)  { auto test = new CUDAArrayTest<uint8_t>;  test->runKernelAdd(); delete test; }
TEST(CUDAArrayTest_F, Add_i64)  { auto test = new CUDAArrayTest<int64_t>;  test->runKernelAdd(); delete test; }
TEST(CUDAArrayTest_F, Add_i32)  { auto test = new CUDAArrayTest<int32_t>;  test->runKernelAdd(); delete test; }
TEST(CUDAArrayTest_F, Add_i16)  { auto test = new CUDAArrayTest<int16_t>;  test->runKernelAdd(); delete test; }
TEST(CUDAArrayTest_F, Add_i8)   { auto test = new CUDAArrayTest<int8_t>;   test->runKernelAdd(); delete test; }

TEST(CUDAArrayTest_F, Sub_ui64) { auto test = new CUDAArrayTest<uint64_t>; test->runKernelSub(); delete test; }
TEST(CUDAArrayTest_F, Sub_ui32) { auto test = new CUDAArrayTest<uint32_t>; test->runKernelSub(); delete test; }
TEST(CUDAArrayTest_F, Sub_ui16) { auto test = new CUDAArrayTest<uint16_t>; test->runKernelSub(); delete test; }
TEST(CUDAArrayTest_F, Sub_ui8)  { auto test = new CUDAArrayTest<uint8_t>;  test->runKernelSub(); delete test; }
TEST(CUDAArrayTest_F, Sub_i64)  { auto test = new CUDAArrayTest<int64_t>;  test->runKernelSub(); delete test; }
TEST(CUDAArrayTest_F, Sub_i32)  { auto test = new CUDAArrayTest<int32_t>;  test->runKernelSub(); delete test; }
TEST(CUDAArrayTest_F, Sub_i16)  { auto test = new CUDAArrayTest<int16_t>;  test->runKernelSub(); delete test; }
TEST(CUDAArrayTest_F, Sub_i8)   { auto test = new CUDAArrayTest<int8_t>;   test->runKernelSub(); delete test; }

TEST(CUDAArrayTest_F, Mul_ui64) { auto test = new CUDAArrayTest<uint64_t>; test->runKernelMul(); delete test; }
TEST(CUDAArrayTest_F, Mul_ui32) { auto test = new CUDAArrayTest<uint32_t>; test->runKernelMul(); delete test; }
TEST(CUDAArrayTest_F, Mul_ui16) { auto test = new CUDAArrayTest<uint16_t>; test->runKernelMul(); delete test; }
TEST(CUDAArrayTest_F, Mul_ui8)  { auto test = new CUDAArrayTest<uint8_t>;  test->runKernelMul(); delete test; }
TEST(CUDAArrayTest_F, Mul_i64)  { auto test = new CUDAArrayTest<int64_t>;  test->runKernelMul(); delete test; }
TEST(CUDAArrayTest_F, Mul_i32)  { auto test = new CUDAArrayTest<int32_t>;  test->runKernelMul(); delete test; }
TEST(CUDAArrayTest_F, Mul_i16)  { auto test = new CUDAArrayTest<int16_t>;  test->runKernelMul(); delete test; }
TEST(CUDAArrayTest_F, Mul_i8)   { auto test = new CUDAArrayTest<int8_t>;   test->runKernelMul(); delete test; }

TEST(CUDAArrayTest_F, Div_ui64) { auto test = new CUDAArrayTest<uint64_t>; test->runKernelDiv(); delete test; }
TEST(CUDAArrayTest_F, Div_ui32) { auto test = new CUDAArrayTest<uint32_t>; test->runKernelDiv(); delete test; }
TEST(CUDAArrayTest_F, Div_ui16) { auto test = new CUDAArrayTest<uint16_t>; test->runKernelDiv(); delete test; }
TEST(CUDAArrayTest_F, Div_ui8)  { auto test = new CUDAArrayTest<uint8_t>;  test->runKernelDiv(); delete test; }
TEST(CUDAArrayTest_F, Div_i64)  { auto test = new CUDAArrayTest<int64_t>;  test->runKernelDiv(); delete test; }
TEST(CUDAArrayTest_F, Div_i32)  { auto test = new CUDAArrayTest<int32_t>;  test->runKernelDiv(); delete test; }
TEST(CUDAArrayTest_F, Div_i16)  { auto test = new CUDAArrayTest<int16_t>;  test->runKernelDiv(); delete test; }
TEST(CUDAArrayTest_F, Div_i8)   { auto test = new CUDAArrayTest<int8_t>;   test->runKernelDiv(); delete test; }