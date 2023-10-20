#include <cstdio>
#include <gtest/gtest.h>

template <typename T> __global__ void add(T *x, T *y, T *out) { out[threadIdx.x] = x[threadIdx.x] + y[threadIdx.x]; }
template <typename T> __global__ void sub(T *x, T *y, T *out) { out[threadIdx.x] = x[threadIdx.x] - y[threadIdx.x]; }
template <typename T> __global__ void mul(T *x, T *y, T *out) { out[threadIdx.x] = x[threadIdx.x] * y[threadIdx.x]; }
template <typename T> __global__ void div(T *x, T *y, T *out) { out[threadIdx.x] = x[threadIdx.x] / y[threadIdx.x]; }
__global__ void mul(int a, float *x, float *out) { out[threadIdx.x] = a * x[threadIdx.x]; }
__global__ void mul(float a, float *x, float *out) { out[threadIdx.x] = a * x[threadIdx.x]; }

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
            in1[i] = i * 3;
            in2[i] = i + 5;
            result[i] = 12;
            reference[i] = 34;

            if (in2[i] == 0) {
                in2[i] = 123;
            }
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

    void compareInteger() {
        getResultData();

        for (int i=0; i<kDataLen; ++i) {
            EXPECT_EQ(result[i], reference[i]);
        }
    }

    void compareFloat() {
        getResultData();

        for (int i=0; i<kDataLen; ++i) {
            EXPECT_DOUBLE_EQ(result[i], reference[i]);
        }
    }

    void referenceAdd() { for (int i=0; i<kDataLen; ++i) { reference[i] = in1[i] + in2[i]; } }
    void referenceSub() { for (int i=0; i<kDataLen; ++i) { reference[i] = in1[i] - in2[i]; } }
    void referenceMul() { for (int i=0; i<kDataLen; ++i) { reference[i] = in1[i] * in2[i]; } }
    void referenceDiv() { for (int i=0; i<kDataLen; ++i) { reference[i] = in1[i] / in2[i]; } }

    void runKernelAdd(void) { add<T><<<1, kDataLen>>>(device_in1, device_in2, device_out); referenceAdd(); }
    void runKernelSub(void) { sub<T><<<1, kDataLen>>>(device_in1, device_in2, device_out); referenceSub(); }
    void runKernelMul(void) { mul<T><<<1, kDataLen>>>(device_in1, device_in2, device_out); referenceMul(); }
    void runKernelDiv(void) { div<T><<<1, kDataLen>>>(device_in1, device_in2, device_out); referenceDiv(); }

    int kDataLen;
    T *in1;
    T *in2;
    T *result;
    T *reference;

    T *device_in1;
    T *device_in2;
    T *device_out;
};

template <typename OP_INARRAY_T, typename OP_INPUT_T, int array_size=1>
class ArrayWithAnOperand {
public:
    ArrayWithAnOperand()  {
        in                   = new OP_INARRAY_T[array_size];
        result               = new OP_INARRAY_T[array_size];
        reference            = new OP_INARRAY_T[array_size];
        for (int i=0; i<array_size; ++i) {
            result[i] = -1;
            reference[i] = -2;
        }
        cudaMalloc(&device_in, array_size * sizeof(OP_INARRAY_T));
        cudaMalloc(&device_out, array_size * sizeof(OP_INARRAY_T));
    }

    ~ArrayWithAnOperand() {
        delete in;
        delete result;
        delete reference;
    }
    OP_INARRAY_T* make_array() {
        for (auto i = 0; i < array_size; i++) {
            in[i] = OP_INARRAY_T(i);
        }
        return in;
    }

    OP_INARRAY_T * make_mul_reference(OP_INPUT_T in1, OP_INARRAY_T* in2) {
        // print the results
        for (int i=0; i<array_size; ++i) {
            reference[i] = in1 * in2[i];
        }
        return reference;
    }

    void expect_eq(OP_INARRAY_T* in1, OP_INARRAY_T* in2) {
        for (int i=0; i<array_size; ++i) {
            if (std::is_same<OP_INARRAY_T, float>::value) {
                EXPECT_FLOAT_EQ(result[i], reference[i]);
            }else {
                EXPECT_EQ(result[i], reference[i]);
            }
        }
    }
    OP_INARRAY_T * array_mul_by(OP_INPUT_T in1, OP_INARRAY_T* in2) {
        cudaMemcpy(device_in, in2, array_size * sizeof(OP_INARRAY_T), cudaMemcpyHostToDevice);

        mul<<<1, array_size>>>(in1, device_in, device_out);
        cudaDeviceSynchronize();
        cudaMemcpy(result, device_out, array_size * sizeof(OP_INARRAY_T), cudaMemcpyDeviceToHost);

        cudaDeviceReset();
        return result;
    }

    OP_INARRAY_T *device_in;
    OP_INARRAY_T *in;

private:
    OP_INARRAY_T *result;
    OP_INARRAY_T *reference;
    OP_INARRAY_T *device_out;
};

TEST(CUDAArrayTest_F, Add_ui64) { auto test = new CUDAArrayTest<uint64_t>; test->runKernelAdd(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Add_ui32) { auto test = new CUDAArrayTest<uint32_t>; test->runKernelAdd(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Add_ui16) { auto test = new CUDAArrayTest<uint16_t>; test->runKernelAdd(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Add_ui8)  { auto test = new CUDAArrayTest<uint8_t>;  test->runKernelAdd(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Add_i64)  { auto test = new CUDAArrayTest<int64_t>;  test->runKernelAdd(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Add_i32)  { auto test = new CUDAArrayTest<int32_t>;  test->runKernelAdd(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Add_i16)  { auto test = new CUDAArrayTest<int16_t>;  test->runKernelAdd(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Add_i8)   { auto test = new CUDAArrayTest<int8_t>;   test->runKernelAdd(); test->compareInteger(); delete test; }

TEST(CUDAArrayTest_F, Sub_ui64) { auto test = new CUDAArrayTest<uint64_t>; test->runKernelSub(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Sub_ui32) { auto test = new CUDAArrayTest<uint32_t>; test->runKernelSub(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Sub_ui16) { auto test = new CUDAArrayTest<uint16_t>; test->runKernelSub(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Sub_ui8)  { auto test = new CUDAArrayTest<uint8_t>;  test->runKernelSub(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Sub_i64)  { auto test = new CUDAArrayTest<int64_t>;  test->runKernelSub(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Sub_i32)  { auto test = new CUDAArrayTest<int32_t>;  test->runKernelSub(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Sub_i16)  { auto test = new CUDAArrayTest<int16_t>;  test->runKernelSub(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Sub_i8)   { auto test = new CUDAArrayTest<int8_t>;   test->runKernelSub(); test->compareInteger(); delete test; }

TEST(CUDAArrayTest_F, Mul_ui64) { auto test = new CUDAArrayTest<uint64_t>; test->runKernelMul(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Mul_ui32) { auto test = new CUDAArrayTest<uint32_t>; test->runKernelMul(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Mul_ui16) { auto test = new CUDAArrayTest<uint16_t>; test->runKernelMul(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Mul_ui8)  { auto test = new CUDAArrayTest<uint8_t>;  test->runKernelMul(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Mul_i64)  { auto test = new CUDAArrayTest<int64_t>;  test->runKernelMul(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Mul_i32)  { auto test = new CUDAArrayTest<int32_t>;  test->runKernelMul(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Mul_i16)  { auto test = new CUDAArrayTest<int16_t>;  test->runKernelMul(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Mul_i8)   { auto test = new CUDAArrayTest<int8_t>;   test->runKernelMul(); test->compareInteger(); delete test; }

TEST(CUDAArrayTest_F, Div_ui64) { auto test = new CUDAArrayTest<uint64_t>; test->runKernelDiv(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Div_ui32) { auto test = new CUDAArrayTest<uint32_t>; test->runKernelDiv(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Div_ui16) { auto test = new CUDAArrayTest<uint16_t>; test->runKernelDiv(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Div_ui8)  { auto test = new CUDAArrayTest<uint8_t>;  test->runKernelDiv(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Div_i64)  { auto test = new CUDAArrayTest<int64_t>;  test->runKernelDiv(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Div_i32)  { auto test = new CUDAArrayTest<int32_t>;  test->runKernelDiv(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Div_i16)  { auto test = new CUDAArrayTest<int16_t>;  test->runKernelDiv(); test->compareInteger(); delete test; }
TEST(CUDAArrayTest_F, Div_i8)   { auto test = new CUDAArrayTest<int8_t>;   test->runKernelDiv(); test->compareInteger(); delete test; }

TEST(CUDAArrayTest_F, Add_float)  { auto test = new CUDAArrayTest<float>;  test->runKernelAdd(); test->compareFloat();   delete test; }
TEST(CUDAArrayTest_F, Add_double) { auto test = new CUDAArrayTest<double>; test->runKernelAdd(); test->compareFloat();   delete test; }

TEST(CUDAArrayTest_F, Sub_float)  { auto test = new CUDAArrayTest<float>;  test->runKernelSub(); test->compareFloat();   delete test; }
TEST(CUDAArrayTest_F, Sub_double) { auto test = new CUDAArrayTest<double>; test->runKernelSub(); test->compareFloat();   delete test; }

TEST(CUDAArrayTest_F, Mul_float)  { auto test = new CUDAArrayTest<float>;  test->runKernelMul(); test->compareFloat();   delete test; }
TEST(CUDAArrayTest_F, Mul_double) { auto test = new CUDAArrayTest<double>; test->runKernelMul(); test->compareFloat();   delete test; }

TEST(CUDAArrayTest_F, Div_float)  { auto test = new CUDAArrayTest<float>;  test->runKernelDiv(); test->compareFloat();   delete test; }
TEST(CUDAArrayTest_F, Div_double) { auto test = new CUDAArrayTest<double>; test->runKernelDiv(); test->compareFloat();   delete test; }

TEST(ArrayWithAnOperand, float_array_mul_by_int) {
auto basic_test = ArrayWithAnOperand<float, int, 2>();
float* array = basic_test.make_array();

auto res = basic_test.array_mul_by(2, array);
auto ref = basic_test.make_mul_reference(2, array);

basic_test.expect_eq(res, ref);
}

TEST(ArrayWithAnOperand, float_array_mul_by_float) {
auto basic_test = ArrayWithAnOperand<float, float, 2>();
float* array = basic_test.make_array();

auto res = basic_test.array_mul_by(1.5, array);
auto ref = basic_test.make_mul_reference(1.5, array);

basic_test.expect_eq(res, ref);
}