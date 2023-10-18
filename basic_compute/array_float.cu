#include <cstdio>
#include <gtest/gtest.h>

__global__ void add(float *x, float *y, float *out)
{
    out[threadIdx.x] = x[threadIdx.x] + y[threadIdx.x];
}

__global__ void sub(float *x, float *y, float *out)
{
    out[threadIdx.x] = x[threadIdx.x] - y[threadIdx.x];
}

__global__ void mul(float *x, float *y, float *out)
{
    out[threadIdx.x] = x[threadIdx.x] * y[threadIdx.x];
}

__global__ void div(float *x, float *y, float *out)
{
    out[threadIdx.x] = x[threadIdx.x] / y[threadIdx.x];
}

class CUDATestOfArrayFloat : public ::testing::Test {
protected:
    void SetUp() override {
        kDataLen    = 64;
        in1         = new float[kDataLen];
        in2         = new float[kDataLen];
        result      = new float[kDataLen];
        reference   = new float[kDataLen];

        for (int i=0; i<kDataLen; ++i) {
            in1[i] = float(i);
            in2[i] = float(i) + 100.0f;
            result[i] = -1.0f;
            reference[i] = -2.0f;
        }

        cudaMalloc(&device_in1, kDataLen * sizeof(float));
        cudaMalloc(&device_in2, kDataLen * sizeof(float));
        cudaMalloc(&device_out, kDataLen * sizeof(float));

        cudaMemcpy(device_in1, in1, kDataLen * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_in2, in2, kDataLen * sizeof(float), cudaMemcpyHostToDevice);
    }

    void TearDown() override {
        delete in1;
        delete in2;
        delete result;
        delete reference;

        cudaDeviceReset();
    }

    int kDataLen;
    float *in1;
    float *in2;
    float *result;
    float *reference;

    float *device_in1;
    float *device_in2;
    float *device_out;
};

// int main()
TEST_F(CUDATestOfArrayFloat, Add)
{
    add<<<1, kDataLen>>>(device_in1, device_in2, device_out);
    // Copy output data to host
    cudaDeviceSynchronize();
    cudaMemcpy(result, device_out, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

    // print the results
    for (int i=0; i<kDataLen; ++i) {
        reference[i] = in1[i] + in2[i];
        EXPECT_FLOAT_EQ(result[i], reference[i]);
    }
}

TEST_F(CUDATestOfArrayFloat, Sub)
{
    sub<<<1, kDataLen>>>(device_in1, device_in2, device_out);
    // Copy output data to host
    cudaDeviceSynchronize();
    cudaMemcpy(result, device_out, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

    // print the results
    for (int i=0; i<kDataLen; ++i) {
        reference[i] = in1[i] - in2[i];
        EXPECT_FLOAT_EQ(result[i], reference[i]);
    }
}

TEST_F(CUDATestOfArrayFloat, Mul)
{
    mul<<<1, kDataLen>>>(device_in1, device_in2, device_out);
    // Copy output data to host
    cudaDeviceSynchronize();
    cudaMemcpy(result, device_out, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

    // print the results
    for (int i=0; i<kDataLen; ++i) {
        reference[i] = in1[i] * in2[i];
        EXPECT_FLOAT_EQ(result[i], reference[i]);
    }
}

TEST_F(CUDATestOfArrayFloat, Div)
{
    div<<<1, kDataLen>>>(device_in1, device_in2, device_out);
    // Copy output data to host
    cudaDeviceSynchronize();
    cudaMemcpy(result, device_out, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

    // print the results
    for (int i=0; i<kDataLen; ++i) {
        reference[i] = in1[i] / in2[i];
        EXPECT_FLOAT_EQ(result[i], reference[i]);
    }
}