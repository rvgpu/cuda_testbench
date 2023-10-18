#include <cstdio>
#include <gtest/gtest.h>

__global__ void mul(int a, float *x, float *y)
{
    y[threadIdx.x] = a * x[threadIdx.x];
}

// int main()a
TEST(BasicCudaTest, MultiAnInteger)
{
    int kDataLen = 64;

    int a = 2;
    float *in                   = new float[kDataLen];
    float *result               = new float[kDataLen];
    float *reference            = new float[kDataLen];
    for (int i=0; i<kDataLen; ++i) {
        in[i] = float(i);
        result[i] = -1.0f;
        reference[i] = -2.0f;
    }

    // Device Function
    float *device_in;
    float *device_out;
    cudaMalloc(&device_in, kDataLen * sizeof(float));
    cudaMalloc(&device_out, kDataLen * sizeof(float));
    cudaMemcpy(device_in, in, kDataLen * sizeof(float), cudaMemcpyHostToDevice);
    mul<<<1, kDataLen>>>(a, device_in, device_out);

    // Copy output data to host
    cudaDeviceSynchronize();
    cudaMemcpy(result, device_out, kDataLen * sizeof(float), cudaMemcpyDeviceToHost);

    // print the results
    for (int i=0; i<kDataLen; ++i) {
        reference[i] = a * in[i];
        EXPECT_FLOAT_EQ(result[i], reference[i]);
    }

    cudaDeviceReset();
}
