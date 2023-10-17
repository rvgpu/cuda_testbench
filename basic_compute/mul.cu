#include <cstdio>
#include <gtest/gtest.h>

__global__ void axpy(int a, float *x, float *y)
{
    y[threadIdx.x] = a * x[threadIdx.x];
}

// int main()a
TEST(Hello, BasicAssersions)
{
    const int kDataLen = 4;

    int a = 2;
    float in[kDataLen] = {1.0f, 2.0f, 3.0f, 4.0f};
    float result[kDataLen];
    float reference[kDataLen];

    // Device Function
    float *device_in;
    float *device_out;
    cudaMalloc(&device_in, kDataLen * sizeof(float));
    cudaMalloc(&device_out, kDataLen * sizeof(float));
    cudaMemcpy(device_in, in, kDataLen * sizeof(float), cudaMemcpyHostToDevice);
    axpy<<<1, kDataLen>>>(a, device_in, device_out);

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
