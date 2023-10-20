#include <cstdio>
#include <gtest/gtest.h>

__global__ void mul(float *in1, float *in2, float *out)
{
    out[threadIdx.x] = in1[threadIdx.x] * in2[threadIdx.x];
}

class TestNThreads : public ::testing::Test {
protected:
    void SetUp() override {
        in1         = new float[maxThreads];
        in2         = new float[maxThreads];
        result      = new float[maxThreads];  
        reference   = new float[maxThreads];

        cudaMalloc(&device_in1, maxThreads * sizeof(float));
        cudaMalloc(&device_in2, maxThreads * sizeof(float));
        cudaMalloc(&device_out, maxThreads * sizeof(float));  
    };

    void TearDown() {
        delete in1;
        delete in2;
        delete result;
        delete reference;
    };

    void run(uint32_t nthread) {
        for (int i=0; i<nthread; ++i) {
            in1[i] = float(i) + 1.1f;
            in2[i] = float(i) + 2.2f;
            result[i] = -1.0f;
            reference[i] = -2.0f;
        }

        cudaMemcpy(device_in1, in1, nthread * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(device_in2, in2, nthread * sizeof(float), cudaMemcpyHostToDevice);
        mul<<<1, nthread>>>(device_in1, device_in2, device_out);

        cudaDeviceSynchronize();
        cudaMemcpy(result, device_out, nthread * sizeof(float), cudaMemcpyDeviceToHost);

        for (int i=0; i<nthread; ++i) {
            reference[i] = in1[i] * in2[i];
            EXPECT_FLOAT_EQ(result[i], reference[i]);
        }
    }

    int maxThreads = 4096;
    float *in1;
    float *in2;
    float *result;
    float *reference;

    float *device_in1;
    float *device_in2;
    float *device_out;

};

// int main()a
TEST_F(TestNThreads, NumThreads)
{
    run(1);
    run(2);
    run(3);
    run(15);
    run(16);
    run(17);
    run(18);
    run(30);
    run(31);
    run(32);
    run(33);
    run(34);
    run(63);
    run(64);
    run(65);
    run(128);
    run(256);
    run(512);
    run(1024);
    run(2048);
    run(2345);
}
