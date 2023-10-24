#include <cmath>
#include <gtest/gtest.h>

#define N 32

class BasicTest : public ::testing::Test {
protected:
    void SetUp() override {
        in1 = (int *) malloc(N * sizeof(int));
        in2 = (int *) malloc(N * sizeof(int));
        in3 = (int *) malloc(N * sizeof(int));
        out = (int *) malloc(N * sizeof(int));
        cudaMalloc(&device_in1, N * sizeof(int));
        cudaMalloc(&device_in2, N * sizeof(int));
        cudaMalloc(&device_in3, N * sizeof(int));
        cudaMalloc(&device_out, N * sizeof(int));

        fin1 = (float *) malloc(N * sizeof(float));
        fin2 = (float *) malloc(N * sizeof(float));
        fin3 = (float *) malloc(N * sizeof(float));
        fout = (float *) malloc(N * sizeof(float));
        cudaMalloc(&device_fin1, N * sizeof(float));
        cudaMalloc(&device_fin2, N * sizeof(float));
        cudaMalloc(&device_fin3, N * sizeof(float));
        cudaMalloc(&device_fout, N * sizeof(float));
    };

    void TearDown() override {
        free(in1); free(in2); free(in3); free(out);
        cudaFree(device_in1);
        cudaFree(device_in2);
        cudaFree(device_in3);
        cudaFree(device_out);

        free(fin1); free(fin2); free(fin3); free(fout);
        cudaFree(device_fin1);
        cudaFree(device_fin2);
        cudaFree(device_fin3);
        cudaFree(device_fout);
    };

    int *in1, *in2, *in3, *out;
    int *device_in1, *device_in2, *device_in3, *device_out;

    float *fin1, *fin2, *fin3, *fout;
    float *device_fin1, *device_fin2, *device_fin3, *device_fout;
};



/* Kernels
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
 */

__global__ void array_add(int *input1, int *output) {
    int i = threadIdx.x;
    output[i] = input1[i] + 100;
}

__global__ void multi_array_add(int *input1, int *input2, int *output) {
    int i = threadIdx.x;
    output[i] = input1[i] + input2[i];
}

__global__ void multi_array_mul(int *input1, int *input2, int *output) {
    int i = threadIdx.x;
    output[i] = input1[i] * input2[i];
}

__global__ void multi_array_muladd(int *input1, int *input2, int *input3, int *output) {
    int i = threadIdx.x;
    output[i] = input1[i] * input2[i] + input3[i];
}

__global__ void multi_array_fmuladd(float *input1, float *input2, float *input3, float *output) {
    int i = threadIdx.x;
    output[i] = input1[i] * input2[i] + input3[i];
}

__global__ void branch_if(int *input1, int *output) {
    int i = threadIdx.x;
    output[i] = input1[i];
    if (i % 2 == 0) {
        output[i] += 100;
    }
}

__global__ void branch_ifelse(int *input1, int *output) {
    int i = threadIdx.x;
    if (i % 2 == 0) {
        output[i] = input1[i] + 100;
    } else {
        output[i] = input1[i] + 200;
    }
}

__global__ void branch_ifif(int *input1, int *output) {
    int i = threadIdx.x;
    if (i % 2 == 0) {
        if (i % 4 == 0) {
            output[i] = input1[i] + 400;
        } else {
            output[i] = input1[i] + 200;
        }
    } else {
        output[i] = input1[i] + 100;
    }
}

__global__ void branch_for(int *input1, int *output) {
    int i = threadIdx.x;
    output[i] = input1[i];
    for (uint32_t j = 0; j < i; j++) {
        output[i] += j;
    }
}

__global__ void math_sin(float *input1, float *output) {
    int i = threadIdx.x;
    output[i] = sin(input1[i]);
}

__global__ void math_cos(float *input1, float *output) {
    int i = threadIdx.x;
    output[i] = cos(input1[i]);
}

__global__ void math_pow(float *input1, float *input2, float *output) {
    int i = threadIdx.x;
    output[i] = pow(input1[i], input2[i]);
}



/* Tests
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
 */

TEST_F(BasicTest, array_add) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        in1[i] = i * 100;
    }

    // Execute
    cudaMemcpy(device_in1, in1, N * sizeof(int), cudaMemcpyHostToDevice);

    array_add<<<1, N>>>(device_in1, device_out);

    cudaMemcpy(out, device_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_EQ(out[i], (i * 100) + 100);
    }
}

TEST_F(BasicTest, multi_array_add) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        in1[i] = i * 100;
        in2[i] = i + 34;
    }

    // Execute
    cudaMemcpy(device_in1, in1, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_in2, in2, N * sizeof(int), cudaMemcpyHostToDevice);

    multi_array_add<<<1, N>>>(device_in1, device_in2, device_out);

    cudaMemcpy(out, device_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_EQ(out[i], (i * 100) + (i + 34));
    }
}

TEST_F(BasicTest, multi_array_mul) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        in1[i] = i * 100;
        in2[i] = i + 34;
    }

    // Execute
    cudaMemcpy(device_in1, in1, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_in2, in2, N * sizeof(int), cudaMemcpyHostToDevice);

    multi_array_mul<<<1, N>>>(device_in1, device_in2, device_out);

    cudaMemcpy(out, device_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_EQ(out[i], (i * 100) * (i + 34));
    }
}

TEST_F(BasicTest, multi_array_muladd) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        in1[i] = i * 100;
        in2[i] = i + 34;
        in3[i] = i * 4;
    }

    // Execute
    cudaMemcpy(device_in1, in1, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_in2, in2, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_in3, in3, N * sizeof(int), cudaMemcpyHostToDevice);
    multi_array_muladd<<<1, N>>>(device_in1, device_in2, device_in3, device_out);
    cudaMemcpy(out, device_out, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_EQ(out[i], (i * 100) * (i + 34) + (i * 4));
    }
}

TEST_F(BasicTest, multi_array_fmuladd) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        fin1[i] = i * 100.0f;
        fin2[i] = i + 34.0f;
        fin3[i] = i * 4.0f;
    }

    // Execute
    cudaMemcpy(device_fin1, fin1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_fin2, fin2, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_fin3, fin3, N * sizeof(float), cudaMemcpyHostToDevice);

    multi_array_fmuladd<<<1, N>>>(device_fin1, device_fin2, device_fin3, device_fout);

    cudaMemcpy(fout, device_fout, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(fout[i], (i * 100.0f) * (i + 34.0f) + (i * 4.0f));
    }
}

TEST_F(BasicTest, branch_if) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        in1[i] = i * 100.0;
    }

    // Execute
    cudaMemcpy(device_in1, in1, N * sizeof(float), cudaMemcpyHostToDevice);

    branch_if<<<1, N>>>(device_in1, device_out);

    cudaMemcpy(out, device_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        if (i % 2 == 0) {
            EXPECT_EQ(out[i], (i * 100) + 100);
        } else {
            EXPECT_EQ(out[i], (i * 100));
        }
    }
}

TEST_F(BasicTest, branch_ifelse) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        in1[i] = i * 100.0;
    }

    // Execute
    cudaMemcpy(device_in1, in1, N * sizeof(float), cudaMemcpyHostToDevice);

    branch_ifelse<<<1, N>>>(device_in1, device_out);

    cudaMemcpy(out, device_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        if (i % 2 == 0) {
            EXPECT_EQ(out[i], (i * 100) + 100);
        } else {
            EXPECT_EQ(out[i], (i * 100) + 200);
        }
    }
}

TEST_F(BasicTest, branch_ifif) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        in1[i] = i * 100.0;
    }

    // Execute
    cudaMemcpy(device_in1, in1, N * sizeof(float), cudaMemcpyHostToDevice);

    branch_ifif<<<1, N>>>(device_in1, device_out);

    cudaMemcpy(out, device_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        if (i % 2 == 0) {
            if (i % 4 == 0) {
                EXPECT_EQ(out[i], (i * 100) + 400);
            } else {
                EXPECT_EQ(out[i], (i * 100) + 200);
            }
        } else {
            EXPECT_EQ(out[i], (i * 100) + 100);
        }
    }
}

TEST_F(BasicTest, branch_for) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        in1[i] = i * 100.0;
    }

    // Execute
    cudaMemcpy(device_in1, in1, N * sizeof(float), cudaMemcpyHostToDevice);

    branch_for<<<1, N>>>(device_in1, device_out);

    cudaMemcpy(out, device_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        int reference = i * 100;
        for (uint32_t j = 0; j < i; j++) {
            reference += j;
        }
        EXPECT_EQ(out[i], reference);
    }
}

TEST_F(BasicTest, math_sin) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        fin1[i] = i * 1.1f;
    }

    // Execute
    cudaMemcpy(device_fin1, fin1, N * sizeof(float), cudaMemcpyHostToDevice);

    math_sin<<<1, N>>>(device_fin1, device_fout);

    cudaMemcpy(fout, device_fout, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(fout[i], std::sin(i * 1.1f));
    }
}

TEST_F(BasicTest, math_cos) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        fin1[i] = i * 1.1f;
    }

    // Execute
    cudaMemcpy(device_fin1, fin1, N * sizeof(float), cudaMemcpyHostToDevice);

    math_cos<<<1, N>>>(device_fin1, device_fout);

    cudaMemcpy(fout, device_fout, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(fout[i], std::cos(i * 1.1f));
    }
}

TEST_F(BasicTest, math_pow) {
    // Initialize
    for (uint32_t i = 0; i < N; i++) {
        fin1[i] = i * 1.1f;
        fin2[i] = i * 2.3f;
    }

    // Execute
    cudaMemcpy(device_fin1, fin1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(device_fin2, fin2, N * sizeof(float), cudaMemcpyHostToDevice);

    math_pow<<<1, N>>>(device_fin1, device_fin2, device_fout);

    cudaMemcpy(fout, device_fout, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Test
    for (uint32_t i = 0; i < N; i++) {
        EXPECT_FLOAT_EQ(fout[i], std::pow(i * 1.1f, i * 2.3f));
    }
}