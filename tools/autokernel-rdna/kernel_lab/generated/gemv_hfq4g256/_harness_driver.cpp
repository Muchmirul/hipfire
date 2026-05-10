// Auto-generated harness driver for gemv_hfq4g256 standalone testing.
// Fixed evaluator — do NOT modify.
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <chrono>

#define HIP_CHECK(call) do { \
    hipError_t e = (call); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

// Declaration — provided by the kernel being tested.
extern "C" __global__ void gemv_hfq4g256(const char* A, const float* x, float* y, int M, int K);

// Reference implementation (CPU, exact arithmetic reference).
static void cpu_gemv_hfq4g256_ref(const char* A, const float* x, float* y_ref, int M, int K) {
    const int groups_per_row = K / 256;
    for (int row = 0; row < M; row++) {
        const char* row_ptr = A + (long long)row * groups_per_row * 136;
        double acc = 0.0;
        for (int g = 0; g < groups_per_row; g++) {
            const char* gp = row_ptr + g * 136;
            float sc = 0.0f, zp = 0.0f;
            memcpy(&sc, gp,     4);
            memcpy(&zp, gp + 4, 4);
            for (int i = 0; i < 32; i++) {
                unsigned int pk = 0;
                memcpy(&pk, gp + 8 + i * 4, 4);
                for (int n = 0; n < 8; n++) {
                    float w = sc * (float)((pk >> (n * 4)) & 0xFu) + zp;
                    acc += (double)w * (double)x[g * 256 + i * 8 + n];
                }
            }
        }
        y_ref[row] = (float)acc;
    }
}

int main(int argc, char* argv[]) {
    int M = atoi(getenv("TEST_M") ? getenv("TEST_M") : "64");
    int K = atoi(getenv("TEST_K") ? getenv("TEST_K") : "7168");
    int trials = atoi(getenv("HARNESS_TRIALS") ? getenv("HARNESS_TRIALS") : "5");
    float tol_abs = atof(getenv("TOL_ABS") ? getenv("TOL_ABS") : "1e-3");

    const int groups_per_row = K / 256;
    const long long A_bytes = (long long)M * groups_per_row * 136;
    const long long x_bytes = (long long)K * sizeof(float);
    const long long y_bytes = (long long)M * sizeof(float);

    // Allocate and fill host buffers
    char*  h_A    = (char*)  malloc(A_bytes);
    float* h_x    = (float*) malloc(x_bytes);
    float* h_y    = (float*) malloc(y_bytes);
    float* h_yref = (float*) malloc(y_bytes);

    // Fill A with synthetic but realistic packed HFQ4 data
    srand(42);
    for (long long row = 0; row < M; row++) {
        char* row_ptr = h_A + row * groups_per_row * 136;
        for (int g = 0; g < groups_per_row; g++) {
            char* gp = row_ptr + g * 136;
            float sc = 0.01f + (rand() % 100) * 0.001f;
            float zp = -0.5f + (rand() % 100) * 0.01f;
            memcpy(gp,     &sc, 4);
            memcpy(gp + 4, &zp, 4);
            for (int i = 8; i < 136; i++) {
                gp[i] = (char)(rand() & 0xFF);
            }
        }
    }
    for (int i = 0; i < K; i++) {
        h_x[i] = -1.0f + (rand() % 2000) * 0.001f;
    }

    // Compute CPU reference
    cpu_gemv_hfq4g256_ref(h_A, h_x, h_yref, M, K);

    // Allocate device buffers
    char*  d_A; float* d_x; float* d_y;
    HIP_CHECK(hipMalloc(&d_A, A_bytes));
    HIP_CHECK(hipMalloc(&d_x, x_bytes));
    HIP_CHECK(hipMalloc(&d_y, y_bytes));
    HIP_CHECK(hipMemcpy(d_A, h_A, A_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x, x_bytes, hipMemcpyHostToDevice));

    // Correctness check (1 run)
    memset(h_y, 0, y_bytes);
    HIP_CHECK(hipMemset(d_y, 0, y_bytes));
    gemv_hfq4g256<<<M, 32>>>(d_A, d_x, d_y, M, K);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_y, d_y, y_bytes, hipMemcpyDeviceToHost));

    float max_abs_err = 0.0f;
    int fail_idx = -1;
    for (int i = 0; i < M; i++) {
        float err = fabsf(h_y[i] - h_yref[i]);
        if (err > max_abs_err) { max_abs_err = err; fail_idx = i; }
    }

    bool correct = (max_abs_err <= tol_abs);
    if (!correct) {
        printf("CORRECTNESS: FAIL\n");
        printf("ERROR: max_abs_err=%.6f at row %d (y=%.6f ref=%.6f) tol=%.6f\n",
               max_abs_err, fail_idx, h_y[fail_idx], h_yref[fail_idx], tol_abs);
        return 1;
    }
    printf("CORRECTNESS: PASS\n");

    // Timing (warmup + trials)
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));

    // Warmup
    for (int t = 0; t < 3; t++) {
        gemv_hfq4g256<<<M, 32>>>(d_A, d_x, d_y, M, K);
    }
    HIP_CHECK(hipDeviceSynchronize());

    float total_ms = 0.0f;
    for (int t = 0; t < trials; t++) {
        HIP_CHECK(hipEventRecord(t0));
        gemv_hfq4g256<<<M, 32>>>(d_A, d_x, d_y, M, K);
        HIP_CHECK(hipEventRecord(t1));
        HIP_CHECK(hipEventSynchronize(t1));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        total_ms += ms;
    }
    float avg_us = (total_ms / trials) * 1000.0f;
    printf("LATENCY_US: %.2f\n", avg_us);

    // Cleanup
    hipFree(d_A); hipFree(d_x); hipFree(d_y);
    free(h_A); free(h_x); free(h_y); free(h_yref);
    return 0;
}
