#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#define HIP_CHECK(call) do { \
    hipError_t e = (call); \
    if (e != hipSuccess) { \
        fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

extern "C" __global__ void gemv_hfq6g256(const char* A, const float* x, float* y, int M, int K);

// CPU reference for HFQ6-G256.
// HFQ6 bit packing: 256 weights * 6 bits = 1536 bits = 192 bytes payload + 8 bytes header.
// 4 weights packed per 3 bytes (24 bits), little-endian.
// w_i (0..3 within 3-byte chunk): (chunk >> (i*6)) & 0x3F
static void cpu_ref(const char* A, const float* x, float* y_ref, int M, int K) {
    const int groups_per_row = K / 256;
    const int GROUP_BYTES = 200;  // 8 header + 192 payload
    for (int row = 0; row < M; row++) {
        const char* row_ptr = A + (long long)row * groups_per_row * GROUP_BYTES;
        double acc = 0.0;
        for (int g = 0; g < groups_per_row; g++) {
            const char* gp = row_ptr + g * GROUP_BYTES;
            float sc = 0.0f, zp = 0.0f;
            memcpy(&sc, gp, 4);
            memcpy(&zp, gp + 4, 4);
            const unsigned char* payload = (const unsigned char*)(gp + 8);
            for (int wi = 0; wi < 256; wi++) {
                // 4 weights per 3 bytes: chunk_idx = wi / 4, bit_offset = (wi % 4) * 6
                int chunk_idx = wi / 4;
                int bit_off   = (wi % 4) * 6;
                // Read 3 bytes as a 24-bit little-endian chunk
                unsigned int chunk = (unsigned int)payload[chunk_idx * 3]
                    | ((unsigned int)payload[chunk_idx * 3 + 1] << 8)
                    | ((unsigned int)payload[chunk_idx * 3 + 2] << 16);
                unsigned int nibble = (chunk >> bit_off) & 0x3Fu;
                float w = sc * (float)nibble + zp;
                acc += (double)w * (double)x[g * 256 + wi];
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
    const int GROUP_BYTES = 200;
    const long long A_bytes = (long long)M * groups_per_row * GROUP_BYTES;

    char*  h_A    = (char*)  malloc(A_bytes);
    float* h_x    = (float*) malloc(K * sizeof(float));
    float* h_y    = (float*) malloc(M * sizeof(float));
    float* h_yref = (float*) malloc(M * sizeof(float));

    srand(42);
    for (long long row = 0; row < M; row++) {
        char* row_ptr = h_A + row * groups_per_row * GROUP_BYTES;
        for (int g = 0; g < groups_per_row; g++) {
            char* gp = row_ptr + g * GROUP_BYTES;
            float sc = 0.01f + (rand() % 100) * 0.001f;
            float zp = -0.5f + (rand() % 100) * 0.01f;
            memcpy(gp, &sc, 4); memcpy(gp + 4, &zp, 4);
            for (int i = 8; i < GROUP_BYTES; i++) gp[i] = (char)(rand() & 0xFF);
        }
    }
    for (int i = 0; i < K; i++) h_x[i] = -1.0f + (rand() % 2000) * 0.001f;

    cpu_ref(h_A, h_x, h_yref, M, K);

    char*  d_A; float* d_x; float* d_y;
    HIP_CHECK(hipMalloc(&d_A, A_bytes));
    HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A, h_A, A_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x, K * sizeof(float), hipMemcpyHostToDevice));

    HIP_CHECK(hipMemset(d_y, 0, M * sizeof(float)));
    gemv_hfq6g256<<<M, 32>>>(d_A, d_x, d_y, M, K);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_y, d_y, M * sizeof(float), hipMemcpyDeviceToHost));

    float max_err = 0.0f; int fail_idx = -1;
    for (int i = 0; i < M; i++) {
        float err = fabsf(h_y[i] - h_yref[i]);
        if (err > max_err) { max_err = err; fail_idx = i; }
    }
    if (max_err > tol_abs) {
        printf("CORRECTNESS: FAIL\n");
        printf("ERROR: max_abs_err=%.6f at row %d (y=%.6f ref=%.6f)\n",
               max_err, fail_idx, h_y[fail_idx], h_yref[fail_idx]);
        return 1;
    }
    printf("CORRECTNESS: PASS\n");

    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    for (int t = 0; t < 3; t++) gemv_hfq6g256<<<M, 32>>>(d_A, d_x, d_y, M, K);
    HIP_CHECK(hipDeviceSynchronize());

    float total_ms = 0.0f;
    for (int t = 0; t < trials; t++) {
        HIP_CHECK(hipEventRecord(t0));
        gemv_hfq6g256<<<M, 32>>>(d_A, d_x, d_y, M, K);
        HIP_CHECK(hipEventRecord(t1));
        HIP_CHECK(hipEventSynchronize(t1));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        total_ms += ms;
    }
    printf("LATENCY_US: %.2f\n", (total_ms / trials) * 1000.0f);

    hipFree(d_A); hipFree(d_x); hipFree(d_y);
    free(h_A); free(h_x); free(h_y); free(h_yref);
    return 0;
}
