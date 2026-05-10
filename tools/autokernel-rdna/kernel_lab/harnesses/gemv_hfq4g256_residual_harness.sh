#!/usr/bin/env bash
# tools/autokernel-rdna/kernel_lab/harnesses/gemv_hfq4g256_residual_harness.sh
#
# Fixed standalone harness for gemv_hfq4g256_residual kernel candidates.
#
# Usage:
#   CANDIDATE=tools/autokernel-rdna/kernel_lab/generated/gemv_hfq4g256_residual/candidate_1.hip \
#     bash tools/autokernel-rdna/kernel_lab/harnesses/gemv_hfq4g256_residual_harness.sh
#
# Output (stdout):
#   CORRECTNESS: PASS|FAIL
#   LATENCY_US: <float>
#   REF_LATENCY_US: <float>
#   SPEEDUP_VS_REF: <float>
#   ERROR: <message>   (only on failure)
#
# This kernel adds residual: y = gemv(A, x) + residual
# Correctness checked against CPU reference with same tolerances.

set -uo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "$TOOL_DIR/../.." && pwd)"

ARCH="${ARCH:-gfx1201}"
CANDIDATE="${CANDIDATE:-}"
HARNESS_TRIALS="${HARNESS_TRIALS:-5}"
TOL_ABS="${TOL_ABS:-1e-3}"
TEST_M="${TEST_M:-64}"
TEST_K="${TEST_K:-7168}"

LAB_DIR="$TOOL_DIR/kernel_lab"
BUILD_DIR="$LAB_DIR/generated/gemv_hfq4g256_residual"
REF_KERNEL="$REPO_ROOT/kernels/src/gemv_hfq4g256_residual.hip"

if [ -z "$CANDIDATE" ]; then
    echo "CORRECTNESS: FAIL"
    echo "ERROR: CANDIDATE env var is not set"
    exit 1
fi
if [ ! -f "$CANDIDATE" ]; then
    echo "CORRECTNESS: FAIL"
    echo "ERROR: Candidate file not found: $CANDIDATE"
    exit 1
fi
if ! command -v hipcc >/dev/null 2>&1; then
    echo "CORRECTNESS: FAIL"
    echo "ERROR: hipcc not found"
    exit 2
fi

DRIVER_SRC="$BUILD_DIR/_harness_driver.cpp"
cat > "$DRIVER_SRC" <<'CPPEOF'
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

extern "C" __global__ void gemv_hfq4g256_residual(
    const char* A, const float* x, const float* residual, float* y, int M, int K);

static void cpu_ref(const char* A, const float* x, const float* residual,
                    float* y_ref, int M, int K) {
    const int groups_per_row = K / 256;
    for (int row = 0; row < M; row++) {
        const char* row_ptr = A + (long long)row * groups_per_row * 136;
        double acc = 0.0;
        for (int g = 0; g < groups_per_row; g++) {
            const char* gp = row_ptr + g * 136;
            float sc = 0.0f, zp = 0.0f;
            memcpy(&sc, gp, 4);
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
        y_ref[row] = (float)acc + residual[row];
    }
}

int main(int argc, char* argv[]) {
    int M = atoi(getenv("TEST_M") ? getenv("TEST_M") : "64");
    int K = atoi(getenv("TEST_K") ? getenv("TEST_K") : "7168");
    int trials = atoi(getenv("HARNESS_TRIALS") ? getenv("HARNESS_TRIALS") : "5");
    float tol_abs = atof(getenv("TOL_ABS") ? getenv("TOL_ABS") : "1e-3");

    const int groups_per_row = K / 256;
    const long long A_bytes = (long long)M * groups_per_row * 136;

    char*  h_A    = (char*)  malloc(A_bytes);
    float* h_x    = (float*) malloc(K * sizeof(float));
    float* h_r    = (float*) malloc(M * sizeof(float));
    float* h_y    = (float*) malloc(M * sizeof(float));
    float* h_yref = (float*) malloc(M * sizeof(float));

    srand(42);
    for (long long row = 0; row < M; row++) {
        char* row_ptr = h_A + row * groups_per_row * 136;
        for (int g = 0; g < groups_per_row; g++) {
            char* gp = row_ptr + g * 136;
            float sc = 0.01f + (rand() % 100) * 0.001f;
            float zp = -0.5f + (rand() % 100) * 0.01f;
            memcpy(gp, &sc, 4); memcpy(gp + 4, &zp, 4);
            for (int i = 8; i < 136; i++) gp[i] = (char)(rand() & 0xFF);
        }
    }
    for (int i = 0; i < K; i++) h_x[i] = -1.0f + (rand() % 2000) * 0.001f;
    for (int i = 0; i < M; i++) h_r[i] = -0.1f + (rand() % 200)  * 0.001f;

    cpu_ref(h_A, h_x, h_r, h_yref, M, K);

    char*  d_A; float* d_x; float* d_r; float* d_y;
    HIP_CHECK(hipMalloc(&d_A, A_bytes));
    HIP_CHECK(hipMalloc(&d_x, K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_r, M * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_y, M * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_A, h_A, A_bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_x, h_x, K * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_r, h_r, M * sizeof(float), hipMemcpyHostToDevice));

    // Correctness
    HIP_CHECK(hipMemset(d_y, 0, M * sizeof(float)));
    gemv_hfq4g256_residual<<<M, 32>>>(d_A, d_x, d_r, d_y, M, K);
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
    for (int t = 0; t < 3; t++)
        gemv_hfq4g256_residual<<<M, 32>>>(d_A, d_x, d_r, d_y, M, K);
    HIP_CHECK(hipDeviceSynchronize());

    float total_ms = 0.0f;
    for (int t = 0; t < trials; t++) {
        HIP_CHECK(hipEventRecord(t0));
        gemv_hfq4g256_residual<<<M, 32>>>(d_A, d_x, d_r, d_y, M, K);
        HIP_CHECK(hipEventRecord(t1));
        HIP_CHECK(hipEventSynchronize(t1));
        float ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
        total_ms += ms;
    }
    printf("LATENCY_US: %.2f\n", (total_ms / trials) * 1000.0f);

    hipFree(d_A); hipFree(d_x); hipFree(d_r); hipFree(d_y);
    free(h_A); free(h_x); free(h_r); free(h_y); free(h_yref);
    return 0;
}
CPPEOF

# Reference binary
REF_BIN="$BUILD_DIR/_harness_ref"
if [ ! -f "$REF_BIN" ] || [ "$REF_KERNEL" -nt "$REF_BIN" ]; then
    hipcc --offload-arch="$ARCH" -O3 -std=c++17 \
        "$REF_KERNEL" "$DRIVER_SRC" -o "$REF_BIN" 2>"$BUILD_DIR/_ref_build.log"
    if [ $? -ne 0 ]; then
        echo "CORRECTNESS: FAIL"
        echo "ERROR: Reference kernel failed to compile"
        exit 2
    fi
fi

ref_output=$(TEST_M="$TEST_M" TEST_K="$TEST_K" HARNESS_TRIALS="$HARNESS_TRIALS" \
    TOL_ABS="$TOL_ABS" "$REF_BIN" 2>/dev/null)
ref_latency=$(echo "$ref_output" | grep '^LATENCY_US:' | awk '{print $2}')

# Candidate
CAND_BASENAME="$(basename "$CANDIDATE" .hip)"
CAND_BIN="$BUILD_DIR/_harness_${CAND_BASENAME}"
hipcc --offload-arch="$ARCH" -O3 -std=c++17 \
    "$CANDIDATE" "$DRIVER_SRC" -o "$CAND_BIN" 2>"$BUILD_DIR/_build_${CAND_BASENAME}.log"
if [ $? -ne 0 ]; then
    echo "CORRECTNESS: FAIL"
    echo "ERROR: Candidate build failed"
    exit 1
fi

cand_output=$(TEST_M="$TEST_M" TEST_K="$TEST_K" HARNESS_TRIALS="$HARNESS_TRIALS" \
    TOL_ABS="$TOL_ABS" "$CAND_BIN" 2>/dev/null)
echo "$cand_output"

cand_correctness=$(echo "$cand_output" | grep '^CORRECTNESS:' | awk '{print $2}')
cand_latency=$(echo "$cand_output" | grep '^LATENCY_US:' | awk '{print $2}')

if [ "$cand_correctness" != "PASS" ]; then exit 1; fi

if [ -n "$cand_latency" ] && [ -n "$ref_latency" ]; then
    speedup=$(python3 -c "print(f'{float($ref_latency)/float($cand_latency):.4f}')" 2>/dev/null || echo "0")
    echo "REF_LATENCY_US: $ref_latency"
    echo "SPEEDUP_VS_REF: $speedup"
else
    echo "SPEEDUP_VS_REF: 0"
fi
