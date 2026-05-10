#!/usr/bin/env bash
# tools/autokernel-rdna/kernel_lab/harnesses/gemv_hfq4g256_harness.sh
#
# Fixed standalone harness for gemv_hfq4g256 kernel candidates.
#
# Usage:
#   CANDIDATE=tools/autokernel-rdna/kernel_lab/generated/gemv_hfq4g256/candidate_1.hip \
#     bash tools/autokernel-rdna/kernel_lab/harnesses/gemv_hfq4g256_harness.sh
#
# Output (stdout):
#   CORRECTNESS: PASS|FAIL
#   LATENCY_US: <float>
#   SPEEDUP_VS_REF: <float>
#   ERROR: <message>   (only on failure)
#
# The CANDIDATE file is the ONLY thing that changes between runs.
# This script (the evaluator) must NOT be modified during a run.
#
# Hard rules:
#   - Correctness checked against reference before any timing is reported
#   - Fast-but-wrong kernels are FAIL, no exceptions
#   - If hipcc is not available, exit 2 (infra failure, not candidate failure)
#   - Output format must remain stable — author_kernel.sh parses it

set -uo pipefail

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="$(cd "$TOOL_DIR/../.." && pwd)"

# ── Config ─────────────────────────────────────────────────────────────────
ARCH="${ARCH:-gfx1201}"
CANDIDATE="${CANDIDATE:-}"
HARNESS_TRIALS="${HARNESS_TRIALS:-5}"
TOL_ABS="${TOL_ABS:-1e-3}"
TOL_REL="${TOL_REL:-1e-4}"

# Test shape: use dominant decode shape (down_proj) for representativeness
TEST_M="${TEST_M:-64}"
TEST_K="${TEST_K:-7168}"
# groups_per_row = K / 256 = 28

LAB_DIR="$TOOL_DIR/kernel_lab"
HARNESS_DIR="$LAB_DIR/harnesses"
BUILD_DIR="$LAB_DIR/generated/gemv_hfq4g256"
REF_KERNEL="$REPO_ROOT/kernels/src/gemv_hfq4g256.hip"

# ── Validate inputs ────────────────────────────────────────────────────────
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
    echo "ERROR: hipcc not found — ROCm not installed or not in PATH"
    exit 2
fi

# ── Generate harness C++ driver (written once per invocation) ──────────────
DRIVER_SRC="$BUILD_DIR/_harness_driver.cpp"
cat > "$DRIVER_SRC" <<'CPPEOF'
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
extern "C" void gemv_hfq4g256(const char* A, const float* x, float* y, int M, int K);

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
CPPEOF

# ── Compile reference kernel binary (oracle timing) ────────────────────────
REF_BIN="$BUILD_DIR/_harness_ref"
if [ ! -f "$REF_BIN" ] || [ "$REF_KERNEL" -nt "$REF_BIN" ]; then
    hipcc --offload-arch="$ARCH" -O3 -std=c++17 \
        -DKERNEL_IS_REFERENCE=1 \
        "$REF_KERNEL" "$DRIVER_SRC" \
        -o "$REF_BIN" 2>"$BUILD_DIR/_ref_build.log"
    if [ $? -ne 0 ]; then
        echo "CORRECTNESS: FAIL"
        echo "ERROR: Reference kernel failed to compile — check $BUILD_DIR/_ref_build.log"
        exit 2
    fi
fi

# ── Get reference timing ───────────────────────────────────────────────────
ref_output=$(TEST_M="$TEST_M" TEST_K="$TEST_K" HARNESS_TRIALS="$HARNESS_TRIALS" \
    TOL_ABS="$TOL_ABS" "$REF_BIN" 2>/dev/null)
ref_latency=$(echo "$ref_output" | grep '^LATENCY_US:' | awk '{print $2}')
if [ -z "$ref_latency" ]; then
    echo "CORRECTNESS: FAIL"
    echo "ERROR: Reference binary failed to produce LATENCY_US output"
    exit 2
fi

# ── Compile candidate kernel binary ───────────────────────────────────────
CAND_BASENAME="$(basename "$CANDIDATE" .hip)"
CAND_BIN="$BUILD_DIR/_harness_${CAND_BASENAME}"
BUILD_LOG="$BUILD_DIR/_build_${CAND_BASENAME}.log"

hipcc --offload-arch="$ARCH" -O3 -std=c++17 \
    "$CANDIDATE" "$DRIVER_SRC" \
    -o "$CAND_BIN" 2>"$BUILD_LOG"
if [ $? -ne 0 ]; then
    echo "CORRECTNESS: FAIL"
    echo "ERROR: Candidate build failed — see $BUILD_LOG"
    exit 1
fi

# ── Run candidate harness ─────────────────────────────────────────────────
cand_output=$(TEST_M="$TEST_M" TEST_K="$TEST_K" HARNESS_TRIALS="$HARNESS_TRIALS" \
    TOL_ABS="$TOL_ABS" "$CAND_BIN" 2>/dev/null)

echo "$cand_output"

# ── Parse and compute speedup ─────────────────────────────────────────────
cand_correctness=$(echo "$cand_output" | grep '^CORRECTNESS:' | awk '{print $2}')
cand_latency=$(echo "$cand_output" | grep '^LATENCY_US:' | awk '{print $2}')

if [ "$cand_correctness" != "PASS" ]; then
    # Error already printed above
    exit 1
fi

if [ -n "$cand_latency" ] && [ -n "$ref_latency" ]; then
    speedup=$(python3 -c "print(f'{float($ref_latency)/float($cand_latency):.4f}')" 2>/dev/null || echo "0")
    echo "REF_LATENCY_US: $ref_latency"
    echo "SPEEDUP_VS_REF: $speedup"
else
    echo "SPEEDUP_VS_REF: 0"
fi
