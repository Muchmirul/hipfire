# HIP Kernel Candidate Template
# tools/autokernel-rdna/kernel_lab/templates/hip_kernel_candidate.md
#
# This template is read by an LLM coding agent when AUTHOR_KERNEL=1.
# It describes the contract for generating a new HIP kernel candidate.
# The evaluator (harness) is fixed. Only the candidate kernel file changes.
#
# --- FILL IN target_spec.json FIELDS AT THE TOP BEFORE WRITING THE KERNEL ---

---

## 1. Target Kernel Purpose

You are writing a new HIP kernel implementation for:

  Kernel:  {{KERNEL_NAME}}
  Purpose: {{KERNEL_PURPOSE}}
  Arch:    gfx1201 (AMD Radeon RX 9070 XT, RDNA4, wave32)

This kernel is called in the hipfire inference hot path during every decode step
of Qwen3.5-27B. It accounts for approximately {{RUNTIME_SHARE}}% of total decode
wall-clock time.

A 1.0x improvement here → approximately {{AMDAHL_E2E}}x end-to-end tok/sec gain.

---

## 2. Input / Output Contract

Function signature (must match exactly):

```c
extern "C" __global__ void {{KERNEL_FUNC_NAME}}(
{{KERNEL_ARGS}}
);
```

**Do not change the function name or argument list.** The dispatch path in
`{{DISPATCH_PATH}}` calls this exact symbol. Changing the signature breaks
hipfire's build or produces silent wrong results.

Shapes (Qwen3.5-27B):
- M (output rows):  {{SHAPE_M}}
- K (input cols):   {{SHAPE_K}}
- groups_per_row:   {{GROUPS_PER_ROW}}  (= K / 256)

---

## 3. Quantization Format

Format: {{QUANT_FORMAT}}

Group size: 256 weights per group
Group layout in memory (per group, packed into {{GROUP_BYTES}} bytes):
  - Bytes 0–3:   scale (float32, IEEE 754)
  - Bytes 4–7:   zero-point (float32, IEEE 754)
  - Bytes 8–135: packed weights ({{WEIGHT_BITS}} bits each, {{WEIGHTS_PER_GROUP}} weights)

Weight layout within the packed block:
  - Each uint32 holds 8 weights (4 bits each, little-endian nibble order)
  - Weight w_i = (pk >> (i*4)) & 0xF
  - Dequantized: w_float = scale * (float)w_nibble + zero_point

**Never change the dequantization formula.** The quality gate verifies
byte-exact output against the committed reference. Any change to the
dequant formula will produce a correctness gate failure.

---

## 4. Reference Output

The harness at `{{HARNESS_PATH}}` generates a reference output using the
current hipfire kernel. Your candidate must produce output within tolerance:

  max_abs_diff: {{TOLERANCE_ABS}}
  max_rel_diff: {{TOLERANCE_REL}}

Correctness is checked per output element (float32). A single element
outside tolerance → FAIL → immediate revert.

**The harness is fixed. Do not modify it.** The harness file is:

  tools/autokernel-rdna/kernel_lab/harnesses/{{KERNEL_NAME}}_harness.sh

---

## 5. Memory Layout Assumptions

- A (weight matrix): row-major, packed HFQ4/HFQ6 groups, {{GROUP_BYTES}} bytes/group
- x (input vector): float32, length K, contiguous
- y (output vector): float32, length M, contiguous, write-once
- No aliasing between A, x, y — use __restrict__ on all pointer args
- Grid: (M, 1, 1) — one block per output row
- Block: (32, 1, 1) — wave32 on RDNA4; 32 threads process one output row

**gfx1201 specifics:**
- Wave32 native — avoid wave64 constructs
- 8 KB LDS per CU (usable: 65536 bytes / CU, shared across co-resident waves)
- GDDR6 memory bandwidth is the bottleneck — minimize global traffic
- No tensor cores (WMMA) for GEMV decode — they require batch dimension > 1
- Vectorized loads (uint4 = 128-bit) recommended for the weight stream

---

## 6. What You Are Allowed to Change

- `__launch_bounds__(threads, min_waves)` — tune for gfx1201 occupancy
- Loop structure (unrolling, tiling, reordering)
- Vectorized load width (uint32 → uint2/uint4 if alignment permits)
- LDS usage for input vector reuse (x[] is read once per group — LDS can amortize)
- Accumulator structure (more accumulators to expose ILP)
- Packed weight traversal order (within correctness constraints)
- Prefetch hints (`__builtin_amdgcn_s_prefetch` — use with caution, see Phase 10 notes)

---

## 7. What You Must NOT Change

- Function name: `{{KERNEL_FUNC_NAME}}`
- Argument list (types and order)
- Dequantization formula (scale * nibble + zero_point)
- Group size (256) or group byte layout ({{GROUP_BYTES}} bytes)
- Output semantics: y[row] = sum over all groups of dequant(A[row,g]) · x[g*256..g*256+255]
- The fallback generic kernel `kernels/src/{{KERNEL_NAME}}.hip` — do not touch it
- Any file outside `kernels/src/{{KERNEL_NAME}}.gfx1201.hip` (or gfx12 variant)

---

## 8. How to Compile Your Candidate

The harness compiles with:

```bash
hipcc --offload-arch=gfx1201 -O3 -std=c++17 \
  -fgpu-rdc \
  tools/autokernel-rdna/kernel_lab/generated/{{KERNEL_NAME}}/candidate_N.hip \
  tools/autokernel-rdna/kernel_lab/harnesses/{{KERNEL_NAME}}_harness_main.cpp \
  -o tools/autokernel-rdna/kernel_lab/generated/{{KERNEL_NAME}}/harness_N
```

Or use the harness script which handles compilation automatically:

```bash
CANDIDATE=tools/autokernel-rdna/kernel_lab/generated/{{KERNEL_NAME}}/candidate_N.hip \
  bash tools/autokernel-rdna/kernel_lab/harnesses/{{KERNEL_NAME}}_harness.sh
```

---

## 9. How to Benchmark

The harness outputs:

```
CORRECTNESS: PASS|FAIL
LATENCY_US: <float>
SPEEDUP_VS_REF: <float>
```

The `author_kernel.sh` script reads these and decides accept/reject.

For manual benchmarking against the full hipfire pipeline:

```bash
ARCH=gfx1201 MODEL=qwen3.5:27b BENCH_TRIALS=3 \
  ./tools/autokernel-rdna/run.sh baseline
```

---

## 10. How to Promote Into hipfire

If the standalone harness passes:
1. Copy candidate to `kernels/src/{{KERNEL_NAME}}.gfx1201.hip`
2. `author_kernel.sh` wires dispatch in `crates/rdna-compute/src/kernels.rs`
3. `cargo build --release` runs
4. Correctness gate runs
5. End-to-end tok/sec benchmark runs
6. Accept/revert decision made

---

## 11. How to Revert

```bash
git checkout kernels/src/{{KERNEL_NAME}}.gfx1201.hip
# Or if no gfx1201 variant existed before:
rm kernels/src/{{KERNEL_NAME}}.gfx1201.hip
```

The dispatch in `crates/rdna-compute/src/kernels.rs` falls back to the
generic `{{KERNEL_NAME}}` symbol automatically when no arch variant is
compiled in.

---

## 12. Optimization Suggestions (Ranked by Expected Impact)

For memory-bound decode GEMV on gfx1201:

1. **LDS input vector staging** — Load the x[] segment for each group into LDS
   before the inner dot product. Reduces global reads from O(K*waves) to O(K).
   Requires 256 * 4 = 1 KB of LDS per block (well within budget).

2. **uint4 vectorized weight loads** — Replace `*(uint32_t*)(gp+8+boff)` with
   `*(uint4*)(gp+8+boff*4)` to fetch 128 bits per thread per group.
   Alignment: group data starts at byte 8 within each 136-byte group.
   136 is not a multiple of 16, so cross-group vectorization needs padding or
   per-group base alignment fixup.

3. **8-accumulator ILP** — Extend from 4 accumulators to 8 to expose more
   instruction-level parallelism to the scheduler.

4. **Prefetch next group** — Use `__builtin_amdgcn_s_prefetch` to prefetch
   the next group's data while computing the current group. Phase 10 showed
   this regresses on gfx1201 in the residual path; re-evaluate for the
   base GEMV with longer stride access patterns.

5. **wave32 warp shuffle** — Use `__builtin_amdgcn_ds_bpermute` or
   `__shfl_down_sync` to broadcast scale/zero-point to all threads rather
   than each thread loading from global. Scale and zp are the same for all
   32 threads in a block.

---

## 13. Known Anti-Patterns (DO NOT do these)

- Do not add extra synchronization (__syncthreads) in single-block GEMV
- Do not use atomics for accumulation (breaks determinism)
- Do not change block dimensions to anything other than (32, 1, 1)
- Do not split into multiple kernels (launch overhead dominates at decode)
- Do not use half-precision accumulators (precision loss breaks quality gate)
- Do not use `__builtin_amdgcn_s_prefetch` in the base GEMV without measuring

---

*This template is read by `author_kernel.sh` and the AUTHOR_KERNEL=1 loop.*
*Do not modify this file during a run.*
