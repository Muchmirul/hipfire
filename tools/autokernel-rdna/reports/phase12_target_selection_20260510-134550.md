# Phase 12 — Target Selection Report
## AutoKernel gfx1201 (RX 9070 XT) — gemv_hfq4g256 8x-unroll optimization

Generated: 2026-05-10T13:45:50Z  
Branch: `autokernel/phase12-gfx1201-gemv-hfq4g256`  
Baseline capture: `tools/autokernel-rdna/baselines/20260510-134440.json`  
Baseline decode: **93.1 tok/s** (σ=0.0471, 3 trials, gfx1201, qwen3.5:9b, lru_cache_pep8_strict.txt)  
Prompt MD5: `df5dedc8040ce70ba55080c4548e6024`

---

## Target Selection

**Selected target:** `gemv_hfq4g256`

### Rationale

Profile data from `reports/profile_20260510-122608.md`:

| Rank | Kernel | Dec% | Amdahl (1.3x) |
|------|--------|------|---------------|
| 1 | `gemv_hfq4g256` | 20% | 1.0484x |
| 2 | `gemv_hfq6g256` | 18% | 1.0433x |
| 3 | `gemv_hfq4g256_residual` | 12% | 1.0285x |

`gemv_hfq4g256` is the highest-Amdahl-leverage kernel at 20% of decode
cycles, ahead of every other candidate. It is called on every linear
layer during autoregressive decode: Q, K, V projections, output
projection, and all MLP projections.

Rank 3 (`gemv_hfq4g256_residual`) was attempted in Phase 10 (experiment
`exp-20260510-001`) and REVERTED because `s_prefetch_data` added
overhead on gfx1201. That negative result directly informs this
experiment's strategy.

---

## Problem Analysis

### Existing gfx1201 kernel (`kernels/src/gemv_hfq4g256.gfx1201.hip`)

The existing file was created but **never wired into dispatch**. Its
dispatch route `gemv_hfq4g256_for_arch()` in `kernels.rs` has:

```rust
// "gfx1200" | "gfx1201" => ...,   // commented out — never activated
_ => (GEMV_HFQ4G256_SRC, "gemv_hfq4g256"), // fallback used
```

So gfx1201 currently runs the **base/generic kernel** which is identical
to the gfx1100 kernel: 4x group unroll + packed uint32 nibble loads.

The existing `gemv_hfq4g256.gfx1201.hip` has three known defects:
1. **Byte loads** (`nib[boff+0]`, `nib[boff+1]`, ...) instead of packed
   uint32 loads — 4 separate load instructions instead of 1 per group
2. **2x unroll only** instead of 4x (gfx1100) or better
3. **`s_prefetch_data`** — proven harmful on gfx1201 per Phase 10 result

### Why the base/gfx1100 kernel approach is correct

Both `gemv_hfq4g256.hip` and `gemv_hfq4g256.gfx1100.hip` use:
- `*(const unsigned int*)(gp + 8 + boff)` — packed uint32 load (4 nibbles
  in one 32-bit read, fully coalesced across the 32-thread wave)
- 4x group unroll with 4 independent accumulators
- Correct tail distribution: tail group `g` accumulates into `acc[g % 4]`
  (the `5302926` bug-fix invariant)

This is the correctness and performance baseline for gfx1201.

---

## Optimization Strategy

**Hypothesis:** Increasing the unroll factor from 4x to 8x will improve
instruction-level parallelism and memory-request pipelining for gfx1201,
which has:
- 32 CUs × 4 SIMDs, wave32 ISA
- GDDR6 288 GB/s peak bandwidth (Navi 44 / gfx1201)
- 32 KB L1 cache per CU

For Qwen3.5-9B decode (the bench target):
- groups_per_row ≈ 14 (small matrices: K=3584) — 1 oct + 6 tail
- groups_per_row ≈ 74 (large matrices: K=18944) — 9 octs + 2 tail

With 8x unroll, 8 independent memory access streams are in flight per
iteration. This better hides L2 latency (estimated 100-300 cycles on
gfx1201) compared to 4x.

**VGPR budget analysis** (`__launch_bounds__(32, 16)` → 96 VGPRs at 16
waves/SIMD on gfx1201):
- 8 accumulators: 8 VGPRs
- 8 gp pointers (64-bit): 16 VGPRs
- 16 scale/zero floats: 16 VGPRs
- 8 packed nibble words: 8 VGPRs
- tid, row, boff, misc: ~10 VGPRs
- **Total: ~58 VGPRs** — well within 96 VGPR budget, no spill risk

**Tail handling:** Groups `groups_per_row % 8` remainder are distributed
into `acc[g % 8]` — same correctness invariant as the gfx1100 4x kernel
but generalized to 8 lanes.

**Lessons applied from Phase 10:** No `s_prefetch_data`. The hardware
prefetcher on gfx1201 handles sequential 136-byte group strides
efficiently without software hints. Adding explicit prefetch for 4-quad
rows added overhead; for 8-group rows the same principle applies.

---

## Files to Change

| File | Change |
|------|--------|
| `kernels/src/gemv_hfq4g256.gfx1201.hip` | Replace with 8x unroll + uint32 loads |
| `crates/rdna-compute/src/kernels.rs` | Add `GEMV_HFQ4G256_GFX1201_SRC` const, wire dispatch |

---

## Acceptance Threshold

Phase 12 baseline median: **93.1 tok/s**  
Phase 11 canonical baseline: 93.2 tok/s  
Speed-gate floor (gfx1201.txt × 0.95): 96.4 tok/s (measured differently)

For Phase 12 to be recorded as ACCEPT:
- Median decode ≥ **95.0 tok/s** (>2 tok/s above baseline, clearly above noise)
- No coherence failures
- speed-gate PASS (no regression)

For Phase 12 REVERT:
- Median decode ≤ baseline + noise floor (~93.2 ± 0.1)
- OR any correctness gate failure

---

*Next in queue if REVERT: `gemv_hfq6g256` (rank 2, 18% decode, Amdahl 1.043x)*
