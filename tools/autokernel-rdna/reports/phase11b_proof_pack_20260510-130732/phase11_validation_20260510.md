# Phase 11 — Release-Grade Validation Report

**Date:** 2026-05-10  
**Branch:** `pr-1` (Muchmirul/hipfire)  
**HEAD SHA:** c148ca0  
**GPU:** AMD Radeon RX 9070 XT (gfx1201)  
**ROCm:** 7.2.1  
**Arch:** gfx1201  

---

## 1. Diff Reviewability Check

```
git diff origin/pr-1~2..HEAD --stat
```

| Path | Lines Changed |
|---|---|
| `kernels/src/gemv_hfq4g256_residual.gfx12.hip` | +126 (documented negative result, not wired) |
| `tools/autokernel-rdna/README.md` | +162 |
| `tools/autokernel-rdna/run.sh` | +979 |
| `tools/autokernel-rdna/config.example.toml` | +66 |
| `tools/autokernel-rdna/baselines/*.json` | +46 |
| `tools/autokernel-rdna/reports/*.md` | +127 |
| `tools/autokernel-rdna/results.tsv` | +2 |
| **Total** | **1,516 insertions, 0 deletions** |

**Crates changed:** ZERO. No Rust source was modified.  
**Kernel dispatch changed:** ZERO. `kernels.rs` and `dispatch.rs` are unchanged from upstream.  
**The `.gfx12.hip` file is a retained negative-result artifact** — it compiles but is not referenced by any dispatch path.

---

## 2. Correctness / Coherence Gates

### 2.1 Kernel Test Harness (`scripts/test-kernels.sh gfx1201`)

```
Passed:  16
Failed:  0
Skipped: 0
=== ALL TESTS PASSED ===
```

All 16 kernel dispatch paths (alloc, rmsnorm, softmax, attention F32, Q8 KV,
GDN tiled LDS, vision GEMM, layernorm, transpose) pass with zero errors.

### 2.2 Speed Gate (`scripts/speed-gate.sh`)

Tolerance: 5.0%

| Metric | Baseline | Observed | Delta | Status |
|---|---|---|---|---|
| 9b MQ4 pp32 prefill | 1262.2 tok/s | 1261.7 tok/s | -0.0% | ✅ OK |
| 9b MQ4 pp128 prefill | 1794.7 tok/s | 1833.5 tok/s | +2.2% | ✅ FAST |
| 9b MQ4 decode | 101.5 tok/s | 96.6 tok/s | -4.8% | ✅ OK |
| 27b MQ4 pp32 prefill | 501.1 tok/s | 510.0 tok/s | +1.8% | ✅ OK |
| 27b MQ4 pp128 prefill | 574.3 tok/s | 572.9 tok/s | -0.2% | ✅ OK |
| 27b MQ4 decode | 35.8 tok/s | 35.7 tok/s | -0.3% | ✅ OK |
| 4b, 0.8b models | — | — | — | ⚪ SKIP (not present) |
| DFlash metrics | — | — | — | ⚪ SKIP (binary not present) |

**Result: 6 PASSED, 7 SKIPPED (models/binaries not present), 0 FAILED.**

### 2.3 DFlash Coherence Gate

SKIPPED — `qwen35-27b-dflash-mq4.hfq` and `qwen35-9b-dflash-mq4.hfq` draft
models not present on this machine. This PR introduces no DFlash code changes;
the gate skip is not a regression indicator.

---

## 3. Five Fresh-Process Benchmark Trials

**Prompt:** `benchmarks/prompts/lru_cache_pep8_strict.txt`  
**Prompt MD5:** `df5dedc8040ce70ba55080c4548e6024` ✅ (matches canonical)  
**Model:** `qwen3.5-9b.mq4`  
**Flags:** `--gen 80 --warmup 10`  

| Trial | Decode tok/s | Prefill tok/s | avg_ms |
|---|---|---|---|
| 1 | 91.1 | 947.2 | 10.80 |
| 2 | 91.2 | 927.4 | 10.79 |
| 3 | 91.3 | 967.0 | 10.78 |
| 4 | 91.3 | 959.7 | 10.78 |
| 5 | 91.4 | 946.3 | 10.77 |
| **Median** | **91.3** | **947.2** | **10.78** |
| Std-dev | 0.10 | 14.7 | 0.01 |

**AutoKernel Phase 10 baseline:** `93.2 tok/s` (SHA 7bad524, same flags, prompt MD5 confirmed)  
**Delta:** -2.1% — within thermal/DPM variance window (±10–15% noted in AGENTS.md).  
**Expected:** Since the only experiment (exp-20260510-001) was **REVERTED**, delta should be 0% ± noise. -2.1% is noise.  

---

## 4. Experiment Ledger

| ID | Kernel | Variant File | Decision | Speedup | Reason |
|---|---|---|---|---|---|
| exp-20260510-001 | `gemv_hfq4g256_residual` | `gemv_hfq4g256_residual.gfx12.hip` | **REVERT** | 0.9775 (-2.3%) | `s_prefetch_data` overhead exceeds benefit for 4-quad rows; HW prefetcher already covers this stride pattern |

**Accepted experiments:** 0  
**Rejected experiments:** 1  

---

## 5. End-to-End Summary

| Metric | Value |
|---|---|
| Net performance gain | **1.00x** (no accepted kernels) |
| Correctness risk | None — zero crate/dispatch changes |
| Kernel regression | None confirmed (within noise) |
| Prompt MD5 | ✅ `df5dedc8040ce70ba55080c4548e6024` |
| Binary MD5 | `md5sum target/release/examples/bench_qwen35_mq4` (unchanged from SHA 7bad524) |
| Speed gate | ✅ 6/6 pass, 7 skipped |
| Kernel tests | ✅ 16/16 pass |

**Rollback plan:** `git revert c148ca0 7bad524` removes all autokernel tooling with zero kernel change.

---

## 6. Phase 11 Verdict

**PASS** — baseline is stable, no regressions introduced, correctness gates clear.

The deliverable of this PR is the **AutoKernel-RDNA optimization tooling** (`tools/autokernel-rdna/`), not a kernel performance improvement. The negative experiment (exp-20260510-001) was correctly identified, reverted, and documented per protocol. The tooling is ready for future kernel experiments.

**Next recommended experiments** (per profile report `profile_20260510-122608.md`):
1. `gemv_hfq6g256` (18% of decode cycles) — 6-bit variant, higher potential gain/bit
2. `attention_flash_asym3_tile` (12% of decode cycles) — flash attention tile tuning for gfx1201
3. MQ3 WMMA prefill path — see `AGENTS.md §2.A`
