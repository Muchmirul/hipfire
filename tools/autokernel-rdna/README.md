# autokernel-rdna

AutoKernel-style autonomous kernel optimization for hipfire on AMD Radeon RX 9070 XT (RDNA4 / `gfx1201`).

Adapts the [AutoKernel](https://github.com/RightNow-AI/autokernel) loop — *profile → rank → edit → bench → keep/revert → repeat* — to hipfire's Rust + HIP architecture. No Python in the hot path; all tooling is Bash + existing hipfire bench/gate infrastructure.

## Quick Start

```bash
# From the repo root:

# 1. Capture baseline (requires gfx1201 GPU + hipfire bench binary)
./tools/autokernel-rdna/run.sh baseline --arch gfx1201 --model qwen3.5:9b

# 2. Profile and rank bottleneck kernels
./tools/autokernel-rdna/run.sh profile --arch gfx1201 --model qwen3.5:9b

# 3. Optimize a specific kernel (interactive — you edit the .hip file)
./tools/autokernel-rdna/run.sh experiment --kernel gemv_hfq4g256 --arch gfx1201

# 4. Full Amdahl-driven orchestration loop
./tools/autokernel-rdna/run.sh orchestrate --arch gfx1201 --model qwen3.5:9b
```

## Commands

### `baseline`

Captures environment metadata and runs the benchmark to establish the performance floor.

- Verifies ROCm/HIP availability and detects GPU arch
- Refuses to run if the active GPU is not `gfx1201` (pass `--allow-other-arch` to override)
- Records: git SHA, ROCm version, hipcc version, GPU name, kernel cache path, prompt MD5, all `HIPFIRE_*` env vars
- Rebuilds `bench_qwen35_mq4` from clean (removes stale binary per bench methodology)
- Runs `$TRIALS` fresh-process trials with DPM warmup (built into the bench binary)
- Saves to `tools/autokernel-rdna/baselines/<timestamp>.json`

### `profile`

Ranks candidate kernels by estimated end-to-end decode impact using Amdahl's law.

- Runs a single bench pass for timing signal
- Inventories all kernels in `kernels/src/` and classifies by type, bound type, and impact
- Notes which kernels already have `gfx12` or `gfx1201` variants
- Applies Amdahl's law: E2E gain = 1 / ((1 − f) + f/s) where f = decode fraction, s = expected speedup
- Saves profile to `tools/autokernel-rdna/reports/profile_<timestamp>.{md,json}`

Optional: if `rocprof` is available, run it against the bench binary and update the decode fraction estimates for higher accuracy.

### `experiment`

Runs a single kernel optimization experiment:

1. Creates a git branch `autokernel/gfx1201-<kernel>-<date>`
2. Creates or selects the correct kernel variant file:
   - Prefers `kernels/src/<kernel>.gfx12.hip` (covers gfx1200+gfx1201)
   - Falls back to `kernels/src/<kernel>.gfx1201.hip` for strictly chip-specific work
3. **You (or an agent) modify the kernel file**
4. Builds from clean state
5. Runs `./scripts/test-kernels.sh` (correctness)
6. Runs `./scripts/coherence-gate-dflash.sh --fast` (token distribution gate)
7. Runs `$TRIALS` fresh-process benchmark trials
8. Compares median vs baseline
9. **KEEPS** if: correctness passes + coherence passes + median speedup ≥ threshold
10. **REVERTS** otherwise — logs failure reason to `results.tsv`

For automated/agent use, place a git-format patch at `tools/autokernel-rdna/experiments/<kernel>.patch`; the script applies it automatically.

### `orchestrate`

Full Amdahl-driven loop:

- Reads the latest profile report
- Picks the highest-impact kernel not yet optimized (or with < 3 failed attempts)
- Runs `experiment` for each candidate in order
- Stops after 8 consecutive failures (anti-thrashing guard)
- Produces `tools/autokernel-rdna/reports/final_<timestamp>.md`

## Directory Layout

```
tools/autokernel-rdna/
  run.sh                 Main entrypoint script
  config.example.toml   Configuration reference (copy to config.toml to customize)
  results.tsv            All experiment attempts (tab-separated, append-only)
  baselines/             Baseline JSON snapshots (<timestamp>.json)
  reports/               Profile and final reports (profile_*.{md,json}, final_*.md)
  experiments/           Per-experiment artifacts and patch files
```

## Hard Rules

Per `docs/methodology/perf-benchmarking.md` and `AGENTS.md` — these are enforced by the script:

- **No speedup claims from a single noisy run.** Minimum 3 fresh-process trials.
- **Always fresh-process.** Stale bench binary is deleted before every rebuild.
- **Byte-identical prompts.** Prompt MD5 is recorded with every experiment.
- **Run coherence gate** after any kernel/dispatch/fusion change (`--no-coherence` is not recommended).
- **Revert on correctness failure**, even if tok/s improves.
- **No bypassing gates** with `--no-verify`.
- **No Python in the hot path.** This tooling is offline-only Bash.

## Kernel Variant Naming

```
kernels/src/<name>.hip          # Generic fallback (all archs)
kernels/src/<name>.gfx12.hip   # RDNA4-family (gfx1200 + gfx1201) — preferred
kernels/src/<name>.gfx1201.hip # RX 9070 XT chip-specific only
```

Prefer `.gfx12.hip` unless the optimization is not safe on `gfx1200`. The compile script (`scripts/compile-kernels.sh`) automatically picks the most-specific variant.

## Optimization Playbook (RDNA4 / gfx1201)

- **Wave32 native** — RDNA4 runs wave32; avoid wave64 constructs unless profiling shows a specific benefit.
- **WMMA available** — `__builtin_amdgcn_wmma_*` intrinsics work on gfx12; use for matrix ops in prefill.
- **LDS bank conflicts** — LDS bank width is 32 bytes; strides that are multiples of 32 bank-conflict. Use padding or odd-stride.
- **Vectorized loads** — Use `float4` (128-bit) or `float2` (64-bit) for coalesced global memory access.
- **Decode = memory-bound** — Minimize global reads; maximize weight reuse across warps.
- **Prefill = compute-bound** — Maximize WMMA/MFMA-equivalent utilization; use shared memory tiling.
- **Tune**: block size, unroll factor, VGPR pressure, LDS usage, occupancy.
- **Fuse** — Fuse dequant + matvec where profitable (already done in `gemv_hfq*` variants).
- **No extra launches** — Avoid splitting into extra kernel invocations in the decode hot path.

## Results Format

`results.tsv` columns (tab-separated):

| Column | Description |
|--------|-------------|
| `experiment_id` | Sequential ID |
| `timestamp` | ISO 8601 UTC |
| `git_base_sha` | SHA of base commit |
| `git_candidate_sha` | SHA after kernel edit |
| `arch` | Target arch (`gfx1201`) |
| `gpu_name` | GPU product name |
| `rocm_version` | ROCm version string |
| `model` | hipfire model tag |
| `quant` | Quantization mode |
| `kv_mode` | KV cache mode |
| `prompt_name` | Bench prompt filename |
| `prompt_md5` | MD5 of prompt file |
| `kernel_name` | Kernel being optimized |
| `kernel_variant_file` | Which `.hip` file was modified |
| `change_summary` | One-line description |
| `correctness_status` | PASS / FAIL / SKIP |
| `coherence_status` | PASS / FAIL / SKIP |
| `build_status` | PASS / FAIL |
| `baseline_decode_tok_s` | Baseline decode tok/s |
| `candidate_decode_tok_s` | Candidate decode tok/s |
| `decode_speedup` | Ratio |
| `baseline_prefill_tok_s` | Baseline prefill tok/s |
| `candidate_prefill_tok_s` | Candidate prefill tok/s |
| `prefill_speedup` | Ratio |
| `end_to_end_estimated_speedup` | Amdahl estimate |
| `measured_end_to_end_speedup` | Measured E2E |
| `median_trials` | Number of successful trials |
| `stddev` | Standard deviation of decode tok/s across trials |
| `vram_mb` | Peak VRAM |
| `decision` | KEEP / REVERT |
| `revert_reason` | Reason for revert (if applicable) |
| `notes` | Freeform notes |
