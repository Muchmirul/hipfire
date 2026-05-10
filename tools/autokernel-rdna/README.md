# autokernel-rdna

AutoKernel-style autonomous kernel optimization for hipfire on AMD Radeon RX 9070 XT (RDNA4 / `gfx1201`).

Adapts the [AutoKernel](https://github.com/RightNow-AI/autokernel) loop — *profile → rank → edit → bench → keep/revert → repeat* — to hipfire's Rust + HIP architecture. No Python in the hot path; all tooling is Bash + existing hipfire bench/gate infrastructure.

## Quick Start

```bash
# From the repo root:

# 1. Capture baseline (requires gfx1201 GPU + hipfire bench binary)
./tools/autokernel-rdna/run.sh baseline --arch gfx1201 --model qwen3.5:27b

# 2. Profile and rank bottleneck kernels
./tools/autokernel-rdna/run.sh profile --arch gfx1201 --model qwen3.5:27b

# 3. Optimize a specific kernel (interactive — you edit the .hip file)
./tools/autokernel-rdna/run.sh experiment --kernel gemv_hfq4g256 --arch gfx1201

# 4. Full Amdahl-driven orchestration loop
./tools/autokernel-rdna/run.sh orchestrate --arch gfx1201 --model qwen3.5:27b

# 5. Autonomous Phase 13 optimization loop (unattended)
ARCH=gfx1201 MODEL=qwen3.5:27b MAX_ITERS=100 ./tools/autokernel-rdna/autokernel_loop.sh
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

## Phase 13: Autonomous Optimization Loop

`autokernel_loop.sh` is the Phase 13 main deliverable — a fully unattended autonomous loop that:

1. Reads `workspace/orchestration_state.json` to know the current target kernel and which strategies have already been tried
2. Picks the next untried strategy from the optimization plan for that target
3. Backs up the kernel source file and applies the strategy mutation via `sed`
4. Rebuilds the bench binary (`cargo build --release`)
5. Runs the correctness gate (`scripts/test-kernels.sh`)
6. Benchmarks across `BENCH_TRIALS` fresh-process runs
7. ACCEPTs if: median speedup ≥ `ACCEPT_MIN_SPEEDUP` AND correctness passes
8. REJECTs (reverts) otherwise
9. Appends a row to `results.tsv` with all 39 fields including Phase 13 loop columns
10. Saves candidate diff + build log to `workspace/candidates/iter-NNN/`
11. Saves accepted diffs to `workspace/accepted/`, rejected to `workspace/rejected/`
12. Updates `orchestration_state.json` for resume across crashes

### How to run

```bash
# Short run (100 iterations — explore 1–2 kernels)
ARCH=gfx1201 MODEL=qwen3.5:27b MAX_ITERS=100 ./tools/autokernel-rdna/autokernel_loop.sh

# Overnight run (300 iterations, lower acceptance threshold, auto-commit)
ARCH=gfx1201 MODEL=qwen3.5:27b MAX_ITERS=300 ACCEPT_MIN_SPEEDUP=1.005 AUTOCOMMIT=1 \
  ./tools/autokernel-rdna/autokernel_loop.sh

# Smoke test with 9B model (faster builds, but less representative)
ARCH=gfx1201 MODEL=qwen3.5:9b MAX_ITERS=10 BENCH_TRIALS=2 \
  ./tools/autokernel-rdna/autokernel_loop.sh
```

### How to stop safely

```bash
# Touch the stop flag — loop will finish the current iteration cleanly and write the final report
touch .autokernel_stop

# Or press Ctrl-C — the SIGINT handler writes the final report before exiting
```

### Where results are saved

| Path | Contents |
|------|----------|
| `tools/autokernel-rdna/results.tsv` | All iterations appended (48 cols: 32 core + 7 Phase 13 + 9 Phase 14) |
| `tools/autokernel-rdna/workspace/orchestration_state.json` | Current loop state (resume on restart) |
| `tools/autokernel-rdna/workspace/candidates/iter-NNN/` | diff + build log for each iteration |
| `tools/autokernel-rdna/workspace/accepted/` | Diffs of all accepted candidates |
| `tools/autokernel-rdna/workspace/rejected/` | Diffs of all rejected candidates |
| `tools/autokernel-rdna/reports/phase13_autokernel_loop_<ts>.{md,json}` | Final summary report |

### Known limitations

- **Qwen3.5-27B requires ~24 GB VRAM** — `hipMalloc` OOM at >16K ctx on 24 GB cards. Use default `--ctx 2048`.
- **Each build takes 5–10 min** (full `cargo build --release`). 100 iterations ≈ 8–17 hours wall-clock.
- **Strategy library is conservative**: launch bounds, `#pragma unroll`, `__restrict__` qualifiers. Complex mutations (LDS tiling, vector-width changes) require manual kernel editing — use `run.sh experiment` for those.
- **Baseline is cached**: delete `tools/autokernel-rdna/baselines/` to force a fresh baseline measurement before the loop.
- **State persists**: `orchestration_state.json` survives crashes. To reset and restart from scratch: `rm tools/autokernel-rdna/workspace/orchestration_state.json`.
- **DFlash disabled by default**: the bench uses `--ar-baseline`. DFlash benches need `hipfire config set dflash_mode auto`.
- **AUTOCOMMIT=1**: auto-commits accepted candidates to the current branch. Review commits before pushing.

---

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

**Phase 13 loop columns** (present only in rows written by `autokernel_loop.sh`):

| Column | Description |
|--------|-------------|
| `loop_iteration` | Iteration number within the loop run |
| `loop_target` | Kernel name being optimized this iteration |
| `loop_strategy` | Strategy name applied (e.g. `launch_w32_o12`) |
| `loop_files_changed` | Semicolon-separated list of modified `.hip` files |
| `benchmark_status` | `ok` / `timeout` / `crash` / `build_fail` / `correctness_fail` |
| `current_best_tok_s` | Best tok/s seen so far in this loop run |
| `speedup_vs_current_best` | `candidate_decode_tok_s / current_best_tok_s` |

**Phase 14 author-kernel columns** (present only in rows written with `AUTHOR_KERNEL=1`):

| Column | Description |
|--------|-------------|
| `author_kernel_mode` | `1` when this row was written by Phase 14 |
| `target_spec_path` | Path to `kernel_lab/generated/<target>/target_spec.json` |
| `candidate_kernel_path` | Path to the generated candidate `.hip` file |
| `harness_correctness_status` | Result from `kernel_lab/harnesses/<target>_harness.sh` (PASS/FAIL) |
| `harness_speedup` | Standalone harness speedup vs reference kernel |
| `promoted_to_hipfire` | `1` if candidate was copied to `kernels/src/` and rebuilt |
| `hipfire_tok_s_before` | End-to-end tok/s before promotion (baseline) |
| `hipfire_tok_s_after` | End-to-end tok/s after promotion (0 if not promoted) |
| `final_decision` | ACCEPT / REJECT / SKIP (from `author_kernel_iteration()`) |

---

## Phase 14 — Real Kernel-Authoring Mode

Phase 14 extends the loop to generate genuinely new HIP kernel code (not just text
mutations), test it in isolated harnesses, and promote winners into hipfire.

### How to run

```bash
# Author new kernel candidates for the top-3 decode targets (50 iterations):
ARCH=gfx1201 MODEL=qwen3.5:27b AUTHOR_KERNEL=1 MAX_ITERS=50 ACCEPT_MIN_SPEEDUP=1.005 \
  ./tools/autokernel-rdna/autokernel_loop.sh
```

### Architecture

```
author_kernel.sh              # Orchestrates one authoring cycle
  ↓
extract_target_spec()         # Reads kernels/src/<target>.hip, emits target_spec.json
  ↓
generate_candidate_skeleton() # Copies reference to kernel_lab/generated/<target>/candidate_N.hip
                              # with authoring header comment (strategy + invariants)
  ↓
  [LLM agent edits candidate_N.hip per the strategy instructions]
  ↓
run_harness()                 # Fixed harness: compile candidate → correctness check → timing
  ↓                           # Script: kernel_lab/harnesses/<target>_harness.sh
  if PASS && speedup >= threshold
  ↓
promote_to_hipfire()          # cp candidate_N.hip → kernels/src/<target>.gfx1201.hip
  ↓                           # cargo build --release
  ↓
E2E bench (3 trials)          # Final accept/revert decision
```

### Key directories

| Path | Contents |
|------|----------|
| `kernel_lab/templates/hip_kernel_candidate.md` | Agent instruction template (strategy, invariants, gfx1201 tips) |
| `kernel_lab/generated/<target>/target_spec.json` | Per-target spec (shapes, tolerance, dispatch path) |
| `kernel_lab/generated/<target>/candidate_N.hip` | Generated candidates |
| `kernel_lab/harnesses/<target>_harness.sh` | Fixed evaluator (NEVER modified during run) |
| `kernel_lab/reports/` | Phase 14 summary reports |

### Harness outputs

Each harness prints to stdout:

```
CORRECTNESS: PASS|FAIL
LATENCY_US: <float>        # candidate latency
REF_LATENCY_US: <float>    # reference latency (same test shape)
SPEEDUP_VS_REF: <float>    # ref / cand (higher = faster candidate)
ERROR: <msg>               # only on FAIL
```

### Hard rules

1. **Correctness is a hard gate.** Harness FAIL → no promotion, no e2e bench.
2. **Harness is fixed.** Never modify `harnesses/*.sh` during a run.
3. **Fallback kernel is never deleted.** `kernels/src/<target>.hip` always exists as oracle.
4. **4-accumulator invariant** in `gemv_hfq4g256` must be preserved. Tail group g uses `acc[g % N_ACCUMULATORS]`. Any candidate that changes this fails the harness.
5. **Revert on e2e rejection** — promoted kernel is `git checkout`'d back if e2e tok/s falls below threshold.

### Known limitations

- **Skeleton-only mode:** `generate_candidate_skeleton()` copies the reference kernel as-is. In fully automated runs (no LLM agent edits the candidate), the harness reports 1.000x and the loop rejects. This is intentional — Phase 14 is designed to evaluate *human or LLM-authored* kernel code, not to apply automatic mutations.
- **gemv_hfq6g256 harness CPU reference assumes 4-weights-per-3-bytes packing.** Verify against `kernels/src/gemv_hfq6g256.hip` before trusting PASS results.
- **Build time:** promotion triggers `cargo build --release` (5–10 min). Budget accordingly.
- **No MoE support:** MQ3 + MoE/A3B is refused at load time (inherited from AGENTS.md). Authoring targets are dense-model GEMV only.
