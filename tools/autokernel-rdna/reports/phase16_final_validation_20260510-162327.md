# Phase 16 — Final End-to-End Validation Report

**Decision: PASS_INFRA_ONLY**

Generated: 2026-05-10T16:23:27Z

---

## Environment

| Field | Value |
|---|---|
| Date | `2026-05-10T16:23:27Z` |
| Host | `dev-MS-7C56` |
| Git SHA | `93dfd15d339cb593a56a5f3204e5d9dd8cbe646a` |
| Dirty files | 37 |
| GPU | 		0x7550 |
| Arch (detected) | unknown |
| Arch (requested) | gfx1201 |
| ROCm | 7.2.1 |
| Kernel | 6.17.0-23-generic |
| Model | qwen3.5:27b |
| Trials | 1 |

## Pipeline Component Checklist

| Component | Status |
|---|---|
| autokernel_loop.sh | ✓ present |
| promote_kernel.sh | ✓ present |
| phase11_validate.sh | ✓ present |
| check_results_tsv.sh | ✓ present |
| kernel_lab/ | ✓ present |
| results.tsv | ✓ present |
| workspace/ | ✓ present |
| reports/ | ✓ present |

## results.tsv Schema

Status: **WARN**

## Candidate and Promotion Records

| | Count |
|---|---|
| Generated candidates | 7 |
| Harness-passing | 0 |
| Promoted (total attempts) | 3 |
| Accepted promotions | 0 |
| Rejected promotions | 3 |
| Latest accepted | none |

## hipfire Build

**PASS**

Build log: `/home/dev/hipfire-pr1-review/tools/autokernel-rdna/reports/phase16_build_20260510-162327.log`

## Correctness Gates

| Gate | Status |
|---|---|
| speed-gate.sh --fast | PASS |
| coherence-gate-dflash.sh | SKIPPED |

## Qwen3.5-27B Benchmark

| | Value |
|---|---|
| Status | PASS |
| Model path | `/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.hipfire/models/qwen3.5-27b.mq4` |
| Prompt MD5 | `df5dedc8040ce70ba55080c4548e6024` |
| Trials | 1 |
| Median tok/s | **35.0** |
| Min tok/s | 35.0 |
| Max tok/s | 35.0 |

## Baseline Comparison

| | Value |
|---|---|
| Baseline status | NEEDS_BASELINE |
| Baseline source | none |
| Baseline tok/s | 0 |
| Final tok/s | 35.0 |
| Delta | 0 tok/s |
| Speedup | 0× |
| Percent | 0% |
| Speedup status | UNKNOWN |

> **No 27B baseline found.** To generate one:
> ```bash
> ARCH=gfx1201 MODEL=qwen3.5:27b ./tools/autokernel-rdna/run.sh baseline
> ```

## Final Decision: PASS_INFRA_ONLY

All infrastructure checks passed. No promoted kernel has improved tok/s yet — pipeline is ready for more candidates.

## Known Limitations

- No Qwen3.5-27B baseline exists in `tools/autokernel-rdna/baselines/` (all stored baselines are for 9b).
- The bench binary writes all output to stderr; this is captured correctly.
- DFlash draft is not required for this benchmark (spec-decode is a separate evaluation).
- DDTree gfx1201 regression is expected and documented (see AGENTS.md §4).

## Rollback Commands

No accepted promotions to roll back.

## Next Recommended Action

Generate a 27B baseline: `ARCH=gfx1201 MODEL=qwen3.5:27b ./tools/autokernel-rdna/run.sh baseline`. Then run `AUTHOR_KERNEL=1 ./tools/autokernel-rdna/autokernel_loop.sh` overnight and promote passing candidates.
