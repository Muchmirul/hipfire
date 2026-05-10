# AutoKernel Loop Report — Phase 13

Generated: 2026-05-10T17:19:43Z

## Run Summary

| Field | Value |
|-------|-------|
| Model | `qwen3.5:27b` |
| Arch | gfx1201 |
| Total iterations | 1 |
| Baseline tok/s | 93.1 |
| Best tok/s | 35.1 |
| Best speedup | 0.9972x |
| Accepted candidates | 0 |
| Rejected/reverted | 0 |
| Crashes | 0 |
| Timeouts | 0 |
| Targets advanced (plateau) | 1 |

## Accepted Candidates

_(none)_

## Rejected / Exhausted Candidates

_(none)_

## Next Recommended Target

`gemv_hfq6g256`

## How to Rerun

```bash
ARCH=gfx1201 MODEL=qwen3.5:27b MAX_ITERS=1 ACCEPT_MIN_SPEEDUP=1.005 \
  ./tools/autokernel-rdna/autokernel_loop.sh
```

## How to Rollback Accepted Changes

```bash
# Show accepted commits:
git log --oneline --grep='autokernel: optimize'
# Revert one:
git revert <sha>
# Or revert all kernel source files to a known-good SHA:
git checkout <base-sha> -- kernels/src/
```

## Final Validation

No accepted candidates — skip validation.

```bash
TRIALS=5 MODEL=qwen3.5:27b ./tools/autokernel-rdna/phase11_validate.sh
```

## Full Results Log

`tools/autokernel-rdna/results.tsv`

