# AutoKernel Full Run — Summary

| Field | Value |
|---|---|
| Timestamp | 20260510-171732 |
| ARCH | gfx1201 |
| MODEL | qwen3.5:27b |
| MAX_ITERS | 1 |
| ACCEPT_MIN_SPEEDUP | 1.005 |
| TRIALS | 5 |
| AUTOCOMMIT | 0 |
| Loop ran | 1 iterations |
| Candidate promoted | /home/dev/hipfire-pr1-review/tools/autokernel-rdna/kernel_lab/generated/gemv_hfq6g256/candidate_1.hip |
| Initial decision | PASS_INFRA_ONLY |
| **Final decision** | **PASS_INFRA_ONLY** |
| Full log | `/home/dev/hipfire-pr1-review/tools/autokernel-rdna/reports/full_autokernel_run_20260510-171732/full_run.log` |
| Rollback report | `/home/dev/hipfire-pr1-review/tools/autokernel-rdna/reports/phase16_rollback_20260510-171946.md` |

## Decision meanings

| Decision | Meaning |
|---|---|
| `PASS_ADOPTED` | ≥1 kernel promoted + verified tok/s improvement + all gates pass. Done. |
| `PASS_INFRA_ONLY` | All gates pass. No kernel improved tok/s yet — run more iterations. |
| `NEEDS_BASELINE` | Validation works but no 27B baseline exists for speedup comparison. |
| `FAIL` | Build or correctness gate failed. Fix before claiming adoption. |

## Rollback

```bash
# See rollback instructions:
cat /home/dev/hipfire-pr1-review/tools/autokernel-rdna/reports/phase16_rollback_20260510-171946.md
```

## Next step

Run more iterations or generate a 27B baseline: `ARCH=gfx1201 MODEL=qwen3.5:27b ./tools/autokernel-rdna/run.sh baseline`
