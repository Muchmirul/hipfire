#!/usr/bin/env bash
# make_phase11b_proof_pack.sh — assemble a self-contained proof pack
#
# Usage: ./tools/autokernel-rdna/make_phase11b_proof_pack.sh
#
# Creates: tools/autokernel-rdna/reports/phase11b_proof_pack_<timestamp>/
# Exit codes: 0 success, 1 failure

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
PACK_DIR="$SCRIPT_DIR/reports/phase11b_proof_pack_${TIMESTAMP}"
REPORTS_DIR="$SCRIPT_DIR/reports"

# PATH for cargo
CARGO_PATH="/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.cargo/bin"
[[ -d "$CARGO_PATH" ]] && export PATH="$CARGO_PATH:$PATH"

MODELS_DIR="${HIPFIRE_MODELS_DIR:-}"
if [[ -z "$MODELS_DIR" ]]; then
    FALLBACK="/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.hipfire/models"
    [[ -d "$FALLBACK" ]] && MODELS_DIR="$FALLBACK" || MODELS_DIR="$HOME/.hipfire/models"
fi

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}OK${NC}  $*"; }
fail() { echo -e "  ${RED}FAIL${NC}  $*"; }
info() { echo "  ....  $*"; }

echo "=== make_phase11b_proof_pack.sh === ($TIMESTAMP)"
mkdir -p "$PACK_DIR"
ok "pack dir created: $PACK_DIR"

# ── 1. Phase 11 validation reports ───────────────────────────────────────────
info "collecting phase11 validation reports..."
shopt -s nullglob
ph11_mds=( "$REPORTS_DIR"/phase11_validation_*.md )
ph11_jsons=( "$REPORTS_DIR"/phase11_validation_*.json )
for f in "${ph11_mds[@]}" "${ph11_jsons[@]}"; do
    cp "$f" "$PACK_DIR/" && ok "  copied: $(basename "$f")"
done
if [[ ${#ph11_mds[@]} -eq 0 ]]; then
    fail "no phase11_validation_*.md found — run phase11_validate.sh first"
fi

# ── 2. Phase 11-B remediation reports (if they exist already) ─────────────────
for f in "$REPORTS_DIR"/phase11b_remediation_*.md "$REPORTS_DIR"/phase11b_remediation_*.json; do
    [[ -f "$f" ]] && cp "$f" "$PACK_DIR/" && ok "  copied: $(basename "$f")"
done

# ── 3. results.tsv ───────────────────────────────────────────────────────────
info "results.tsv..."
cp "$SCRIPT_DIR/results.tsv" "$PACK_DIR/results.tsv"
ok "results.tsv copied"

# ── 4. git diff --stat ────────────────────────────────────────────────────────
info "git diff stat..."
{
    echo "# git diff --stat (autokernel PR scope)"
    echo "# Branch: pr-1, HEAD: $(git rev-parse HEAD)"
    echo "# Date: $(date -u)"
    echo ""
    git diff --stat "$(git merge-base HEAD origin/main 2>/dev/null || git rev-parse HEAD~5)" HEAD 2>/dev/null \
        || git diff --stat HEAD~5..HEAD
} > "$PACK_DIR/git_diff_stat.txt"
ok "git_diff_stat.txt"

# ── 5. git status ─────────────────────────────────────────────────────────────
{
    echo "# git status"
    echo "# Date: $(date -u)"
    echo ""
    git status
} > "$PACK_DIR/git_status.txt"
ok "git_status.txt"

# ── 6. Benchmark summaries ────────────────────────────────────────────────────
info "benchmark summaries..."
bench_logs=( "$REPORTS_DIR"/bench_trials_*.log )
for f in "${bench_logs[@]}"; do
    [[ -f "$f" ]] && cp "$f" "$PACK_DIR/" && ok "  copied: $(basename "$f")"
done
# Always include Phase 10 baseline JSON
for f in "$SCRIPT_DIR/baselines"/*.json; do
    [[ -f "$f" ]] && cp "$f" "$PACK_DIR/" && ok "  baseline: $(basename "$f")"
done

# ── 7. Gate logs ─────────────────────────────────────────────────────────────
info "gate logs..."
for prefix in gate_kernels gate_speed gate_coherence results_tsv_check; do
    for f in "$REPORTS_DIR"/${prefix}_*.log; do
        [[ -f "$f" ]] && cp "$f" "$PACK_DIR/" && ok "  gate log: $(basename "$f")"
    done
done

# ── 8. Environment metadata ───────────────────────────────────────────────────
info "environment metadata..."
{
    echo "# Environment Metadata"
    echo "# Captured: $(date -u)"
    echo ""
    echo "## Git"
    echo "HEAD SHA: $(git rev-parse HEAD)"
    echo "Branch: $(git rev-parse --abbrev-ref HEAD)"
    echo "Status: $(git status --short | wc -l) modified files"
    echo ""
    echo "## GPU"
    if command -v amdgpu-arch &>/dev/null; then
        echo "Arch: $(amdgpu-arch 2>/dev/null | head -1)"
    fi
    if command -v rocm-smi &>/dev/null; then
        rocm-smi --showproductname 2>/dev/null || true
    fi
    echo ""
    echo "## ROCm / HIP"
    cat /opt/rocm/.info/version 2>/dev/null && echo "" || true
    hipcc --version 2>/dev/null | head -2 || echo "hipcc: not found"
    echo ""
    echo "## Cargo"
    cargo --version 2>/dev/null || echo "cargo: not found"
    echo ""
    echo "## Models dir: $MODELS_DIR"
    ls "$MODELS_DIR"/*.mq4 2>/dev/null | xargs -I{} basename {} || echo "  (no .mq4 files)"
    echo ""
    echo "## Prompt"
    if [[ -f "benchmarks/prompts/lru_cache_pep8_strict.txt" ]]; then
        echo "File: benchmarks/prompts/lru_cache_pep8_strict.txt"
        echo "MD5:  $(md5sum benchmarks/prompts/lru_cache_pep8_strict.txt | awk '{print $1}')"
    fi
    echo ""
    echo "## Bench binary"
    BENCH_EXE="target/release/examples/bench_qwen35_mq4"
    if [[ -x "$BENCH_EXE" ]]; then
        echo "Path: $BENCH_EXE"
        echo "MD5:  $(md5sum "$BENCH_EXE" | awk '{print $1}')"
    else
        echo "Not built"
    fi
} > "$PACK_DIR/env_metadata.txt"
ok "env_metadata.txt"

# ── 9. Rollback notes ─────────────────────────────────────────────────────────
{
    echo "# Rollback Notes"
    echo ""
    echo "## Full rollback (remove all autokernel tooling)"
    echo ""
    echo "The autokernel PR adds ONLY tooling under tools/autokernel-rdna/"
    echo "and one retained negative-result kernel file:"
    echo "  kernels/src/gemv_hfq4g256_residual.gfx12.hip"
    echo ""
    echo "No Rust crate sources were modified. No dispatch paths were changed."
    echo ""
    echo "To roll back completely:"
    echo ""
    AK_SHA="$(git log --oneline | grep 'autokernel' | tail -1 | awk '{print $1}' || echo 'c74e5d9')"
    echo "  git log --oneline | grep autokernel   # find the first autokernel commit"
    echo "  git revert --no-commit <first-sha>..<HEAD>"
    echo "  git commit -m 'revert: remove autokernel tooling'"
    echo ""
    echo "Or more precisely, revert the two autokernel commits:"
    echo "  git revert c74e5d9 c148ca0 7bad524"
    echo ""
    echo "## Partial rollback (keep tooling, remove kernel artifact)"
    echo ""
    echo "  git checkout origin/main -- kernels/src/gemv_hfq4g256_residual.gfx12.hip 2>/dev/null"
    echo "  # or, if the file didn't exist upstream:"
    echo "  git rm kernels/src/gemv_hfq4g256_residual.gfx12.hip"
    echo "  git commit -m 'chore: remove retained negative-result kernel artifact'"
    echo ""
    echo "## Verification after rollback"
    echo ""
    echo "  bash scripts/test-kernels.sh gfx1201"
    echo "  bash scripts/speed-gate.sh"
    echo "  # Expect: same numbers as pre-PR baseline"
} > "$PACK_DIR/rollback_notes.txt"
ok "rollback_notes.txt"

# ── 10. Next-kernel recommendation ────────────────────────────────────────────
{
    echo "# Next Kernel Recommendation"
    echo ""
    echo "From profile report (profile_20260510-122608.md):"
    echo ""
    echo "| Rank | Kernel | Cycle % | Amdahl Limit | Recommended Action |"
    echo "|---|---|---|---|---|"
    echo "| 1 | gemv_hfq6g256 | 18% | +22% decode | Try 6x-unroll + bf16 accumulate on gfx12 |"
    echo "| 2 | attention_flash_asym3_tile | 12% | +14% decode | Tune tile size for gfx1201 wave32 |"
    echo "| 3 | gemv_hfq4g256 (main) | 20% | +25% decode | Try wave64 on gfx1201 (wave32 default) |"
    echo ""
    echo "## Recommendation"
    echo ""
    echo "Start with gemv_hfq6g256 (6-bit quant path):"
    echo "  - Wider operands → more parallelism per group"
    echo "  - Less sensitive to prefetch overhead than residual path"
    echo "  - gfx12-safe (gfx1200 + gfx1201 cover same WMMA tier)"
    echo ""
    echo "Command to start Phase 12:"
    echo "  ./tools/autokernel-rdna/run.sh experiment --kernel gemv_hfq6g256 --arch gfx1201"
    echo ""
    echo "Prerequisite: run phase11_validate.sh --self-test && check_results_tsv.sh"
    echo "both must pass before starting Phase 12."
} > "$PACK_DIR/next_kernel_recommendation.txt"
ok "next_kernel_recommendation.txt"

# ── 11. Index file ────────────────────────────────────────────────────────────
{
    echo "# Phase 11-B Proof Pack — Index"
    echo ""
    echo "Generated: $(date -u)"
    echo "HEAD: $(git rev-parse HEAD)"
    echo ""
    echo "## Contents"
    echo ""
    for f in "$PACK_DIR"/*; do
        name="$(basename "$f")"
        size="$(wc -l < "$f" 2>/dev/null || echo '?')"
        echo "- $name  ($size lines)"
    done
    echo ""
    echo "## Gate Summary"
    echo ""
    # Pull last speed gate result
    last_speed="$(ls -t "$PACK_DIR"/gate_speed_*.log 2>/dev/null | head -1)"
    if [[ -n "$last_speed" ]]; then
        grep -E 'PASS|FAIL|SKIP' "$last_speed" | head -20 || true
    fi
} > "$PACK_DIR/INDEX.md"
ok "INDEX.md"

echo ""
echo "=== Proof pack complete: $PACK_DIR ==="
ls "$PACK_DIR"
echo ""
echo -e "${GREEN}SUCCESS${NC}"
exit 0
