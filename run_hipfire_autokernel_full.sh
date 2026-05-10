#!/usr/bin/env bash
# run_hipfire_autokernel_full.sh
#
# Convenience wrapper — runs the complete AutoKernel adoption pipeline
# for hipfire in one command, from the repo root.
#
# Chains existing Phase 13-16 scripts:
#   1. final_validate.sh  (current state check)
#   2. autokernel_loop.sh  (AUTHOR_KERNEL search, if needed)
#   3. promote_kernel.sh   (promote best candidate)
#   4. final_validate.sh   (post-promotion proof)
#   5. summary.md          (written to reports/full_autokernel_run_<ts>/)
#
# Usage:
#   Short run:
#     ARCH=gfx1201 MODEL=qwen3.5:27b MAX_ITERS=20 AUTOCOMMIT=0 \
#       ./run_hipfire_autokernel_full.sh
#
#   Aggressive overnight run:
#     ARCH=gfx1201 MODEL=qwen3.5:27b MAX_ITERS=300 AUTOCOMMIT=1 \
#       ./run_hipfire_autokernel_full.sh
#
#   Manual candidate override (skip search, go straight to promote):
#     ARCH=gfx1201 MODEL=qwen3.5:27b \
#       CANDIDATE_DIR=tools/autokernel-rdna/kernel_lab/generated/gemv_hfq4g256/candidate_1 \
#       ./run_hipfire_autokernel_full.sh
#
# Exit codes:
#   0  PASS_ADOPTED
#   2  no promotable candidate found
#   3  pipeline ran but final decision is not PASS_ADOPTED
#   1  script/validation error
#
# Hard rules:
#   - Does not touch kernels directly — all changes go through promote_kernel.sh
#   - Does not bypass final_validate.sh
#   - Default model is qwen3.5:27b (never 9b)

set -uo pipefail

# ── Resolve repo root ─────────────────────────────────────────────────────────
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || {
    echo "[wrapper] ERROR: not inside a git repo. Run from hipfire root." >&2
    exit 1
}
cd "$REPO_ROOT"

# ── Config ────────────────────────────────────────────────────────────────────
ARCH="${ARCH:-gfx1201}"
MODEL="${MODEL:-qwen3.5:27b}"
MAX_ITERS="${MAX_ITERS:-300}"
ACCEPT_MIN_SPEEDUP="${ACCEPT_MIN_SPEEDUP:-1.005}"
TRIALS="${TRIALS:-5}"
AUTOCOMMIT="${AUTOCOMMIT:-1}"
RUN_FINAL_VALIDATE="${RUN_FINAL_VALIDATE:-0}"
CANDIDATE_DIR="${CANDIDATE_DIR:-}"        # optional: skip search, use this dir
FORCE_CONTINUE="${FORCE_CONTINUE:-0}"     # set 1 to continue past initial FAIL
SKIP_COHERENCE="${SKIP_COHERENCE:-0}"     # passed through to final_validate
SKIP_BENCH="${SKIP_BENCH:-0}"            # passed through to final_validate

TOOL_DIR="$REPO_ROOT/tools/autokernel-rdna"
REPORTS_DIR="$TOOL_DIR/reports"

TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
RUN_DIR="$REPORTS_DIR/full_autokernel_run_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

FULL_LOG="$RUN_DIR/full_run.log"
SUMMARY_MD="$RUN_DIR/summary.md"

# Tee everything to full_run.log
exec > >(tee -a "$FULL_LOG") 2>&1

# ── Colors ────────────────────────────────────────────────────────────────────
C_RESET='\033[0m'
C_GREEN='\033[0;32m'
C_RED='\033[0;31m'
C_YELLOW='\033[0;33m'
C_CYAN='\033[0;36m'
C_BOLD='\033[1m'

log()      { echo -e "${C_CYAN}[wrapper]${C_RESET} $*"; }
log_ok()   { echo -e "${C_GREEN}[wrapper]${C_RESET} $*"; }
log_err()  { echo -e "${C_RED}[wrapper]${C_RESET} $*"; }
log_warn() { echo -e "${C_YELLOW}[wrapper]${C_RESET} $*"; }

# ── State ─────────────────────────────────────────────────────────────────────
INITIAL_DECISION="UNKNOWN"
FINAL_DECISION="UNKNOWN"
PROMOTED_CANDIDATE=""
LOOP_RAN=0
PROMOTE_RAN=0

# ── Helpers ───────────────────────────────────────────────────────────────────

# Read the decision from the most-recent phase16_final_validation_*.md
latest_decision() {
    local latest
    latest=$(ls -t "$REPORTS_DIR"/phase16_final_validation_*.md 2>/dev/null | head -1)
    if [[ -z "$latest" ]]; then
        echo "UNKNOWN"
        return
    fi
    grep -oP '\*\*Decision: \K[A-Z_]+' "$latest" 2>/dev/null | head -1 || echo "UNKNOWN"
}

# Return all promotable candidate .hip files, sorted by harness speedup descending.
# Skips candidates already present in accepted_promotions/ (already wired into hipfire).
# Output: one absolute path per line.
find_all_candidate_hips_sorted() {
    local already_promoted_dir="$TOOL_DIR/workspace/accepted_promotions"

    # Collect all passing harness candidates with their speedup scores.
    # Format: "<speedup_float> <hip_file_path>"
    local -a scored=()

    # Source 1: workspace/accepted/ — passed loop's quick benchmark, not yet in hipfire
    while IFS= read -r hip; do
        [[ -f "$hip" ]] || continue
        local base; base="$(basename "$hip")"
        # Skip if already promoted (same filename in accepted_promotions/)
        if ls "$already_promoted_dir"/"${base%.*}"_*.hip "$already_promoted_dir/$base" \
               2>/dev/null | grep -q .; then
            continue
        fi
        scored+=("1.0000 $hip")  # no harness speedup recorded; treat as neutral
    done < <(ls -t "$TOOL_DIR/workspace/accepted/"*.hip 2>/dev/null)

    # Source 2: kernel_lab generated — pick the best .hip per target dir by speedup
    while IFS= read -r result_json; do
        if grep -qE '"correctness"\s*:\s*"PASS"' "$result_json" 2>/dev/null; then
            local dir; dir="$(dirname "$result_json")"
            # Extract speedup_vs_ref (handles spaces around colon and quotes)
            local spd
            spd=$(grep -oE '"speedup_vs_ref"\s*:\s*"[0-9.]+"' "$result_json" \
                      2>/dev/null | grep -oE '[0-9.]+' | tail -1)
            spd="${spd:-1.0000}"
            # Pick the highest-numbered candidate .hip (newest within this dir)
            local hip; hip=$(ls -t "$dir"/candidate_*.hip 2>/dev/null | head -1)
            if [[ -f "$hip" ]]; then
                scored+=("$spd $hip")
            fi
        fi
    done < <(find "$TOOL_DIR/kernel_lab/generated" -name "harness_result.json" 2>/dev/null)

    # Sort by speedup descending, emit only the path
    printf '%s\n' "${scored[@]}" | sort -rn | awk '{print $2}'
}

# Compatibility wrapper — returns only the single best candidate.
find_best_candidate_hip() {
    find_all_candidate_hips_sorted | head -1
}

write_summary() {
    local rollback_report
    rollback_report=$(ls -t "$REPORTS_DIR"/phase16_rollback_*.md 2>/dev/null | head -1 || echo "none")

    cat > "$SUMMARY_MD" << SEOF
# AutoKernel Full Run — Summary

| Field | Value |
|---|---|
| Timestamp | $TIMESTAMP |
| ARCH | $ARCH |
| MODEL | $MODEL |
| MAX_ITERS | $MAX_ITERS |
| ACCEPT_MIN_SPEEDUP | $ACCEPT_MIN_SPEEDUP |
| TRIALS | $TRIALS |
| AUTOCOMMIT | $AUTOCOMMIT |
| Loop ran | $LOOP_RAN iterations |
| Candidate promoted | ${PROMOTED_CANDIDATE:-none} |
| Initial decision | $INITIAL_DECISION |
| **Final decision** | **$FINAL_DECISION** |
| Full log | \`$FULL_LOG\` |
| Rollback report | \`$rollback_report\` |

## Decision meanings

| Decision | Meaning |
|---|---|
| \`PASS_ADOPTED\` | ≥1 kernel promoted + verified tok/s improvement + all gates pass. Done. |
| \`PASS_INFRA_ONLY\` | All gates pass. No kernel improved tok/s yet — run more iterations. |
| \`NEEDS_BASELINE\` | Validation works but no 27B baseline exists for speedup comparison. |
| \`FAIL\` | Build or correctness gate failed. Fix before claiming adoption. |

## Rollback

\`\`\`bash
# See rollback instructions:
cat $rollback_report
\`\`\`

## Next step

$(case "$FINAL_DECISION" in
    PASS_ADOPTED)    echo "Adoption complete. Eyeball output, push, and run overnight loop for further gains." ;;
    PASS_INFRA_ONLY) echo "Run more iterations or generate a 27B baseline: \`ARCH=gfx1201 MODEL=qwen3.5:27b ./tools/autokernel-rdna/run.sh baseline\`" ;;
    NEEDS_BASELINE)  echo "Generate a 27B baseline first, then re-run this wrapper." ;;
    *)               echo "Fix the failing gate(s) shown in the Phase 16 report, then re-run." ;;
esac)
SEOF
}

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${C_BOLD}═══════════════════════════════════════════════════════${C_RESET}"
echo -e "${C_BOLD} hipfire AutoKernel Full Pipeline${C_RESET}"
echo -e "${C_BOLD}═══════════════════════════════════════════════════════${C_RESET}"
echo "  ARCH=$ARCH  MODEL=$MODEL"
echo "  MAX_ITERS=$MAX_ITERS  TRIALS=$TRIALS  AUTOCOMMIT=$AUTOCOMMIT"
echo "  Run dir: $RUN_DIR"
echo ""

# ── Step 1: Check required scripts ───────────────────────────────────────────
log "Step 1: Verifying required scripts..."
MISSING=0
for s in \
    "$TOOL_DIR/autokernel_loop.sh" \
    "$TOOL_DIR/promote_kernel.sh" \
    "$TOOL_DIR/final_validate.sh"
do
    if [[ ! -f "$s" ]]; then
        log_err "  MISSING: $s"
        MISSING=1
    else
        chmod +x "$s"
        log_ok "  ✓ $s"
    fi
done
if [[ $MISSING -eq 1 ]]; then
    log_err "Required scripts missing — cannot continue."
    FINAL_DECISION="FAIL"
    write_summary
    exit 1
fi

# ── Step 2: Initial final_validate ───────────────────────────────────────────
log "Step 2: Running initial final_validate.sh..."
ARCH="$ARCH" MODEL="$MODEL" TRIALS="$TRIALS" \
    SKIP_COHERENCE="$SKIP_COHERENCE" SKIP_BENCH="$SKIP_BENCH" \
    bash "$TOOL_DIR/final_validate.sh" || true

INITIAL_DECISION="$(latest_decision)"
log "  Initial decision: ${C_BOLD}$INITIAL_DECISION${C_RESET}"

if [[ "$INITIAL_DECISION" == "PASS_ADOPTED" ]]; then
    log_ok "hipfire in PASS_ADOPTED state — continuing loop to search for further improvements."
fi

if [[ "$INITIAL_DECISION" == "FAIL" && "$FORCE_CONTINUE" != "1" ]]; then
    log_err "Initial validation returned FAIL. Fix gates before continuing."
    log_err "Set FORCE_CONTINUE=1 to override."
    FINAL_DECISION="FAIL"
    write_summary
    exit 1
fi

# ── Step 3: Run AUTHOR_KERNEL search (if no manual CANDIDATE_DIR) ─────────────
if [[ -z "$CANDIDATE_DIR" ]]; then
    log "Step 3: Running AUTHOR_KERNEL search loop (MAX_ITERS=$MAX_ITERS)..."
    AUTHOR_KERNEL=1 \
    ARCH="$ARCH" \
    MODEL="$MODEL" \
    MAX_ITERS="$MAX_ITERS" \
    ACCEPT_MIN_SPEEDUP="$ACCEPT_MIN_SPEEDUP" \
    AUTOCOMMIT="$AUTOCOMMIT" \
    RUN_FINAL_VALIDATE="$RUN_FINAL_VALIDATE" \
        bash "$TOOL_DIR/autokernel_loop.sh" || true
    # Count rows added to results.tsv during this run as a proxy for iterations
    LOOP_RAN=$(wc -l < "$TOOL_DIR/results.tsv" 2>/dev/null || echo "$MAX_ITERS")
    log "  Loop finished."
else
    log "Step 3: CANDIDATE_DIR set — skipping search loop."
fi

# ── Step 4: Find best candidate ──────────────────────────────────────────────
log "Step 4: Finding best promotable candidate..."

BEST_CANDIDATE_HIP=""

if [[ -n "$CANDIDATE_DIR" ]]; then
    # User supplied a directory — resolve to a .hip file inside it
    cdir="$CANDIDATE_DIR"
    [[ "${cdir}" != /* ]] && cdir="$REPO_ROOT/$cdir"
    if [[ -f "$cdir" && "$cdir" == *.hip ]]; then
        # They passed a .hip file directly
        BEST_CANDIDATE_HIP="$cdir"
    elif [[ -d "$cdir" ]]; then
        BEST_CANDIDATE_HIP=$(ls -t "$cdir"/candidate_*.hip "$cdir"/*.hip 2>/dev/null | head -1)
    else
        log_err "  CANDIDATE_DIR='$CANDIDATE_DIR' is not a .hip file or directory."
    fi
    if [[ -n "$BEST_CANDIDATE_HIP" ]]; then
        log "  Using user-supplied candidate: $BEST_CANDIDATE_HIP"
    fi
fi

if [[ -z "$BEST_CANDIDATE_HIP" ]]; then
    BEST_CANDIDATE_HIP="$(find_best_candidate_hip)"
fi

if [[ -z "$BEST_CANDIDATE_HIP" || ! -f "$BEST_CANDIDATE_HIP" ]]; then
    log_err "No promotable candidate found."
    log_err ""
    log_err "To specify one manually:"
    log_err "  CANDIDATE_DIR=tools/autokernel-rdna/kernel_lab/generated/<target>/candidate_<N> \\"
    log_err "    ./run_hipfire_autokernel_full.sh"
    log_err ""
    log_err "Available candidates:"
    find "$TOOL_DIR/kernel_lab/generated" -name "candidate_*.hip" 2>/dev/null \
        | sort | while IFS= read -r f; do log_err "  $f"; done
    FINAL_DECISION="PASS_INFRA_ONLY"
    write_summary
    exit 2
fi

PROMOTED_CANDIDATE="$BEST_CANDIDATE_HIP"
log_ok "  Candidate: $BEST_CANDIDATE_HIP"

# ── Step 5: Promote — try all passing candidates in speedup order ─────────────
log "Step 5: Promoting candidate(s) via promote_kernel.sh..."
PROMOTE_RAN=1

# Build the full ranked list; user-supplied candidate is always first.
if [[ -n "$CANDIDATE_DIR" ]]; then
    ALL_CANDIDATES=("$BEST_CANDIDATE_HIP")
else
    mapfile -t ALL_CANDIDATES < <(find_all_candidate_hips_sorted)
    # Put the already-selected best first (it's already #1 but be explicit)
    [[ ${#ALL_CANDIDATES[@]} -eq 0 ]] && ALL_CANDIDATES=("$BEST_CANDIDATE_HIP")
fi

PROMOTE_ACCEPTED=0
for _cand in "${ALL_CANDIDATES[@]}"; do
    [[ -f "$_cand" ]] || continue
    log "  Trying candidate: $_cand"
    ARCH="$ARCH" \
    MODEL="$MODEL" \
    CANDIDATE="$_cand" \
    ACCEPT_MIN_SPEEDUP="$ACCEPT_MIN_SPEEDUP" \
    AUTOCOMMIT="$AUTOCOMMIT" \
    SKIP_COHERENCE="$SKIP_COHERENCE" \
        bash "$TOOL_DIR/promote_kernel.sh" && PROMOTE_ACCEPTED=1 && break || true
done

if [[ $PROMOTE_ACCEPTED -eq 0 ]]; then
    log_warn "  All ${#ALL_CANDIDATES[@]} candidate(s) rejected by promote_kernel.sh."
fi

# ── Step 6: Final validation ──────────────────────────────────────────────────
log "Step 6: Running final_validate.sh (post-promotion)..."
ARCH="$ARCH" MODEL="$MODEL" TRIALS="$TRIALS" \
    SKIP_COHERENCE="$SKIP_COHERENCE" SKIP_BENCH="$SKIP_BENCH" \
    bash "$TOOL_DIR/final_validate.sh" || true

FINAL_DECISION="$(latest_decision)"
log "  Final decision: ${C_BOLD}$FINAL_DECISION${C_RESET}"

# ── Step 7: Summary ───────────────────────────────────────────────────────────
write_summary

echo ""
echo -e "${C_BOLD}═══ PIPELINE COMPLETE ═══${C_RESET}"
echo ""
echo "  Initial decision: $INITIAL_DECISION"
echo -e "  Final decision:   ${C_BOLD}$FINAL_DECISION${C_RESET}"
echo ""
echo "  Summary:  $SUMMARY_MD"
echo "  Full log: $FULL_LOG"
echo ""

case "$FINAL_DECISION" in
    PASS_ADOPTED)
        echo -e "${C_GREEN}SUCCESS — hipfire adoption complete.${C_RESET}"
        exit 0
        ;;
    *)
        echo -e "${C_YELLOW}Not yet PASS_ADOPTED (${FINAL_DECISION}).${C_RESET}"
        echo "  See summary for next steps."
        exit 3
        ;;
esac
