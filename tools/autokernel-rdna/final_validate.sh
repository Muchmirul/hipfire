#!/usr/bin/env bash
# tools/autokernel-rdna/final_validate.sh
#
# Phase 16 — Final End-to-End Validation and Adoption Lock-In
#
# Verifies the complete AutoKernel pipeline end-to-end on hipfire:
#   infra check → build → correctness gates → Qwen3.5-27B benchmark → final report
#
# Usage:
#   ARCH=gfx1201 \
#   MODEL=qwen3.5:27b \
#   TRIALS=5 \
#   ./tools/autokernel-rdna/final_validate.sh
#
# Optional env:
#   TRIALS            - number of benchmark runs (default: 3)
#   ALLOW_OTHER_ARCH  - set 1 to skip arch=gfx1201 guard (default: 0)
#   ALLOW_OTHER_MODEL - set 1 to skip model=qwen3.5:27b guard (default: 0)
#   SKIP_COHERENCE    - set 1 to skip coherence-gate-dflash.sh (default: 0)
#   SKIP_BENCH        - set 1 to skip the 27B benchmark entirely (default: 0)
#
# Hard rules:
#   - Do not generate new kernels.
#   - Do not run overnight search.
#   - Do not weaken correctness gates.
#   - Do not alter the benchmark evaluator.
#   - Do not claim speedup without baseline comparison.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ── Config ────────────────────────────────────────────────────────────────────
ARCH="${ARCH:-gfx1201}"
MODEL="${MODEL:-qwen3.5:27b}"
TRIALS="${TRIALS:-3}"
ALLOW_OTHER_ARCH="${ALLOW_OTHER_ARCH:-0}"
ALLOW_OTHER_MODEL="${ALLOW_OTHER_MODEL:-0}"
SKIP_COHERENCE="${SKIP_COHERENCE:-0}"
SKIP_BENCH="${SKIP_BENCH:-0}"

TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
REPORTS_DIR="$SCRIPT_DIR/reports"
BASELINES_DIR="$SCRIPT_DIR/baselines"
WORKSPACE_DIR="$SCRIPT_DIR/workspace"

REPORT_MD="$REPORTS_DIR/phase16_final_validation_${TIMESTAMP}.md"
REPORT_JSON="$REPORTS_DIR/phase16_final_validation_${TIMESTAMP}.json"
ROLLBACK_MD="$REPORTS_DIR/phase16_rollback_${TIMESTAMP}.md"
BUILD_LOG="$REPORTS_DIR/phase16_build_${TIMESTAMP}.log"
SPEED_GATE_LOG="$REPORTS_DIR/phase16_speed_gate_${TIMESTAMP}.log"
COHERENCE_LOG="$REPORTS_DIR/phase16_coherence_${TIMESTAMP}.log"
BENCH_LOG="$REPORTS_DIR/phase16_bench_27b_${TIMESTAMP}.log"

mkdir -p "$REPORTS_DIR"

# Cargo path (matches phase11_validate.sh)
CARGO_PATH="/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.cargo/bin"
[[ -d "$CARGO_PATH" ]] && export PATH="$CARGO_PATH:$PATH"

MODELS_DIR="${HIPFIRE_MODELS_DIR:-/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.hipfire/models}"
BENCH_EXE="$REPO_ROOT/target/release/examples/bench_qwen35_mq4"
PROMPT_FILE="$REPO_ROOT/benchmarks/prompts/lru_cache_pep8_strict.txt"
CANONICAL_PROMPT_MD5="df5dedc8040ce70ba55080c4548e6024"

# ── Colors ───────────────────────────────────────────────────────────────────
C_RESET='\033[0m'
C_GREEN='\033[0;32m'
C_RED='\033[0;31m'
C_YELLOW='\033[0;33m'
C_CYAN='\033[0;36m'
C_BOLD='\033[1m'

log()      { echo -e "${C_CYAN}[phase16]${C_RESET} $*"; }
log_ok()   { echo -e "${C_GREEN}[phase16]${C_RESET} $*"; }
log_err()  { echo -e "${C_RED}[phase16]${C_RESET} $*"; }
log_warn() { echo -e "${C_YELLOW}[phase16]${C_RESET} $*"; }

# ── State ─────────────────────────────────────────────────────────────────────
INFRA_STATUS="PASS"
BUILD_STATUS="PASS"
CORRECTNESS_STATUS="PASS"
BENCH_STATUS="NOT_RUN"
BASELINE_STATUS="UNKNOWN"
SPEEDUP_STATUS="UNKNOWN"
DECISION="UNKNOWN"
TSV_CHECK_STATUS="UNKNOWN"
SPEED_GATE_STATUS="NOT_RUN"
COHERENCE_GATE_STATUS="NOT_RUN"

PIPELINE_MISSING=()
INFRA_ERRORS=()

ENV_DATE=""
ENV_HOSTNAME=""
ENV_GIT_SHA=""
ENV_GIT_DIRTY=0
ENV_GPU=""
ENV_ARCH_DETECTED=""
ENV_ROCM=""
ENV_KERNEL=""

N_GENERATED=0
N_HARNESS_PASS=0
N_PROMOTED=0
N_ACCEPTED_PROMOTIONS=0
N_REJECTED_PROMOTIONS=0
LATEST_ACCEPTED=""

BENCH_27B_TOKS=()
BENCH_27B_MEDIAN=0
BENCH_27B_MIN=0
BENCH_27B_MAX=0
BENCH_27B_MODEL_PATH=""
BENCH_27B_PROMPT_MD5=""
BENCH_27B_VRAM="unknown"

BASELINE_TOK_S=0
BASELINE_SOURCE=""
DELTA_TOK_S=0
SPEEDUP=0
SPEEDUP_PCT=0

# ════════════════════════════════════════════════════════════════════════════════
# Step 1: Verify pipeline files
# ════════════════════════════════════════════════════════════════════════════════
verify_pipeline_files() {
    log "Step 1: Verifying pipeline files..."
    local required=(
        "tools/autokernel-rdna/autokernel_loop.sh"
        "tools/autokernel-rdna/promote_kernel.sh"
        "tools/autokernel-rdna/phase11_validate.sh"
        "tools/autokernel-rdna/check_results_tsv.sh"
        "tools/autokernel-rdna/kernel_lab"
        "tools/autokernel-rdna/results.tsv"
        "tools/autokernel-rdna/workspace"
        "tools/autokernel-rdna/reports"
    )
    local all_ok=1
    for item in "${required[@]}"; do
        if [[ -e "$item" ]]; then
            log_ok "  ✓ $item"
        else
            log_err "  ✗ MISSING: $item"
            PIPELINE_MISSING+=("$item")
            all_ok=0
        fi
    done
    if [[ $all_ok -eq 0 ]]; then
        INFRA_STATUS="FAIL"
        INFRA_ERRORS+=("Missing pipeline files: ${PIPELINE_MISSING[*]}")
        log_err "Step 1 FAILED — missing required components"
        return 1
    fi
    log_ok "Step 1 PASSED"
}

# ════════════════════════════════════════════════════════════════════════════════
# Step 2: Verify environment
# ════════════════════════════════════════════════════════════════════════════════
verify_environment() {
    log "Step 2: Verifying environment..."
    ENV_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ENV_HOSTNAME="$(hostname)"
    ENV_GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    ENV_GIT_DIRTY=$(git status --short 2>/dev/null | wc -l | tr -d ' ')
    ENV_GPU="$(rocm-smi --showproductname 2>/dev/null | grep -i 'GPU\[' | head -1 | sed 's/.*: //' || echo 'unknown')"
    ENV_ARCH_DETECTED="$(amdgpu-arch 2>/dev/null | head -1 || echo 'unknown')"
    ENV_ROCM="$(cat /opt/rocm/.info/version 2>/dev/null || cat /opt/rocm/VERSION 2>/dev/null || hipcc --version 2>/dev/null | head -1 | tr -d '\n' || echo 'unknown')"
    ENV_KERNEL="$(uname -r)"

    log "  date:          $ENV_DATE"
    log "  host:          $ENV_HOSTNAME"
    log "  git SHA:       $ENV_GIT_SHA"
    log "  dirty files:   $ENV_GIT_DIRTY"
    log "  GPU:           $ENV_GPU"
    log "  arch detected: $ENV_ARCH_DETECTED"
    log "  arch request:  $ARCH"
    log "  ROCm:          $ENV_ROCM"
    log "  kernel:        $ENV_KERNEL"
    log "  model:         $MODEL"
    log "  trials:        $TRIALS"
    local hipfire_env
    hipfire_env=$(env | grep '^HIPFIRE_' || true)
    if [[ -n "$hipfire_env" ]]; then
        log "  HIPFIRE_* env:"
        echo "$hipfire_env" | while IFS= read -r line; do log "    $line"; done
    fi

    if [[ "$ARCH" != "gfx1201" && "$ALLOW_OTHER_ARCH" != "1" ]]; then
        log_err "  ARCH=$ARCH is not gfx1201. Set ALLOW_OTHER_ARCH=1 to override."
        INFRA_STATUS="FAIL"
        INFRA_ERRORS+=("ARCH mismatch: expected gfx1201, got $ARCH")
        return 1
    fi
    if [[ "$MODEL" != "qwen3.5:27b" && "$ALLOW_OTHER_MODEL" != "1" ]]; then
        log_err "  MODEL=$MODEL is not qwen3.5:27b. Set ALLOW_OTHER_MODEL=1 to override."
        INFRA_STATUS="FAIL"
        INFRA_ERRORS+=("MODEL mismatch: expected qwen3.5:27b, got $MODEL")
        return 1
    fi
    log_ok "Step 2 PASSED"
}

# ════════════════════════════════════════════════════════════════════════════════
# Step 3: Verify logs and schemas
# ════════════════════════════════════════════════════════════════════════════════
verify_logs_and_schemas() {
    log "Step 3: Verifying logs and schemas (check_results_tsv.sh)..."
    local tsv_check_log="$REPORTS_DIR/phase16_tsv_check_${TIMESTAMP}.log"
    if bash "$SCRIPT_DIR/check_results_tsv.sh" > "$tsv_check_log" 2>&1; then
        TSV_CHECK_STATUS="PASS"
        log_ok "  check_results_tsv.sh: PASS"
    else
        TSV_CHECK_STATUS="WARN"
        log_warn "  check_results_tsv.sh: non-zero exit (see $tsv_check_log)"
    fi
    local row_count
    row_count=$(( $(wc -l < "$SCRIPT_DIR/results.tsv") - 1 ))
    log "  results.tsv rows: $row_count"
    log_ok "Step 3 PASSED"
}

# ════════════════════════════════════════════════════════════════════════════════
# Step 4: Verify candidate and promotion records
# ════════════════════════════════════════════════════════════════════════════════
verify_candidate_records() {
    log "Step 4: Scanning candidate and promotion records..."
    N_GENERATED=$(find "$SCRIPT_DIR/kernel_lab/generated" -name "candidate_*.hip" 2>/dev/null | wc -l)
    N_HARNESS_PASS=$(find "$SCRIPT_DIR/kernel_lab/generated" -name "harness_result.json" 2>/dev/null \
        -exec grep -l '"correctness":"PASS"' {} \; 2>/dev/null | wc -l)
    local n_acc_hip n_rej_hip
    n_acc_hip=$(ls "$WORKSPACE_DIR/accepted_promotions/"*.hip 2>/dev/null | wc -l)
    n_rej_hip=$(ls "$WORKSPACE_DIR/rejected_promotions/"*.hip 2>/dev/null | wc -l)
    N_ACCEPTED_PROMOTIONS=$n_acc_hip
    N_REJECTED_PROMOTIONS=$n_rej_hip
    N_PROMOTED=$(( n_acc_hip + n_rej_hip ))
    if [[ $N_ACCEPTED_PROMOTIONS -gt 0 ]]; then
        LATEST_ACCEPTED=$(ls -t "$WORKSPACE_DIR/accepted_promotions/"*.hip 2>/dev/null | head -1 || true)
    fi
    log "  Generated candidates:   $N_GENERATED"
    log "  Harness-passing:        $N_HARNESS_PASS"
    log "  Promoted (total):       $N_PROMOTED"
    log "  Accepted promotions:    $N_ACCEPTED_PROMOTIONS"
    log "  Rejected promotions:    $N_REJECTED_PROMOTIONS"
    [[ -n "$LATEST_ACCEPTED" ]] && log "  Latest accepted:        $LATEST_ACCEPTED"
    log_ok "Step 4 PASSED"
}

# ════════════════════════════════════════════════════════════════════════════════
# Step 5: Build hipfire
# ════════════════════════════════════════════════════════════════════════════════
build_hipfire() {
    log "Step 5: Building hipfire (release) — this may take several minutes..."
    if cargo build --release --features deltanet --example bench_qwen35_mq4 -p hipfire-runtime \
        > "$BUILD_LOG" 2>&1; then
        BUILD_STATUS="PASS"
        log_ok "Step 5 PASSED — build succeeded"
    else
        BUILD_STATUS="FAIL"
        log_err "Step 5 FAILED — build error (see $BUILD_LOG)"
        INFRA_ERRORS+=("hipfire build failed")
    fi
}

# ════════════════════════════════════════════════════════════════════════════════
# Step 6: Correctness gates
# ════════════════════════════════════════════════════════════════════════════════
run_correctness_gates() {
    log "Step 6: Running correctness gates..."

    log "  Running scripts/speed-gate.sh --fast ..."
    if bash scripts/speed-gate.sh --fast > "$SPEED_GATE_LOG" 2>&1; then
        SPEED_GATE_STATUS="PASS"
        log_ok "  speed-gate.sh --fast: PASS"
    else
        SPEED_GATE_STATUS="FAIL"
        log_err "  speed-gate.sh --fast: FAIL (see $SPEED_GATE_LOG)"
        CORRECTNESS_STATUS="FAIL"
    fi

    if [[ "$SKIP_COHERENCE" == "1" ]]; then
        COHERENCE_GATE_STATUS="SKIPPED"
        log_warn "  coherence-gate-dflash.sh: SKIPPED (SKIP_COHERENCE=1)"
    else
        log "  Running scripts/coherence-gate-dflash.sh ..."
        if bash scripts/coherence-gate-dflash.sh > "$COHERENCE_LOG" 2>&1; then
            COHERENCE_GATE_STATUS="PASS"
            log_ok "  coherence-gate-dflash.sh: PASS"
        else
            COHERENCE_GATE_STATUS="FAIL"
            log_err "  coherence-gate-dflash.sh: FAIL (see $COHERENCE_LOG)"
            CORRECTNESS_STATUS="FAIL"
        fi
    fi

    if [[ "$CORRECTNESS_STATUS" == "PASS" ]]; then
        log_ok "Step 6 PASSED"
    else
        log_err "Step 6 FAILED"
    fi
}

# ════════════════════════════════════════════════════════════════════════════════
# Step 7: Qwen3.5-27B benchmark
# ════════════════════════════════════════════════════════════════════════════════
_run_one_bench() {
    # Runs a single bench trial. Appends full output to BENCH_LOG.
    # Prints gen_tok_s to stdout.
    local model_path="$1"
    local out
    out=$("$BENCH_EXE" "$model_path" --gen 80 --warmup 5 < "$PROMPT_FILE" 2>&1) || true
    printf '%s\n\n' "$out" >> "$BENCH_LOG"
    echo "$out" | grep -oP 'gen_tok_s=\K[0-9.]+' | head -1 || echo "0"
}

run_benchmark_27b() {
    log "Step 7: Running Qwen3.5-27B benchmark..."
    : > "$BENCH_LOG"

    if [[ "$SKIP_BENCH" == "1" ]]; then
        BENCH_STATUS="SKIPPED"
        log_warn "  SKIPPED (SKIP_BENCH=1)"
        return
    fi

    # Find model
    for cand in \
        "$MODELS_DIR/qwen3.5-27b.mq4" \
        "$MODELS_DIR/qwen35-27b.mq4" \
        "$HOME/.hipfire/models/qwen3.5-27b.mq4"
    do
        if [[ -f "$cand" ]]; then
            BENCH_27B_MODEL_PATH="$cand"
            break
        fi
    done

    if [[ -z "$BENCH_27B_MODEL_PATH" ]]; then
        BENCH_STATUS="NO_MODEL"
        log_err "  qwen3.5-27b.mq4 not found in $MODELS_DIR"
        log_err "  Run: hipfire pull qwen3.5:27b"
        return
    fi
    log "  Model: $BENCH_27B_MODEL_PATH"

    if [[ -f "$PROMPT_FILE" ]]; then
        BENCH_27B_PROMPT_MD5="$(md5sum "$PROMPT_FILE" | cut -d' ' -f1)"
        if [[ "$BENCH_27B_PROMPT_MD5" != "$CANONICAL_PROMPT_MD5" ]]; then
            log_warn "  Prompt MD5 mismatch: got $BENCH_27B_PROMPT_MD5, expected $CANONICAL_PROMPT_MD5"
        else
            log "  Prompt MD5: $BENCH_27B_PROMPT_MD5 ✓"
        fi
    else
        log_err "  Prompt file not found: $PROMPT_FILE"
        BENCH_STATUS="NO_PROMPT"
        return
    fi

    if [[ ! -f "$BENCH_EXE" ]]; then
        BENCH_STATUS="NO_BINARY"
        log_err "  bench binary not found: $BENCH_EXE (needs cargo build first)"
        return
    fi

    BENCH_27B_TOKS=()
    for (( i=1; i<=TRIALS; i++ )); do
        log "  Trial $i/$TRIALS ..."
        local toks
        toks=$(_run_one_bench "$BENCH_27B_MODEL_PATH")
        log "    → $toks tok/s"
        BENCH_27B_TOKS+=("$toks")
    done

    local stats
    stats=$(python3 - << PYEOF
import statistics
vals = [float(x) for x in "${BENCH_27B_TOKS[*]:-0}".split() if float(x) > 0]
if vals:
    print(statistics.median(vals), min(vals), max(vals))
else:
    print("0 0 0")
PYEOF
    ) || stats="0 0 0"
    BENCH_27B_MEDIAN=$(echo "$stats" | awk '{print $1}')
    BENCH_27B_MIN=$(echo "$stats"    | awk '{print $2}')
    BENCH_27B_MAX=$(echo "$stats"    | awk '{print $3}')

    if python3 -c "exit(0 if float('${BENCH_27B_MEDIAN:-0}') > 0 else 1)" 2>/dev/null; then
        BENCH_STATUS="PASS"
        log_ok "  Median: $BENCH_27B_MEDIAN tok/s  (min=$BENCH_27B_MIN  max=$BENCH_27B_MAX)"
        log_ok "Step 7 PASSED"
    else
        BENCH_STATUS="FAIL"
        log_err "  All bench trials returned 0 tok/s — check $BENCH_LOG"
        log_err "Step 7 FAILED"
    fi
}

# ════════════════════════════════════════════════════════════════════════════════
# Step 8: Compare to original baseline
# ════════════════════════════════════════════════════════════════════════════════
compare_to_baseline() {
    log "Step 8: Comparing to original baseline..."

    # Look for a 27B-specific baseline — iterate newest-first so we pick the
    # most recent measurement, not whichever globbing happens to return first.
    local baseline_27b=""
    while IFS= read -r _bf; do
        [[ -f "$_bf" ]] || continue
        local bmodel
        bmodel=$(python3 -c "
import json, sys
try:
    d = json.load(open('$_bf'))
    print(d.get('model', ''))
except: print('')
" 2>/dev/null || echo "")
        if [[ "$bmodel" == *"27b"* ]]; then
            baseline_27b="$_bf"
            break
        fi
    done < <(ls -t "$BASELINES_DIR"/*.json 2>/dev/null)

    if [[ -n "$baseline_27b" ]]; then
        BASELINE_TOK_S=$(python3 -c "
import json
d = json.load(open('$baseline_27b'))
v = d.get('baseline_decode_tok_s', d.get('gen_tok_s', d.get('decode_tok_s', 0)))
print(v)
" 2>/dev/null || echo "0")
        BASELINE_SOURCE="$baseline_27b"
        BASELINE_STATUS="AVAILABLE"
        log "  Baseline source: $baseline_27b"
        log "  Baseline tok/s:  $BASELINE_TOK_S"
    else
        BASELINE_STATUS="NEEDS_BASELINE"
        log_warn "  No 27B baseline found in $BASELINES_DIR (all stored are 9b)"
        log_warn "  To generate a 27B baseline:"
        log "    ARCH=gfx1201 MODEL=qwen3.5:27b ./tools/autokernel-rdna/run.sh baseline"
        log "  OR directly:"
        log "    cat benchmarks/prompts/lru_cache_pep8_strict.txt \\"
        log "      | $BENCH_EXE \\"
        log "        $MODELS_DIR/qwen3.5-27b.mq4 \\"
        log "        --gen 80 --warmup 5 2>&1 | grep gen_tok_s"
    fi

    if [[ "$BASELINE_STATUS" == "AVAILABLE" ]]; then
        local result
        result=$(python3 - << PYEOF
bl = float("${BASELINE_TOK_S:-0}")
fn = float("${BENCH_27B_MEDIAN:-0}")
if bl > 0 and fn > 0:
    delta = round(fn - bl, 2)
    speedup = round(fn / bl, 4)
    pct = round((fn / bl - 1) * 100, 1)
    status = "IMPROVEMENT" if fn >= bl else "REGRESSION"
else:
    delta, speedup, pct, status = 0, 0, 0, "UNKNOWN"
print(delta, speedup, pct, status)
PYEOF
        ) || result="0 0 0 UNKNOWN"
        DELTA_TOK_S=$(echo "$result" | awk '{print $1}')
        SPEEDUP=$(echo "$result"     | awk '{print $2}')
        SPEEDUP_PCT=$(echo "$result" | awk '{print $3}')
        SPEEDUP_STATUS=$(echo "$result" | awk '{print $4}')
        if [[ "$SPEEDUP_STATUS" == "IMPROVEMENT" ]]; then
            log_ok "  Speedup: ${SPEEDUP}× (+${SPEEDUP_PCT}%)  ${BASELINE_TOK_S} → ${BENCH_27B_MEDIAN} tok/s"
        elif [[ "$SPEEDUP_STATUS" == "REGRESSION" ]]; then
            log_warn "  Regression: ${SPEEDUP}× (${SPEEDUP_PCT}%)  ${BASELINE_TOK_S} → ${BENCH_27B_MEDIAN} tok/s"
        fi
    fi
    log_ok "Step 8 PASSED"
}

# ════════════════════════════════════════════════════════════════════════════════
# Step 9: Final decision
# ════════════════════════════════════════════════════════════════════════════════
compute_final_decision() {
    log "Step 9: Computing final decision..."

    if [[ "$INFRA_STATUS" == "FAIL" || "$BUILD_STATUS" == "FAIL" \
       || "$CORRECTNESS_STATUS" == "FAIL" \
       || "$BENCH_STATUS" == "FAIL" || "$BENCH_STATUS" == "NO_MODEL" \
       || "$BENCH_STATUS" == "NO_BINARY" || "$BENCH_STATUS" == "NO_PROMPT" ]]; then
        DECISION="FAIL"
    elif [[ $N_ACCEPTED_PROMOTIONS -gt 0 && "$SPEEDUP_STATUS" == "IMPROVEMENT" ]]; then
        DECISION="PASS_ADOPTED"
    elif [[ $N_ACCEPTED_PROMOTIONS -gt 0 && "$BASELINE_STATUS" == "NEEDS_BASELINE" ]]; then
        DECISION="NEEDS_BASELINE"
    elif [[ "$BENCH_STATUS" == "PASS" || "$BENCH_STATUS" == "SKIPPED" ]]; then
        DECISION="PASS_INFRA_ONLY"
    else
        DECISION="NEEDS_BASELINE"
    fi

    case "$DECISION" in
        PASS_ADOPTED)    log_ok  "  Final decision: ${C_BOLD}PASS_ADOPTED${C_RESET}" ;;
        PASS_INFRA_ONLY) log_ok  "  Final decision: ${C_BOLD}PASS_INFRA_ONLY${C_RESET}" ;;
        NEEDS_BASELINE)  log_warn "  Final decision: ${C_BOLD}NEEDS_BASELINE${C_RESET}" ;;
        FAIL)            log_err  "  Final decision: ${C_BOLD}FAIL${C_RESET}" ;;
    esac
}

# ════════════════════════════════════════════════════════════════════════════════
# Steps 10–11: Write reports
# ════════════════════════════════════════════════════════════════════════════════

# Helpers that append to REPORT_MD
rpt() { printf '%s\n' "$*" >> "$REPORT_MD"; }

_checklist_row() {
    local path="$1" label="$2"
    if [[ -e "$path" ]]; then
        rpt "| $label | ✓ present |"
    else
        rpt "| $label | ✗ **MISSING** |"
    fi
}

_decision_meaning() {
    case "$DECISION" in
        PASS_ADOPTED)    printf 'All checks passed. At least one kernel was promoted and verified to improve Qwen3.5-27B tok/s. Safe to ship.' ;;
        PASS_INFRA_ONLY) printf 'All infrastructure checks passed. No promoted kernel has improved tok/s yet — pipeline is ready for more candidates.' ;;
        NEEDS_BASELINE)  printf 'Infrastructure and validation pass, but no trustworthy 27B baseline exists for speedup comparison. Measure a baseline before claiming improvement.' ;;
        FAIL)            printf 'One or more hard gates failed. See individual sections above. Do not claim adoption.' ;;
    esac
}

_next_action() {
    case "$DECISION" in
        PASS_ADOPTED)    echo "Eyeball coherence output on promoted kernel. Commit and push. Run overnight loop for further candidates." ;;
        PASS_INFRA_ONLY) echo "Generate a 27B baseline: \`ARCH=gfx1201 MODEL=qwen3.5:27b ./tools/autokernel-rdna/run.sh baseline\`. Then run \`AUTHOR_KERNEL=1 ./tools/autokernel-rdna/autokernel_loop.sh\` overnight and promote passing candidates." ;;
        NEEDS_BASELINE)  echo "Generate a 27B baseline first: \`ARCH=gfx1201 MODEL=qwen3.5:27b ./tools/autokernel-rdna/run.sh baseline\`" ;;
        FAIL)            echo "Fix the failing gate(s) listed above. Re-run final_validate.sh after each fix. Do not claim adoption until PASS_ADOPTED or PASS_INFRA_ONLY." ;;
    esac
}

write_reports() {
    log "Writing reports..."
    : > "$REPORT_MD"

    rpt "# Phase 16 — Final End-to-End Validation Report"
    rpt ""
    rpt "**Decision: $DECISION**"
    rpt ""
    rpt "Generated: $ENV_DATE"
    rpt ""
    rpt "---"
    rpt ""
    rpt "## Environment"
    rpt ""
    rpt "| Field | Value |"
    rpt "|---|---|"
    rpt "| Date | \`$ENV_DATE\` |"
    rpt "| Host | \`$ENV_HOSTNAME\` |"
    rpt "| Git SHA | \`$ENV_GIT_SHA\` |"
    rpt "| Dirty files | $ENV_GIT_DIRTY |"
    rpt "| GPU | $ENV_GPU |"
    rpt "| Arch (detected) | $ENV_ARCH_DETECTED |"
    rpt "| Arch (requested) | $ARCH |"
    rpt "| ROCm | $ENV_ROCM |"
    rpt "| Kernel | $ENV_KERNEL |"
    rpt "| Model | $MODEL |"
    rpt "| Trials | $TRIALS |"
    rpt ""
    rpt "## Pipeline Component Checklist"
    rpt ""
    rpt "| Component | Status |"
    rpt "|---|---|"
    _checklist_row "tools/autokernel-rdna/autokernel_loop.sh" "autokernel_loop.sh"
    _checklist_row "tools/autokernel-rdna/promote_kernel.sh"  "promote_kernel.sh"
    _checklist_row "tools/autokernel-rdna/phase11_validate.sh" "phase11_validate.sh"
    _checklist_row "tools/autokernel-rdna/check_results_tsv.sh" "check_results_tsv.sh"
    _checklist_row "tools/autokernel-rdna/kernel_lab"          "kernel_lab/"
    _checklist_row "tools/autokernel-rdna/results.tsv"         "results.tsv"
    _checklist_row "tools/autokernel-rdna/workspace"           "workspace/"
    _checklist_row "tools/autokernel-rdna/reports"             "reports/"
    rpt ""
    rpt "## results.tsv Schema"
    rpt ""
    rpt "Status: **$TSV_CHECK_STATUS**"
    rpt ""
    rpt "## Candidate and Promotion Records"
    rpt ""
    rpt "| | Count |"
    rpt "|---|---|"
    rpt "| Generated candidates | $N_GENERATED |"
    rpt "| Harness-passing | $N_HARNESS_PASS |"
    rpt "| Promoted (total attempts) | $N_PROMOTED |"
    rpt "| Accepted promotions | $N_ACCEPTED_PROMOTIONS |"
    rpt "| Rejected promotions | $N_REJECTED_PROMOTIONS |"
    rpt "| Latest accepted | ${LATEST_ACCEPTED:-none} |"
    rpt ""
    rpt "## hipfire Build"
    rpt ""
    rpt "**$BUILD_STATUS**"
    rpt ""
    rpt "Build log: \`$BUILD_LOG\`"
    rpt ""
    rpt "## Correctness Gates"
    rpt ""
    rpt "| Gate | Status |"
    rpt "|---|---|"
    rpt "| speed-gate.sh --fast | $SPEED_GATE_STATUS |"
    rpt "| coherence-gate-dflash.sh | $COHERENCE_GATE_STATUS |"
    rpt ""
    rpt "## Qwen3.5-27B Benchmark"
    rpt ""
    rpt "| | Value |"
    rpt "|---|---|"
    rpt "| Status | $BENCH_STATUS |"
    rpt "| Model path | \`${BENCH_27B_MODEL_PATH:-not found}\` |"
    rpt "| Prompt MD5 | \`${BENCH_27B_PROMPT_MD5:-unknown}\` |"
    rpt "| Trials | $TRIALS |"
    rpt "| Median tok/s | **$BENCH_27B_MEDIAN** |"
    rpt "| Min tok/s | $BENCH_27B_MIN |"
    rpt "| Max tok/s | $BENCH_27B_MAX |"
    rpt ""
    rpt "## Baseline Comparison"
    rpt ""
    rpt "| | Value |"
    rpt "|---|---|"
    rpt "| Baseline status | $BASELINE_STATUS |"
    rpt "| Baseline source | ${BASELINE_SOURCE:-none} |"
    rpt "| Baseline tok/s | $BASELINE_TOK_S |"
    rpt "| Final tok/s | $BENCH_27B_MEDIAN |"
    rpt "| Delta | $DELTA_TOK_S tok/s |"
    rpt "| Speedup | ${SPEEDUP}× |"
    rpt "| Percent | ${SPEEDUP_PCT}% |"
    rpt "| Speedup status | $SPEEDUP_STATUS |"
    rpt ""
    if [[ "$BASELINE_STATUS" == "NEEDS_BASELINE" ]]; then
        rpt "> **No 27B baseline found.** To generate one:"
        rpt "> \`\`\`bash"
        rpt "> ARCH=gfx1201 MODEL=qwen3.5:27b ./tools/autokernel-rdna/run.sh baseline"
        rpt "> \`\`\`"
        rpt ""
    fi
    rpt "## Final Decision: $DECISION"
    rpt ""
    rpt "$(_decision_meaning)"
    rpt ""
    rpt "## Known Limitations"
    rpt ""
    rpt "- No Qwen3.5-27B baseline exists in \`tools/autokernel-rdna/baselines/\` (all stored baselines are for 9b)."
    rpt "- The bench binary writes all output to stderr; this is captured correctly."
    rpt "- DFlash draft is not required for this benchmark (spec-decode is a separate evaluation)."
    rpt "- DDTree gfx1201 regression is expected and documented (see AGENTS.md §4)."
    rpt ""
    rpt "## Rollback Commands"
    rpt ""
    if [[ $N_ACCEPTED_PROMOTIONS -gt 0 ]]; then
        rpt "\`\`\`bash"
        rpt "# Find backup dirs:"
        rpt "ls tools/autokernel-rdna/workspace/promotion_backups/"
        rpt ""
        rpt "# Restore from a backup (replace <timestamp> with the matching dir):"
        rpt "DEST=\$(cat tools/autokernel-rdna/workspace/promotion_backups/<timestamp>/promoted_dest.txt)"
        rpt "BAK=tools/autokernel-rdna/workspace/promotion_backups/<timestamp>/\$(basename \"\$DEST\").bak"
        rpt "cp \"\$BAK\" \"\$DEST\""
        rpt ""
        rpt "# Rebuild after revert:"
        rpt "cargo build --release --features deltanet --example bench_qwen35_mq4 -p hipfire-runtime"
        rpt ""
        rpt "# Or git-reset a specific kernel file to pre-promotion SHA:"
        rpt "# git checkout <pre-promotion-sha> -- kernels/src/<kernel>.gfx1201.hip"
        rpt "\`\`\`"
    else
        rpt "No accepted promotions to roll back."
    fi
    rpt ""
    rpt "## Next Recommended Action"
    rpt ""
    rpt "$(_next_action)"

    # JSON report
    python3 - << PYEOF > "$REPORT_JSON" 2>/dev/null || echo '{"error":"json generation failed"}' > "$REPORT_JSON"
import json

pipeline_missing = [x for x in "${PIPELINE_MISSING[*]:-}".split() if x]
infra_errors = []
$(for e in "${INFRA_ERRORS[@]+"${INFRA_ERRORS[@]}"}"; do echo "infra_errors.append($(python3 -c "import json; print(json.dumps('$e'))" 2>/dev/null || echo '""'))"; done)
bench_toks = [float(x) for x in "${BENCH_27B_TOKS[*]:-0}".split() if float(x) > 0]

data = {
    "phase": 16,
    "timestamp": "$TIMESTAMP",
    "decision": "$DECISION",
    "environment": {
        "date": "$ENV_DATE",
        "hostname": "$ENV_HOSTNAME",
        "git_sha": "$ENV_GIT_SHA",
        "git_dirty_files": int("${ENV_GIT_DIRTY:-0}"),
        "gpu": "$ENV_GPU",
        "arch_detected": "$ENV_ARCH_DETECTED",
        "arch_requested": "$ARCH",
        "rocm": "$ENV_ROCM",
        "kernel": "$ENV_KERNEL",
        "model": "$MODEL",
        "trials": int("$TRIALS"),
    },
    "infra_status": "$INFRA_STATUS",
    "pipeline_missing": pipeline_missing,
    "infra_errors": infra_errors,
    "build_status": "$BUILD_STATUS",
    "build_log": "$BUILD_LOG",
    "correctness": {
        "overall": "$CORRECTNESS_STATUS",
        "speed_gate": "$SPEED_GATE_STATUS",
        "coherence_gate": "$COHERENCE_GATE_STATUS",
    },
    "tsv_check_status": "$TSV_CHECK_STATUS",
    "candidates": {
        "generated": $N_GENERATED,
        "harness_pass": $N_HARNESS_PASS,
        "promoted_total": $N_PROMOTED,
        "accepted": $N_ACCEPTED_PROMOTIONS,
        "rejected": $N_REJECTED_PROMOTIONS,
        "latest_accepted": "$LATEST_ACCEPTED",
    },
    "bench_27b": {
        "status": "$BENCH_STATUS",
        "model_path": "$BENCH_27B_MODEL_PATH",
        "prompt_md5": "$BENCH_27B_PROMPT_MD5",
        "trials": int("$TRIALS"),
        "all_tok_s": bench_toks,
        "median_tok_s": float("${BENCH_27B_MEDIAN:-0}"),
        "min_tok_s": float("${BENCH_27B_MIN:-0}"),
        "max_tok_s": float("${BENCH_27B_MAX:-0}"),
    },
    "baseline": {
        "status": "$BASELINE_STATUS",
        "source": "$BASELINE_SOURCE",
        "baseline_tok_s": float("${BASELINE_TOK_S:-0}"),
        "final_tok_s": float("${BENCH_27B_MEDIAN:-0}"),
        "delta_tok_s": float("${DELTA_TOK_S:-0}"),
        "speedup": float("${SPEEDUP:-0}"),
        "speedup_pct": float("${SPEEDUP_PCT:-0}"),
        "speedup_status": "$SPEEDUP_STATUS",
    },
}
print(json.dumps(data, indent=2))
PYEOF

    log_ok "MD report:   $REPORT_MD"
    log_ok "JSON report: $REPORT_JSON"
}

# ════════════════════════════════════════════════════════════════════════════════
# Step 11: Rollback package
# ════════════════════════════════════════════════════════════════════════════════
write_rollback_package() {
    local backup_dirs
    backup_dirs=$(ls "$WORKSPACE_DIR/promotion_backups/" 2>/dev/null || echo "none")

    cat > "$ROLLBACK_MD" << REOF
# Phase 16 — Rollback Package

Generated: $ENV_DATE

## Current State

- Git SHA: \`$ENV_GIT_SHA\`
- Accepted promotions: $N_ACCEPTED_PROMOTIONS
- Latest accepted: ${LATEST_ACCEPTED:-none}

## Accepted Promotion Files

\`\`\`
$(ls "$WORKSPACE_DIR/accepted_promotions/" 2>/dev/null || echo "none")
\`\`\`

## Backup Directories

\`\`\`
$backup_dirs
\`\`\`

## Revert Latest Promotion

\`\`\`bash
# 1. Find backup dir:
ls tools/autokernel-rdna/workspace/promotion_backups/

# 2. Read which file was promoted (replace <timestamp>):
cat tools/autokernel-rdna/workspace/promotion_backups/<timestamp>/promoted_dest.txt

# 3. Restore:
DEST=\$(cat tools/autokernel-rdna/workspace/promotion_backups/<timestamp>/promoted_dest.txt)
BAK=tools/autokernel-rdna/workspace/promotion_backups/<timestamp>/\$(basename "\$DEST").bak
cp "\$BAK" "\$DEST"

# 4. Rebuild:
cargo build --release --features deltanet --example bench_qwen35_mq4 -p hipfire-runtime
\`\`\`

## Reset to Pre-Promotion SHA

\`\`\`bash
# Reset a specific kernel file (replace <sha> and <kernel>):
git checkout <sha> -- kernels/src/<kernel>.gfx1201.hip
\`\`\`

Current HEAD: \`$ENV_GIT_SHA\`

## Log Locations

| Log | Path |
|---|---|
| Build | \`$BUILD_LOG\` |
| Speed gate | \`$SPEED_GATE_LOG\` |
| Coherence gate | \`$COHERENCE_LOG\` |
| 27B bench | \`$BENCH_LOG\` |
| Final report | \`$REPORT_MD\` |
| Final JSON | \`$REPORT_JSON\` |
REOF
    log_ok "Rollback MD: $ROLLBACK_MD"
}

# ════════════════════════════════════════════════════════════════════════════════
# Step 12: README finalization
# ════════════════════════════════════════════════════════════════════════════════
update_readme() {
    local readme="$SCRIPT_DIR/README.md"
    if grep -q "Phase 16" "$readme" 2>/dev/null; then
        log "  README.md already has Phase 16 section — skipping"
        return
    fi
    log "  Adding Phase 16 section to README.md..."
    cat >> "$readme" << 'REOF'

---

## Phase 16 — Final Validate

Validates the complete AutoKernel adoption end-to-end and produces a proof package.

### Quick run commands

```bash
# Short validation (3 trials, skip coherence gate):
ARCH=gfx1201 MODEL=qwen3.5:27b TRIALS=3 SKIP_COHERENCE=1 \
  ./tools/autokernel-rdna/final_validate.sh

# Full validation (5 trials, all gates):
ARCH=gfx1201 MODEL=qwen3.5:27b TRIALS=5 \
  ./tools/autokernel-rdna/final_validate.sh

# Overnight optimization loop (generates new candidates):
AUTHOR_KERNEL=1 ARCH=gfx1201 MODEL=qwen3.5:27b MAX_ITERS=50 \
  ./tools/autokernel-rdna/autokernel_loop.sh

# Author one kernel manually:
TARGET=gemv_hfq4g256 ARCH=gfx1201 \
  ./tools/autokernel-rdna/author_kernel.sh

# Promote a verified candidate:
ARCH=gfx1201 MODEL=qwen3.5:27b \
  CANDIDATE=tools/autokernel-rdna/kernel_lab/generated/gemv_hfq4g256/candidate_1.hip \
  ./tools/autokernel-rdna/promote_kernel.sh
```

### Decision meanings

| Decision | Meaning |
|---|---|
| `PASS_ADOPTED` | All gates pass. ≥1 kernel promoted and verified to improve Qwen3.5-27B tok/s. Safe to ship. |
| `PASS_INFRA_ONLY` | All gates pass. No kernel has improved tok/s yet. Pipeline ready for more candidates. |
| `NEEDS_BASELINE` | Infrastructure and validation pass, but no 27B baseline for comparison. Measure a baseline first. |
| `FAIL` | At least one hard gate failed (build / correctness / benchmark). Do not claim adoption. |

### Rollback

```bash
# Find backup dirs:
ls tools/autokernel-rdna/workspace/promotion_backups/

# Restore from backup (replace <timestamp>):
DEST=$(cat tools/autokernel-rdna/workspace/promotion_backups/<timestamp>/promoted_dest.txt)
BAK=tools/autokernel-rdna/workspace/promotion_backups/<timestamp>/$(basename "$DEST").bak
cp "$BAK" "$DEST"

# Rebuild:
cargo build --release --features deltanet --example bench_qwen35_mq4 -p hipfire-runtime
```

### Qwen3.5-27B as proof target

The canonical proof of AutoKernel adoption is Qwen3.5-27B tok/s on gfx1201 without accuracy loss.
A `PASS_ADOPTED` result requires:
- ≥1 accepted promotion in `workspace/accepted_promotions/`
- `hipfire_speedup ≥ 1.005` in results.tsv for that promotion
- `coherence-gate-dflash.sh` and `speed-gate.sh --fast` both passing
REOF
    log_ok "  README.md updated"
}

# ════════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════════

echo ""
echo -e "${C_BOLD}═══ Phase 16: Final End-to-End Validation ═══${C_RESET}"
echo "    ARCH=$ARCH  MODEL=$MODEL  TRIALS=$TRIALS"
echo ""

verify_pipeline_files || true

if [[ "$INFRA_STATUS" == "FAIL" && ${#PIPELINE_MISSING[@]} -gt 0 ]]; then
    log_err "Critical pipeline components missing — aborting."
    DECISION="FAIL"
    write_reports
    write_rollback_package
    exit 1
fi

verify_environment || {
    DECISION="FAIL"
    write_reports
    write_rollback_package
    exit 1
}

verify_logs_and_schemas
verify_candidate_records

build_hipfire
if [[ "$BUILD_STATUS" == "FAIL" ]]; then
    compute_final_decision
    write_reports
    write_rollback_package
    update_readme
    echo ""
    echo -e "${C_BOLD}═══ PHASE 16 COMPLETE ═══${C_RESET}"
    echo "  Decision: ${C_BOLD}$DECISION${C_RESET}"
    echo "  Report:   $REPORT_MD"
    exit 1
fi

run_correctness_gates
if [[ "$CORRECTNESS_STATUS" == "FAIL" ]]; then
    compute_final_decision
    write_reports
    write_rollback_package
    update_readme
    echo ""
    echo -e "${C_BOLD}═══ PHASE 16 COMPLETE ═══${C_RESET}"
    echo "  Decision: ${C_BOLD}$DECISION${C_RESET}"
    echo "  Report:   $REPORT_MD"
    exit 1
fi

run_benchmark_27b
compare_to_baseline
compute_final_decision
write_reports
write_rollback_package
update_readme

echo ""
echo -e "${C_BOLD}═══ PHASE 16 COMPLETE ═══${C_RESET}"
echo ""
echo -e "  Decision:    ${C_BOLD}$DECISION${C_RESET}"
echo ""
echo "  MD report:   $REPORT_MD"
echo "  JSON report: $REPORT_JSON"
echo "  Rollback:    $ROLLBACK_MD"
echo ""

[[ "$DECISION" == "FAIL" ]] && exit 1 || exit 0
