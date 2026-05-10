#!/usr/bin/env bash
# check_results_tsv.sh — strict schema and content validator for results.tsv
#
# Usage: ./tools/autokernel-rdna/check_results_tsv.sh [path/to/results.tsv]
#
# Exit codes:
#   0  all checks pass
#   1  one or more checks fail

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TSV="${1:-$SCRIPT_DIR/results.tsv}"

GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}OK${NC}  $*"; }
fail() { echo -e "  ${RED}FAIL${NC}  $*"; FAIL_COUNT=$(( FAIL_COUNT + 1 )); }
warn() { echo -e "  ${YELLOW}WARN${NC}  $*"; }

FAIL_COUNT=0

echo "=== check_results_tsv.sh: $TSV ==="

# ── 1. File exists ────────────────────────────────────────────────────────────
if [[ ! -f "$TSV" ]]; then
    fail "results.tsv not found: $TSV"
    echo -e "${RED}FAIL${NC}: $FAIL_COUNT error(s)"
    exit 1
fi
ok "file exists"

# ── 2. Required columns ───────────────────────────────────────────────────────
REQUIRED_COLS=(
    experiment_id
    timestamp
    git_base_sha
    git_candidate_sha
    arch
    gpu_name
    rocm_version
    model
    quant
    kv_mode
    prompt_name
    prompt_md5
    kernel_name
    kernel_variant_file
    change_summary
    correctness_status
    coherence_status
    build_status
    baseline_decode_tok_s
    candidate_decode_tok_s
    decode_speedup
    baseline_prefill_tok_s
    candidate_prefill_tok_s
    prefill_speedup
    end_to_end_estimated_speedup
    measured_end_to_end_speedup
    median_trials
    stddev
    vram_mb
    decision
    revert_reason
    notes
)

# Phase 13 columns — optional (present only when rows come from autokernel_loop.sh).
OPTIONAL_PHASE13_COLS=(
    loop_iteration
    loop_target
    loop_strategy
    loop_files_changed
    benchmark_status
    current_best_tok_s
    speedup_vs_current_best
)

HEADER="$(head -1 "$TSV")"
for col in "${REQUIRED_COLS[@]}"; do
    if echo "$HEADER" | grep -qw "$col"; then
        ok "column present: $col"
    else
        fail "required column MISSING: $col"
    fi
done

# Phase 13 optional columns — warn if missing (not error), info if present.
for col in "${OPTIONAL_PHASE13_COLS[@]}"; do
    if echo "$HEADER" | grep -qw "$col"; then
        ok "phase13 column present: $col"
    else
        warn "phase13 column absent (ok for pre-Phase-13 rows): $col"
    fi
done

# Phase 14 columns — optional (present only when AUTHOR_KERNEL=1 rows exist).
OPTIONAL_PHASE14_COLS=(
    author_kernel_mode
    target_spec_path
    candidate_kernel_path
    harness_correctness_status
    harness_speedup
    promoted_to_hipfire
    hipfire_tok_s_before
    hipfire_tok_s_after
    final_decision
)
for col in "${OPTIONAL_PHASE14_COLS[@]}"; do
    if echo "$HEADER" | grep -qw "$col"; then
        ok "phase14 column present: $col"
    else
        warn "phase14 column absent (ok for pre-Phase-14 rows): $col"
    fi
done

# Build column index map
declare -A COL_IDX
IFS=$'\t' read -ra COLS <<< "$HEADER"
for i in "${!COLS[@]}"; do
    COL_IDX["${COLS[$i]}"]=$i
done

get_field() {
    local row="$1" col="$2"
    local idx="${COL_IDX[$col]:-}"
    if [[ -z "$idx" ]]; then echo ""; return; fi
    echo "$row" | cut -f$(( idx + 1 ))
}

# ── 3. Row-level checks ───────────────────────────────────────────────────────
ROW_NUM=1
while IFS=$'\t' read -r line; do
    [[ $ROW_NUM -eq 1 ]] && { ROW_NUM=$(( ROW_NUM + 1 )); continue; }
    [[ -z "$line" ]] && { ROW_NUM=$(( ROW_NUM + 1 )); continue; }

    exp_id="$(get_field "$line" experiment_id)"
    decision="$(get_field "$line" decision)"
    correctness="$(get_field "$line" correctness_status)"
    prompt_md5="$(get_field "$line" prompt_md5)"
    model="$(get_field "$line" model)"
    arch="$(get_field "$line" arch)"
    git_base="$(get_field "$line" git_base_sha)"
    baseline_dec="$(get_field "$line" baseline_decode_tok_s)"
    candidate_dec="$(get_field "$line" candidate_decode_tok_s)"
    decode_speedup="$(get_field "$line" decode_speedup)"
    revert_reason="$(get_field "$line" revert_reason)"
    kernel_name="$(get_field "$line" kernel_name)"

    echo "  -- row $ROW_NUM: $exp_id --"

    # decision must not be empty
    if [[ -z "$decision" ]]; then
        fail "[$exp_id] decision is empty"
    else
        ok "[$exp_id] decision=$decision"
    fi

    # speedup fields must be numeric
    for field_name in decode_speedup baseline_decode_tok_s candidate_decode_tok_s; do
        val="$(get_field "$line" "$field_name")"
        if [[ -n "$val" ]] && ! echo "$val" | grep -qP '^\d+(\.\d+)?$'; then
            fail "[$exp_id] $field_name='$val' is not numeric"
        fi
    done

    # prompt_md5 must be present (32 hex chars)
    if echo "$prompt_md5" | grep -qP '^[0-9a-f]{32}$'; then
        ok "[$exp_id] prompt_md5 present"
    else
        fail "[$exp_id] prompt_md5 missing or malformed: '$prompt_md5'"
    fi

    # model, arch, git_base must be non-empty
    for field_name in model arch git_base_sha; do
        val="$(get_field "$line" "$field_name")"
        if [[ -z "$val" ]]; then
            fail "[$exp_id] $field_name is empty"
        fi
    done

    # ACCEPT checks
    if [[ "$decision" == "ACCEPT" || "$decision" == "KEEP" ]]; then
        # must have correctness evidence
        if [[ "$correctness" == "PASS" ]]; then
            ok "[$exp_id] ACCEPTED: correctness_status=PASS"
        else
            fail "[$exp_id] ACCEPTED experiment missing correctness evidence (got '$correctness')"
        fi
        # must have benchmark evidence (candidate_decode_tok_s numeric)
        if echo "$candidate_dec" | grep -qP '^\d+(\.\d+)?$'; then
            ok "[$exp_id] ACCEPTED: candidate_decode_tok_s=$candidate_dec"
        else
            fail "[$exp_id] ACCEPTED experiment missing benchmark evidence (candidate_decode_tok_s='$candidate_dec')"
        fi
    fi

    # REVERT/REJECT checks: must have revert_reason
    if [[ "$decision" == "REVERT" || "$decision" == "REJECT" ]]; then
        if [[ -n "$revert_reason" ]]; then
            ok "[$exp_id] REVERTED: revert_reason present"
        else
            fail "[$exp_id] REVERTED experiment missing revert_reason"
        fi
    fi

    ROW_NUM=$(( ROW_NUM + 1 ))
done < "$TSV"

# ── 4. Final ──────────────────────────────────────────────────────────────────
echo ""
DATA_ROWS=$(( ROW_NUM - 2 ))
echo "  Checked $DATA_ROWS data row(s), $FAIL_COUNT error(s)"
if [[ $FAIL_COUNT -eq 0 ]]; then
    echo -e "${GREEN}=== PASS ===${NC}"
    exit 0
else
    echo -e "${RED}=== FAIL: $FAIL_COUNT error(s) ===${NC}"
    exit 1
fi
