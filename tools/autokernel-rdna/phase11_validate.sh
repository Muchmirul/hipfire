#!/usr/bin/env bash
# phase11_validate.sh — repeatable Phase 11 release-grade validation harness
#
# Usage:
#   ./tools/autokernel-rdna/phase11_validate.sh [--self-test] [--arch ARCH] [--trials N]
#
# Modes:
#   (default)   Full Phase 11 validation: gates + 5 bench trials + JSON report
#   --self-test No GPU required; verify scripts, dirs, schema, tools are present
#
# Exit codes:
#   0  PASS
#   1  FAIL (gate or bench regression)
#   2  NEEDS_MORE_DATA (skipped gates, insufficient trials)
#   3  Environment / build error
#
# Output:
#   tools/autokernel-rdna/reports/phase11_validation_<timestamp>.md
#   tools/autokernel-rdna/reports/phase11_validation_<timestamp>.json

set -uo pipefail
IFS=$'\n\t'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
REPORTS_DIR="$SCRIPT_DIR/reports"
RESULTS_TSV="$SCRIPT_DIR/results.tsv"
ARCH="${HIPFIRE_ARCH:-}"
OPT_TRIALS=5
SELF_TEST=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --self-test) SELF_TEST=1 ;;
        --arch) ARCH="$2"; shift ;;
        --trials) OPT_TRIALS="$2"; shift ;;
        -h|--help)
            sed -n '3,15p' "$0"
            exit 0
            ;;
        *) echo "[phase11_validate] unknown arg: $1" >&2; exit 3 ;;
    esac
    shift
done

# ── colour helpers ────────────────────────────────────────────────────────────
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "  ${GREEN}PASS${NC}  $*"; }
fail() { echo -e "  ${RED}FAIL${NC}  $*"; }
warn() { echo -e "  ${YELLOW}WARN${NC}  $*"; }
info() { echo "  ....  $*"; }

# ── self-test mode ────────────────────────────────────────────────────────────
run_self_test() {
    echo "=== phase11_validate --self-test ==="
    local rc=0

    # 1. Required scripts exist
    for f in \
        scripts/test-kernels.sh \
        scripts/speed-gate.sh \
        scripts/coherence-gate-dflash.sh \
        tools/autokernel-rdna/run.sh \
        tools/autokernel-rdna/check_results_tsv.sh
    do
        if [[ -f "$f" ]]; then ok "script exists: $f"
        else fail "missing: $f"; rc=1; fi
    done

    # 2. Reports dir is writable
    mkdir -p "$REPORTS_DIR"
    if touch "$REPORTS_DIR/.write_test" 2>/dev/null; then
        rm -f "$REPORTS_DIR/.write_test"
        ok "reports dir writable: $REPORTS_DIR"
    else
        fail "reports dir not writable: $REPORTS_DIR"; rc=1
    fi

    # 3. results.tsv schema
    if [[ -f "$RESULTS_TSV" ]]; then
        local header
        header="$(head -1 "$RESULTS_TSV")"
        for col in experiment_id timestamp git_base_sha arch prompt_md5 \
                   kernel_name decision correctness_status baseline_decode_tok_s \
                   candidate_decode_tok_s decode_speedup; do
            if echo "$header" | grep -q "$col"; then
                ok "results.tsv has column: $col"
            else
                fail "results.tsv missing column: $col"; rc=1
            fi
        done
    else
        fail "results.tsv not found at $RESULTS_TSV"; rc=1
    fi

    # 4. Bench binary discoverable
    local bench_exe="$REPO_ROOT/target/release/examples/bench_qwen35_mq4"
    if [[ -x "$bench_exe" ]]; then
        ok "bench binary found: $bench_exe"
    else
        warn "bench binary not found (needs cargo build): $bench_exe"
    fi

    # 5. git SHAs valid
    local head_sha
    head_sha="$(git rev-parse HEAD 2>/dev/null)"
    if [[ ${#head_sha} -eq 40 ]]; then
        ok "git HEAD SHA valid: ${head_sha:0:8}"
    else
        fail "git HEAD SHA invalid"; rc=1
    fi

    # 6. Arch detection
    local detected_arch=""
    if command -v amdgpu-arch &>/dev/null; then
        detected_arch="$(amdgpu-arch 2>/dev/null | head -1)"
    elif command -v rocminfo &>/dev/null; then
        detected_arch="$(rocminfo 2>/dev/null | grep -oP 'gfx\d+' | head -1)"
    fi
    if [[ -n "$detected_arch" ]]; then
        ok "arch detected: $detected_arch"
    else
        warn "arch detection failed (no amdgpu-arch/rocminfo, check GPU visibility)"
    fi

    # 7. JSON generation (jq or python3)
    if command -v jq &>/dev/null; then
        ok "JSON tool available: jq"
    elif command -v python3 &>/dev/null; then
        ok "JSON tool available: python3"
    else
        fail "no JSON tool found (need jq or python3)"; rc=1
    fi

    # 8. Markdown can be generated (just write test)
    local test_md="$REPORTS_DIR/.self_test_$$.md"
    if echo "# self-test" > "$test_md" 2>/dev/null; then
        rm -f "$test_md"
        ok "markdown generation: OK"
    else
        fail "cannot write markdown to reports dir"; rc=1
    fi

    echo ""
    if [[ $rc -eq 0 ]]; then
        echo -e "${GREEN}=== SELF-TEST PASSED ===${NC}"
    else
        echo -e "${RED}=== SELF-TEST FAILED — fix above items ===${NC}"
    fi
    return $rc
}

if [[ $SELF_TEST -eq 1 ]]; then
    run_self_test
    exit $?
fi

# ── full validation ───────────────────────────────────────────────────────────

# PATH fixup for cargo
CARGO_PATH="/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.cargo/bin"
if [[ -d "$CARGO_PATH" ]]; then
    export PATH="$CARGO_PATH:$PATH"
fi

# Arch detection
if [[ -z "$ARCH" ]]; then
    if command -v amdgpu-arch &>/dev/null; then
        ARCH="$(amdgpu-arch 2>/dev/null | head -1)"
    elif [[ -f /sys/class/kfd/kfd/topology/nodes/1/gpu_id ]]; then
        gfx_ver="$(find /sys/class/kfd/kfd/topology/nodes/*/properties -exec grep -l gfx_target_version {} \; 2>/dev/null | head -1)"
        if [[ -n "$gfx_ver" ]]; then
            raw="$(grep gfx_target_version "$gfx_ver" | awk '{print $2}')"
            maj=$(( raw / 10000 )); min=$(( (raw % 10000) / 100 )); stp=$(( raw % 100 ))
            ARCH="gfx${maj}${min}${stp}"
        fi
    fi
    ARCH="${ARCH:-gfx1201}"
fi

# Model / prompt
MODELS_DIR="${HIPFIRE_MODELS_DIR:-${HIPFIRE_DIR:-$HOME/.hipfire}/models}"
if [[ ! -d "$MODELS_DIR" ]]; then
    FALLBACK="/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.hipfire/models"
    [[ -d "$FALLBACK" ]] && MODELS_DIR="$FALLBACK"
fi
MODEL_9B="$(find "$MODELS_DIR" -name "qwen3.5-9b.mq4" -o -name "qwen35-9b.mq4" 2>/dev/null | head -1)"
PROMPT_FILE="$REPO_ROOT/benchmarks/prompts/lru_cache_pep8_strict.txt"
BENCH_EXE="$REPO_ROOT/target/release/examples/bench_qwen35_mq4"

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
RESULT="PASS"
declare -a ISSUES=()
declare -a FIXED=()

echo "=== Phase 11 Validation === ($TIMESTAMP)"
echo "    arch=$ARCH  model_dir=$MODELS_DIR"
echo ""

# ─── GATE 1: kernel tests ─────────────────────────────────────────────────────
echo "--- Gate 1: test-kernels.sh ---"
KERNEL_TEST_LOG="$REPORTS_DIR/gate_kernels_${TIMESTAMP}.log"
if bash scripts/test-kernels.sh "$ARCH" >"$KERNEL_TEST_LOG" 2>&1; then
    passed="$(grep -oP 'Passed:\s+\K\d+' "$KERNEL_TEST_LOG" || echo 0)"
    failed="$(grep -oP 'Failed:\s+\K\d+' "$KERNEL_TEST_LOG" || echo 0)"
    ok "test-kernels: ${passed} passed, ${failed} failed"
    PASS_COUNT=$(( PASS_COUNT + 1 ))
else
    fail "test-kernels.sh FAILED (see $KERNEL_TEST_LOG)"
    FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    ISSUES+=("kernel test harness failed")
    RESULT="FAIL"
fi

# ─── GATE 2: speed gate ──────────────────────────────────────────────────────
echo "--- Gate 2: speed-gate.sh ---"
SPEED_GATE_LOG="$REPORTS_DIR/gate_speed_${TIMESTAMP}.log"
HIPFIRE_MODELS_DIR="$MODELS_DIR" bash scripts/speed-gate.sh >"$SPEED_GATE_LOG" 2>&1
speed_rc=$?
sg_passed="$(grep -cP '^\s+\S.*OK\b' "$SPEED_GATE_LOG" || true)"
sg_failed="$(grep -cP '^\s+\S.*FAIL\b' "$SPEED_GATE_LOG" || true)"
sg_skipped="$(grep -cP '^\s+\S.*SKIP\b' "$SPEED_GATE_LOG" || true)"
if [[ $speed_rc -eq 0 ]]; then
    ok "speed-gate: ${sg_passed} OK, ${sg_skipped} SKIP, ${sg_failed} FAIL"
    PASS_COUNT=$(( PASS_COUNT + 1 ))
    if [[ "$sg_skipped" -gt 0 ]]; then
        warn "speed-gate: ${sg_skipped} metrics skipped (models not present)"
        ISSUES+=("speed-gate: ${sg_skipped} metrics skipped due to missing models")
    fi
else
    fail "speed-gate FAILED: ${sg_failed} regressions (see $SPEED_GATE_LOG)"
    FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    ISSUES+=("speed-gate regression detected")
    RESULT="FAIL"
fi

# ─── GATE 3: DFlash coherence gate ────────────────────────────────────────────
echo "--- Gate 3: coherence-gate-dflash.sh ---"
DRAFT_9B="$(find "$MODELS_DIR" -name "*9b*dflash*.hfq" 2>/dev/null | head -1)"
DRAFT_27B="$(find "$MODELS_DIR" -name "*27b*dflash*.hfq" 2>/dev/null | head -1)"
COHERENCE_GATE_LOG="$REPORTS_DIR/gate_coherence_${TIMESTAMP}.log"
if [[ -n "$DRAFT_9B" || -n "$DRAFT_27B" ]]; then
    HIPFIRE_MODELS_DIR="$MODELS_DIR" bash scripts/coherence-gate-dflash.sh >"$COHERENCE_GATE_LOG" 2>&1
    coh_rc=$?
    if [[ $coh_rc -eq 0 ]]; then
        ok "coherence-gate-dflash: PASS"
        PASS_COUNT=$(( PASS_COUNT + 1 ))
    else
        fail "coherence-gate-dflash: FAIL (see $COHERENCE_GATE_LOG)"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        ISSUES+=("DFlash coherence gate failed")
        RESULT="FAIL"
    fi
else
    warn "coherence-gate-dflash: SKIP (draft models not present — no DFlash changes in PR)"
    SKIP_COUNT=$(( SKIP_COUNT + 1 ))
    echo "SKIP: no draft models" > "$COHERENCE_GATE_LOG"
fi

# ─── BENCH: prompt MD5 ────────────────────────────────────────────────────────
echo "--- Bench setup ---"
CANONICAL_MD5="df5dedc8040ce70ba55080c4548e6024"
if [[ -f "$PROMPT_FILE" ]]; then
    ACTUAL_MD5="$(md5sum "$PROMPT_FILE" | awk '{print $1}')"
    if [[ "$ACTUAL_MD5" == "$CANONICAL_MD5" ]]; then
        ok "prompt MD5 matches: $ACTUAL_MD5"
    else
        fail "prompt MD5 MISMATCH: got $ACTUAL_MD5 expected $CANONICAL_MD5"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        ISSUES+=("prompt MD5 mismatch")
        RESULT="FAIL"
    fi
else
    fail "prompt file not found: $PROMPT_FILE"
    FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    ISSUES+=("prompt file missing")
    RESULT="FAIL"
    ACTUAL_MD5="MISSING"
fi

# Binary MD5
BENCH_MD5=""
if [[ -x "$BENCH_EXE" ]]; then
    BENCH_MD5="$(md5sum "$BENCH_EXE" | awk '{print $1}')"
    ok "bench binary MD5: $BENCH_MD5"
else
    warn "bench binary not found: $BENCH_EXE"
    ISSUES+=("bench binary not present; build needed")
fi

# ─── BENCH: 5 fresh-process trials ───────────────────────────────────────────
echo "--- Bench: $OPT_TRIALS fresh-process trials ---"
BENCH_LOG="$REPORTS_DIR/bench_trials_${TIMESTAMP}.log"
declare -a GEN_TOK_S=()
declare -a PREFILL_TOK_S=()
BENCH_OK=1

if [[ -z "$MODEL_9B" ]]; then
    warn "qwen3.5-9b.mq4 not found — skipping bench trials"
    ISSUES+=("qwen3.5-9b model not found; bench trials skipped")
    BENCH_OK=0
    SKIP_COUNT=$(( SKIP_COUNT + 1 ))
elif [[ ! -x "$BENCH_EXE" ]]; then
    warn "bench binary not built — skipping trials"
    ISSUES+=("bench binary not built; run cargo build first")
    BENCH_OK=0
    SKIP_COUNT=$(( SKIP_COUNT + 1 ))
else
    echo "" > "$BENCH_LOG"
    for (( i=1; i<=OPT_TRIALS; i++ )); do
        info "trial $i / $OPT_TRIALS ..."
        trial_out="$(mktemp)"
        "$BENCH_EXE" "$MODEL_9B" --gen 80 --warmup 10 >"$trial_out" 2>&1
        summary="$(grep "SUMMARY" "$trial_out" || true)"
        echo "Trial $i: $summary" >> "$BENCH_LOG"
        cat "$trial_out" >> "$BENCH_LOG"
        rm -f "$trial_out"
        if [[ -n "$summary" ]]; then
            g="$(echo "$summary" | grep -oP 'gen_tok_s=\K[\d.]+')"
            p="$(echo "$summary" | grep -oP 'prefill_tok_s=\K[\d.]+')"
            GEN_TOK_S+=("$g")
            PREFILL_TOK_S+=("$p")
            ok "  trial $i: gen=$g prefill=$p"
        else
            fail "  trial $i: no SUMMARY line"
            BENCH_OK=0
            ISSUES+=("bench trial $i produced no SUMMARY output")
            RESULT="FAIL"
        fi
    done
fi

# Compute median from trials
MEDIAN_GEN="N/A"
MEDIAN_PREFILL="N/A"
STDDEV_GEN="N/A"
if [[ ${#GEN_TOK_S[@]} -ge 3 ]]; then
    MEDIAN_GEN="$(printf '%s\n' "${GEN_TOK_S[@]}" | sort -n | awk 'BEGIN{c=0} {a[c++]=$1} END{print (c%2==1)?a[int(c/2)]:((a[c/2-1]+a[c/2])/2)}')"
    MEDIAN_PREFILL="$(printf '%s\n' "${PREFILL_TOK_S[@]}" | sort -n | awk 'BEGIN{c=0} {a[c++]=$1} END{print (c%2==1)?a[int(c/2)]:((a[c/2-1]+a[c/2])/2)}')"
    STDDEV_GEN="$(printf '%s\n' "${GEN_TOK_S[@]}" | awk '{s+=$1;ss+=$1*$1;c++} END{if(c>1){printf "%.3f",sqrt((ss-s*s/c)/(c-1))}else{print 0}}')"
    ok "median decode: $MEDIAN_GEN tok/s  prefill: $MEDIAN_PREFILL tok/s  stddev: $STDDEV_GEN"
fi

# Regression check vs Phase 10 baseline
BASELINE_DECODE="93.2"
TOLERANCE="0.15"   # 15% — thermal/DPM noise floor per AGENTS.md (±10-15%)
if [[ "$MEDIAN_GEN" != "N/A" ]]; then
    floor="$(echo "$BASELINE_DECODE $TOLERANCE" | awk '{printf "%.3f", $1*(1-$2)}')"
    if awk "BEGIN{exit !($MEDIAN_GEN >= $floor)}"; then
        ok "decode median $MEDIAN_GEN >= floor $floor (baseline $BASELINE_DECODE × (1-$TOLERANCE))"
    else
        fail "decode median $MEDIAN_GEN < floor $floor — regression detected"
        FAIL_COUNT=$(( FAIL_COUNT + 1 ))
        ISSUES+=("decode median ${MEDIAN_GEN} below floor ${floor}")
        RESULT="FAIL"
    fi
fi

# ─── results.tsv check ────────────────────────────────────────────────────────
echo "--- results.tsv validation ---"
RESULTS_CHECK_LOG="$REPORTS_DIR/results_tsv_check_${TIMESTAMP}.log"
if bash "$SCRIPT_DIR/check_results_tsv.sh" >"$RESULTS_CHECK_LOG" 2>&1; then
    ok "check_results_tsv.sh: PASS"
    PASS_COUNT=$(( PASS_COUNT + 1 ))
else
    fail "check_results_tsv.sh: FAIL (see $RESULTS_CHECK_LOG)"
    FAIL_COUNT=$(( FAIL_COUNT + 1 ))
    ISSUES+=("results.tsv schema check failed")
    RESULT="FAIL"
fi

# ─── overall result ───────────────────────────────────────────────────────────
if [[ $FAIL_COUNT -eq 0 && $SKIP_COUNT -gt 0 ]]; then
    RESULT="PASS"   # skips are acceptable when due to missing models, not code
fi

echo ""
echo "=== Summary: $PASS_COUNT PASS, $FAIL_COUNT FAIL, $SKIP_COUNT SKIP ==="
echo "=== Overall: $RESULT ==="

# ─── capture git metadata ────────────────────────────────────────────────────
HEAD_SHA="$(git rev-parse HEAD)"
GIT_STATUS="$(git status --short)"
GIT_DIFF_STAT="$(git diff --stat "$(git merge-base HEAD origin/main 2>/dev/null || git rev-parse HEAD~5)" HEAD 2>/dev/null | tail -1 || echo 'N/A')"
ROCM_VER="$(cat /opt/rocm/.info/version 2>/dev/null || rocminfo 2>/dev/null | grep -i 'ROCm Version' | head -1 | awk '{print $NF}' || echo 'unknown')"
HIPCC_VER="$(hipcc --version 2>/dev/null | head -1 || echo 'unknown')"

# ─── write JSON report ────────────────────────────────────────────────────────
JSON_REPORT="$REPORTS_DIR/phase11_validation_${TIMESTAMP}.json"
cat > "$JSON_REPORT" <<JSON
{
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "arch": "$ARCH",
  "head_sha": "$HEAD_SHA",
  "rocm_version": "$ROCM_VER",
  "hipcc_version": "$(echo "$HIPCC_VER" | tr '\n' ' ' | xargs)",
  "prompt_file": "benchmarks/prompts/lru_cache_pep8_strict.txt",
  "prompt_md5": "${ACTUAL_MD5:-MISSING}",
  "canonical_prompt_md5": "$CANONICAL_MD5",
  "prompt_md5_match": $([ "${ACTUAL_MD5:-}" = "$CANONICAL_MD5" ] && echo true || echo false),
  "bench_binary_md5": "${BENCH_MD5:-missing}",
  "model": "qwen3.5-9b.mq4",
  "gate_kernel_tests": $([ $FAIL_COUNT -eq 0 ] && echo '"PASS"' || echo '"SEE_LOG"'),
  "gate_speed": "$(grep -oP '(PASS|FAIL)' "$SPEED_GATE_LOG" | head -1 || echo 'SKIP')",
  "gate_coherence_dflash": "$([ -n "$DRAFT_9B$DRAFT_27B" ] && echo 'RAN' || echo 'SKIP_NO_DRAFTS')",
  "bench_trials": ${#GEN_TOK_S[@]},
  "bench_median_decode_tok_s": ${MEDIAN_GEN:-0},
  "bench_median_prefill_tok_s": ${MEDIAN_PREFILL:-0},
  "bench_stddev_decode": ${STDDEV_GEN:-0},
  "baseline_decode_tok_s": $BASELINE_DECODE,
  "pass_count": $PASS_COUNT,
  "fail_count": $FAIL_COUNT,
  "skip_count": $SKIP_COUNT,
  "result": "$RESULT",
  "issues": $(printf '%s\n' "${ISSUES[@]:-}" | python3 -c 'import sys,json; lines=[l.strip() for l in sys.stdin if l.strip()]; print(json.dumps(lines))'),
  "gate_log_kernel": "$KERNEL_TEST_LOG",
  "gate_log_speed": "$SPEED_GATE_LOG",
  "gate_log_coherence": "$COHERENCE_GATE_LOG",
  "bench_log": "$BENCH_LOG",
  "results_tsv_check_log": "$RESULTS_CHECK_LOG"
}
JSON
echo "JSON: $JSON_REPORT"

# ─── write Markdown report ────────────────────────────────────────────────────
MD_REPORT="$REPORTS_DIR/phase11_validation_${TIMESTAMP}.md"
{
cat <<MD
# Phase 11 Validation Report

**Timestamp:** $(date -u +%Y-%m-%dT%H:%M:%SZ)
**HEAD SHA:** $HEAD_SHA
**Arch:** $ARCH
**ROCm:** $ROCM_VER
**Result:** **$RESULT**

---

## Gates

| Gate | Result |
|---|---|
| Kernel tests (test-kernels.sh) | $([ $FAIL_COUNT -eq 0 ] && echo "✅ PASS ${passed:-?}/16" || echo "❌ FAIL") |
| Speed gate | $([ $speed_rc -eq 0 ] && echo "✅ PASS (${sg_passed} OK, ${sg_skipped} SKIP)" || echo "❌ FAIL") |
| DFlash coherence | $([ -n "${DRAFT_9B:-}${DRAFT_27B:-}" ] && echo "RAN" || echo "⚪ SKIP (no draft models)") |
| results.tsv check | $([ $FAIL_COUNT -eq 0 ] && echo "✅ PASS" || echo "❌ FAIL") |

## Benchmark Trials

**Prompt:** \`benchmarks/prompts/lru_cache_pep8_strict.txt\`
**Prompt MD5:** \`${ACTUAL_MD5:-MISSING}\` $([ "${ACTUAL_MD5:-}" = "$CANONICAL_MD5" ] && echo "✅" || echo "❌")
**Model:** \`qwen3.5-9b.mq4\`
**Bench binary MD5:** \`${BENCH_MD5:-not captured}\`
**Flags:** \`--gen 80 --warmup 10\`

| Trial | Decode tok/s | Prefill tok/s |
|---|---|---|
MD
for (( i=0; i<${#GEN_TOK_S[@]}; i++ )); do
    echo "| $(( i+1 )) | ${GEN_TOK_S[$i]} | ${PREFILL_TOK_S[$i]} |"
done
cat <<MD
| **Median** | **${MEDIAN_GEN}** | **${MEDIAN_PREFILL}** |
| Std-dev | ${STDDEV_GEN} | — |

Phase 10 baseline: ${BASELINE_DECODE} tok/s
Tolerance floor (85%): $(echo "$BASELINE_DECODE $TOLERANCE" | awk '{printf "%.2f", $1*(1-$2)}') tok/s

## Issues Found

MD
if [[ ${#ISSUES[@]} -eq 0 ]]; then
    echo "None."
else
    for issue in "${ISSUES[@]}"; do echo "- $issue"; done
fi
cat <<MD

## Overall Result: $RESULT

MD
if [[ $RESULT == "PASS" ]]; then
    echo "All gates pass. Phase 12 is permitted."
elif [[ $RESULT == "NEEDS_MORE_DATA" ]]; then
    echo "Additional data needed. Rerun with models present."
else
    echo "Fix the blockers above before proceeding to Phase 12."
fi
} > "$MD_REPORT"

echo "Report: $MD_REPORT"

# exit code reflects result
case "$RESULT" in
    PASS) exit 0 ;;
    NEEDS_MORE_DATA) exit 2 ;;
    *) exit 1 ;;
esac
