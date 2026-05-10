#!/usr/bin/env bash
# tools/autokernel-rdna/author_kernel.sh
#
# Phase 14 — Real AutoKernel Kernel-Authoring Mode
#
# Called by autokernel_loop.sh when AUTHOR_KERNEL=1, or standalone.
#
# Responsibilities:
#   1. extract_target_spec(target)        — emit a filled target_spec.json from source
#   2. generate_candidate_skeleton(target, N) — create candidate_N.hip from reference
#   3. run_harness(target, candidate)     — compile + evaluate candidate in isolation
#   4. promote_to_hipfire(target, candidate, arch) — copy to kernels/src/ and rebuild
#   5. author_kernel_iteration(target, iter, strategy) — orchestrate one full cycle
#   6. write_phase14_report(output_dir)   — emit markdown + JSON summary
#
# Design principle: This file is OFFLINE TOOLING only.
# No Python in the hot path. This script exists outside the runtime.
# The harnesses are the fixed evaluators; only the candidate .hip file changes.
#
# Usage:
#   source tools/autokernel-rdna/author_kernel.sh
#   author_kernel_iteration gemv_hfq4g256 1 author_gemv_lds_staging
#
# Or run a single cycle manually:
#   ARCH=gfx1201 TARGET=gemv_hfq4g256 ITER=1 STRATEGY=author_gemv_lds_staging \
#     bash tools/autokernel-rdna/author_kernel.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOL_DIR="$SCRIPT_DIR"
LAB_DIR="$TOOL_DIR/kernel_lab"

# ── Runtime config (overridable) ────────────────────────────────────────────
ARCH="${ARCH:-gfx1201}"
ACCEPT_MIN_SPEEDUP="${ACCEPT_MIN_SPEEDUP:-1.005}"
HARNESS_TRIALS="${HARNESS_TRIALS:-5}"
TOL_ABS="${TOL_ABS:-1e-3}"
VERBOSE="${VERBOSE:-0}"

# ── Colour helpers (duplicated from autokernel_loop.sh for standalone use) ───
if [[ -t 1 ]]; then
    C_GREEN='\033[0;32m'; C_YELLOW='\033[0;33m'; C_RED='\033[0;31m'
    C_CYAN='\033[0;36m'; C_BOLD='\033[1m'; C_RESET='\033[0m'
else
    C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''; C_BOLD=''; C_RESET=''
fi

log_info()  { echo -e "${C_CYAN}[author_kernel]${C_RESET} $*" >&2; }
log_ok()    { echo -e "${C_GREEN}[author_kernel]${C_RESET} $*" >&2; }
log_warn()  { echo -e "${C_YELLOW}[author_kernel]${C_RESET} $*" >&2; }
log_error() { echo -e "${C_RED}[author_kernel]${C_RESET} $*" >&2; }

# ── Available authoring strategies ─────────────────────────────────────────
# Each strategy corresponds to a code transformation an LLM agent should apply.
# Strategies are ordered from most to least likely to help on gfx1201 GDDR6.
declare -a AUTHOR_STRATEGIES_gemv_hfq4g256=(
    "author_gemv_lds_staging_x"
    "author_gemv_uint4_loads"
    "author_gemv_8acc_ilp"
    "author_gemv_scale_broadcast"
    "author_gemv_prefetch_next_group"
    "author_gemv_wave32_vector_mac"
)

declare -a AUTHOR_STRATEGIES_gemv_hfq6g256=(
    "author_gemv6_lds_staging_x"
    "author_gemv6_uint4_loads"
    "author_gemv6_4acc_ilp"
)

declare -a AUTHOR_STRATEGIES_gemv_hfq4g256_residual=(
    "author_gemv_residual_lds_staging_x"
    "author_gemv_residual_uint4_loads"
    "author_gemv_residual_8acc_ilp"
    "author_gemv_residual_lds_staging_r"
)

# ── Strategy descriptions (written into generated candidate header) ──────────
strategy_description() {
    local strategy="$1"
    case "$strategy" in
        author_gemv_lds_staging_x)
            echo "Stage the x[] activations into LDS (shared memory) per group before computing dot products. Each group uses 256*4=1024 bytes of LDS. All 32 threads cooperatively load x[g*256..(g+1)*256], then read from LDS in the inner loop. This reduces global reads of x[] from O(groups_per_row * 32) thread-accesses to O(groups_per_row * 32) total (cooperative load), which for 28+ groups should show a meaningful BW saving on gfx1201 GDDR6." ;;
        author_gemv_uint4_loads)
            echo "Replace the 4-byte (uint32) weight load 'pk = *(uint32*)(gp+8+boff)' with 16-byte vectorized loads (uint4). Process 4x uint32 at once, yielding 32 nibbles (32 weights) per load instead of 8. The group payload starts at byte 8 of each 136-byte group; each thread has boff = tid*4, so alignment within the group is: byte 8+boff, which is 4-byte aligned. For uint4 (16-byte), boff must be multiple of 16 => tid % 4 == 0 requirement. Partition 32 threads into 4 groups of 8, each loading 4 consecutive uint32s as uint4." ;;
        author_gemv_8acc_ilp)
            echo "Extend from 4 accumulators (acc0..acc3) to 8 (acc0..acc7) to expose more ILP to the gfx1201 scheduler. Process 8 groups per outer iteration instead of 4. The 4-accumulator combine at the end (warp shuffle sum) extends to 8 lanes. WARNING: preserve the tail group accumulator invariant: tail group g goes into acc[g % 8]." ;;
        author_gemv_scale_broadcast)
            echo "In the current kernel each thread loads its own scale/zp from global memory (bytes 0-7 of the group). Since all 32 threads in the block process the same group and need the same scale/zp, use __shfl_sync / DS_BPERMUTE to load once per block (thread 0 loads) then broadcast. Saves 31 redundant global loads per group per block." ;;
        author_gemv_prefetch_next_group)
            echo "While computing dot product for group g, issue a prefetch hint for group g+1. Use __builtin_amdgcn_s_prefetch or __builtin_prefetch on the next group pointer. WARNING: Phase 10 showed this regresses on gfx1201 in the residual path — re-evaluate only for the base GEMV path with careful measurement. Do not use for _residual kernels." ;;
        author_gemv_wave32_vector_mac)
            echo "Replace the scalar FMA chain with vectorized 4-wide (float4) MAC operations using __builtin_amdgcn_fmaf or explicit SIMD intrinsics. Dequantize 4 weights at once into a float4, load 4 activations as float4, use dot4 or fmaf4. Requires regrouping the 8-weight-per-uint32 DOG macro into pairs of 4." ;;
        author_gemv_residual_lds_staging_x)
            echo "LDS staging of x[] activations (same as author_gemv_lds_staging_x but for the _residual variant). Additionally consider staging residual[] in LDS since it is also a per-row global load." ;;
        author_gemv_residual_lds_staging_r)
            echo "Stage the residual[] vector in LDS. Since residual[row] for the current block is a single float32, this may not save BW on its own, but combining with x[] staging can improve overall occupancy by sharing LDS budget across both vectors." ;;
        *)
            echo "Custom strategy: $strategy. See kernel_lab/templates/hip_kernel_candidate.md for guidance." ;;
    esac
}

# ── extract_target_spec ──────────────────────────────────────────────────────
# Writes a filled target_spec.json to kernel_lab/generated/<target>/target_spec.json
# (if not already present).
extract_target_spec() {
    local target="$1"
    local spec_dir="$LAB_DIR/generated/$target"
    local spec_file="$spec_dir/target_spec.json"

    mkdir -p "$spec_dir"

    if [ -f "$spec_file" ]; then
        log_info "target_spec.json already exists for $target — skipping extraction"
        return 0
    fi

    log_info "Extracting target spec for $target..."
    local src_file="$REPO_ROOT/kernels/src/${target}.hip"
    if [ ! -f "$src_file" ]; then
        log_error "Reference kernel not found: $src_file"
        return 1
    fi

    # Extract signature line from source
    local sig
    sig=$(grep -E '^extern "C" __global__' "$src_file" | head -1 || echo "not found")

    # Create minimal spec (to be overridden by pre-seeded specs in generated/)
    cat > "$spec_file" <<EOF
{
  "_autogenerated": true,
  "_note": "Auto-extracted by author_kernel.sh. Pre-seeded specs in kernel_lab/generated/ take precedence.",
  "kernel_name": "$target",
  "source_file": "kernels/src/${target}.hip",
  "gfx1201_variant": null,
  "arch": "$ARCH",
  "extracted_signature": "$sig",
  "harness": "tools/autokernel-rdna/kernel_lab/harnesses/${target}_harness.sh",
  "candidate_dir": "tools/autokernel-rdna/kernel_lab/generated/$target/",
  "promotion_target": "kernels/src/${target}.gfx1201.hip",
  "correctness_tolerance": 1e-3
}
EOF
    log_ok "  Written: $spec_file"
}

# ── generate_candidate_skeleton ────────────────────────────────────────────
# Creates candidate_N.hip from the reference kernel with an authoring header.
generate_candidate_skeleton() {
    local target="$1"
    local iter="$2"
    local strategy="$3"
    local cand_dir="$LAB_DIR/generated/$target"
    local cand_file="$cand_dir/candidate_${iter}.hip"
    local ref_file="$REPO_ROOT/kernels/src/${target}.hip"
    local gfx_variant="$REPO_ROOT/kernels/src/${target}.gfx1201.hip"

    mkdir -p "$cand_dir"

    # If a gfx1201-specific variant exists, use that as the base to improve on
    local base_file
    if [ -f "$gfx_variant" ]; then
        base_file="$gfx_variant"
        log_info "  Using gfx1201 variant as base: $gfx_variant"
    else
        base_file="$ref_file"
        log_info "  Using generic kernel as base: $ref_file"
    fi

    if [ ! -f "$base_file" ]; then
        log_error "Base kernel not found: $base_file"
        return 1
    fi

    local desc
    desc=$(strategy_description "$strategy")
    local ts
    ts=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

    # Write candidate with authoring header
    {
        cat <<HDR
// ============================================================
// AUTOKERNEL CANDIDATE — GENERATED BY author_kernel.sh
// ============================================================
//
// Target:    $target
// Strategy:  $strategy
// Iteration: $iter
// Arch:      $ARCH
// Timestamp: $ts
//
// OBJECTIVE:
// $desc
//
// RULES (do not violate or correctness gate will fail):
//   1. Function name must remain: $target
//   2. Argument list must remain identical (see signature below)
//   3. Dequant formula: w_float = scale * (float)(nibble) + zero_point
//   4. Group layout (HFQ4-G256): 4B scale + 4B zp + 128B packed nibbles = 136B/group
//   5. groups_per_row = K / 256
//   6. Output y[row] = sum over all groups and all weights in group
//   7. Tail accumulator invariant: tail group g must use acc[g % N_ACCUMULATORS]
//
// HOW TO TEST:
//   CANDIDATE=$(pwd)/$cand_file \\
//     bash tools/autokernel-rdna/kernel_lab/harnesses/${target}_harness.sh
//
// HOW TO REVERT:
//   git checkout kernels/src/${target}.gfx1201.hip  # or delete if newly created
//
// ============================================================
// BASE SOURCE (reference implementation):
// File: $base_file
// ============================================================

HDR
        cat "$base_file"
    } > "$cand_file"

    log_ok "  Generated candidate: $cand_file"
    echo "$cand_file"
}

# ── run_harness ─────────────────────────────────────────────────────────────
# Returns:
#   HARNESS_CORRECTNESS=PASS|FAIL
#   HARNESS_LATENCY_US=<float>
#   HARNESS_SPEEDUP=<float>
run_harness() {
    local target="$1"
    local candidate="$2"
    local harness_script="$TOOL_DIR/kernel_lab/harnesses/${target}_harness.sh"

    HARNESS_CORRECTNESS="FAIL"
    HARNESS_LATENCY_US="0"
    HARNESS_SPEEDUP="0"
    HARNESS_ERROR=""

    if [ ! -f "$harness_script" ]; then
        HARNESS_ERROR="Harness script not found: $harness_script"
        log_error "$HARNESS_ERROR"
        return 1
    fi

    log_info "  Running harness: $harness_script"
    log_info "  Candidate: $candidate"

    local harness_out
    harness_out=$(ARCH="$ARCH" CANDIDATE="$candidate" \
        HARNESS_TRIALS="$HARNESS_TRIALS" \
        TOL_ABS="$TOL_ABS" \
        bash "$harness_script" 2>&1)
    local harness_exit=$?

    if [ $VERBOSE -eq 1 ]; then
        echo "$harness_out"
    fi

    HARNESS_CORRECTNESS=$(echo "$harness_out" | grep '^CORRECTNESS:' | awk '{print $2}')
    HARNESS_LATENCY_US=$(echo "$harness_out" | grep '^LATENCY_US:' | awk '{print $2}')
    HARNESS_SPEEDUP=$(echo "$harness_out" | grep '^SPEEDUP_VS_REF:' | awk '{print $2}')
    HARNESS_ERROR=$(echo "$harness_out" | grep '^ERROR:' | sed 's/^ERROR: //')

    if [ -z "$HARNESS_CORRECTNESS" ]; then
        HARNESS_CORRECTNESS="FAIL"
        HARNESS_ERROR="Harness produced no CORRECTNESS line (exit $harness_exit)"
    fi
    if [ -z "$HARNESS_SPEEDUP" ]; then HARNESS_SPEEDUP="0"; fi
    if [ -z "$HARNESS_LATENCY_US" ]; then HARNESS_LATENCY_US="0"; fi

    log_info "  Correctness: $HARNESS_CORRECTNESS"
    if [ "$HARNESS_CORRECTNESS" == "PASS" ]; then
        log_info "  Latency: ${HARNESS_LATENCY_US} µs | Speedup: ${HARNESS_SPEEDUP}x"
    else
        log_warn "  Error: $HARNESS_ERROR"
    fi
}

# ── promote_to_hipfire ───────────────────────────────────────────────────────
# Copy a passing candidate to kernels/src/<target>.gfx1201.hip and rebuild.
# Returns: PROMOTE_STATUS=OK|FAIL
promote_to_hipfire() {
    local target="$1"
    local candidate="$2"
    local arch="${3:-$ARCH}"
    local dest="$REPO_ROOT/kernels/src/${target}.gfx1201.hip"

    PROMOTE_STATUS="FAIL"
    PROMOTE_BUILD_LOG=""

    log_info "Promoting candidate to hipfire: $dest"
    cp "$candidate" "$dest"

    # Rebuild hipfire
    local cargo_bin
    cargo_bin=$(which cargo 2>/dev/null || echo "/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.cargo/bin/cargo")

    local build_log="$LAB_DIR/reports/_promote_build_$(date +%s).log"
    PROMOTE_BUILD_LOG="$build_log"

    log_info "  Rebuilding hipfire (this may take 1-3 minutes)..."
    pushd "$REPO_ROOT" >/dev/null
    CARGO_INCREMENTAL=1 "$cargo_bin" build --release --example bench_qwen35_mq4 \
        >"$build_log" 2>&1
    local build_exit=$?
    popd >/dev/null

    if [ $build_exit -ne 0 ]; then
        log_error "  Build failed — see $build_log"
        log_info "  Rolling back: git checkout $dest"
        git -C "$REPO_ROOT" checkout -- "kernels/src/${target}.gfx1201.hip" 2>/dev/null || \
            rm -f "$dest"
        return 1
    fi

    PROMOTE_STATUS="OK"
    log_ok "  Build succeeded"
}

# ── author_kernel_iteration ─────────────────────────────────────────────────
# Full cycle: extract spec → generate skeleton → run harness → promote if good
# → hipfire bench → accept/reject.
#
# Returns (env vars set for caller):
#   AK_CORRECTNESS    PASS|FAIL
#   AK_HARNESS_SPEEDUP  float
#   AK_PROMOTED       0|1
#   AK_HIPFIRE_TOK_S  float (0 if not promoted)
#   AK_DECISION       ACCEPT|REJECT|SKIP
#   AK_CANDIDATE_PATH path to candidate file
author_kernel_iteration() {
    local target="${1:-gemv_hfq4g256}"
    local iter="${2:-1}"
    local strategy="${3:-author_gemv_lds_staging_x}"

    AK_CORRECTNESS="FAIL"
    AK_HARNESS_SPEEDUP="0"
    AK_PROMOTED=0
    AK_HIPFIRE_TOK_S="0"
    AK_DECISION="SKIP"
    AK_CANDIDATE_PATH=""

    log_info "=== Author kernel iteration: $target / $strategy / iter $iter ==="

    # Step 1: Ensure target spec exists
    extract_target_spec "$target" || return 1

    # Step 2: Generate candidate skeleton
    local cand_path
    cand_path=$(generate_candidate_skeleton "$target" "$iter" "$strategy")
    if [ ! -f "$cand_path" ]; then
        log_error "Candidate generation failed"
        AK_DECISION="SKIP"
        return 1
    fi
    AK_CANDIDATE_PATH="$cand_path"

    log_warn "  NOTE: Candidate skeleton was generated from the reference kernel."
    log_warn "  An LLM agent should now edit $cand_path"
    log_warn "  per the strategy instructions above, THEN this harness will evaluate it."
    log_warn "  If running fully automated (no LLM edit), harness will show 1.000x (no change)."

    # Step 3: Run harness
    run_harness "$target" "$cand_path"
    AK_CORRECTNESS="$HARNESS_CORRECTNESS"
    AK_HARNESS_SPEEDUP="$HARNESS_SPEEDUP"

    if [ "$AK_CORRECTNESS" != "PASS" ]; then
        log_warn "  Correctness FAIL — skipping promotion"
        AK_DECISION="REJECT"
        return 0
    fi

    # Step 4: Check speedup threshold
    local above_threshold
    above_threshold=$(python3 -c "print(1 if float('$AK_HARNESS_SPEEDUP') >= float('$ACCEPT_MIN_SPEEDUP') else 0)" 2>/dev/null || echo "0")

    if [ "$above_threshold" != "1" ]; then
        log_warn "  Speedup ${AK_HARNESS_SPEEDUP}x below threshold ${ACCEPT_MIN_SPEEDUP}x — skip promotion"
        AK_DECISION="REJECT"
        return 0
    fi

    log_ok "  Speedup ${AK_HARNESS_SPEEDUP}x above threshold — promoting to hipfire"

    # Step 5: Promote to hipfire and rebuild
    local baseline_tok_s=0
    if [ -f "$TOOL_DIR/workspace/orchestration_state.json" ]; then
        baseline_tok_s=$(python3 -c "
import json
s = json.load(open('$TOOL_DIR/workspace/orchestration_state.json'))
print(s.get('current_best_tok_s', 0))
" 2>/dev/null || echo "0")
    fi

    promote_to_hipfire "$target" "$cand_path" "$ARCH"
    if [ "$PROMOTE_STATUS" != "OK" ]; then
        log_error "  Promotion failed"
        AK_DECISION="REJECT"
        return 0
    fi
    AK_PROMOTED=1

    # Step 6: End-to-end hipfire benchmark
    # Bug fix: bench writes ONLY to stderr; 2>/dev/null swallowed all output.
    # Bug fix: hardcoded 27b path ignored MODEL var — now uses resolve_model_path.
    log_info "  Running end-to-end hipfire bench (3 trials)..."
    local bench_bin="$REPO_ROOT/target/release/examples/bench_qwen35_mq4"
    # Resolve model path respecting MODEL env var (default qwen3.5:27b)
    local _model_size; _model_size=$(echo "${MODEL:-qwen3.5:27b}" | grep -oP '\d+b' | head -1)
    local model_path=""
    for _mp in \
        "$MODELS_DIR/qwen3.5-${_model_size}.mq4" \
        "$MODELS_DIR/qwen35-${_model_size}.mq4" \
        "$HOME/.hipfire/models/qwen3.5-${_model_size}.mq4"
    do
        if [ -f "$_mp" ]; then model_path="$_mp"; break; fi
    done
    local prompt_file="$REPO_ROOT/benchmarks/prompts/lru_cache_pep8_strict.txt"

    if [ ! -f "$bench_bin" ] || [ ! -f "$model_path" ] || [ ! -f "$prompt_file" ]; then
        log_warn "  E2E bench skipped (bench binary or model or prompt not found)"
        log_warn "    bench_bin=$bench_bin  model_path=$model_path"
        AK_HIPFIRE_TOK_S="0"
    else
        local tok_s_sum=0
        local valid_runs=0
        for trial in 1 2 3; do
            local trial_out
            # bench writes to stderr — capture with 2>&1, NOT 2>/dev/null
            trial_out=$(cat "$prompt_file" | "$bench_bin" "$model_path" --gen 80 --warmup 10 2>&1 | grep 'gen_tok_s=' | head -1)
            local ts
            ts=$(echo "$trial_out" | grep -oP 'gen_tok_s=\K[0-9.]+')
            if [ -n "$ts" ]; then
                tok_s_sum=$(python3 -c "print($tok_s_sum + $ts)")
                valid_runs=$((valid_runs + 1))
            fi
        done
        if [ "$valid_runs" -gt 0 ]; then
            AK_HIPFIRE_TOK_S=$(python3 -c "print(f'{$tok_s_sum/$valid_runs:.1f}')")
        fi
    fi

    # Step 7: Accept/reject based on e2e if available, else harness-only
    local accept=0
    if [ "$AK_HIPFIRE_TOK_S" != "0" ]; then
        accept=$(python3 -c "
b = float('$baseline_tok_s')
n = float('$AK_HIPFIRE_TOK_S')
print(1 if b == 0 or n >= b * float('$ACCEPT_MIN_SPEEDUP') else 0)
" 2>/dev/null || echo "0")
    else
        accept=1  # promoted + harness passed, no e2e data → accept tentatively
    fi

    if [ "$accept" == "1" ]; then
        AK_DECISION="ACCEPT"
        log_ok "  ACCEPTED: harness speedup ${AK_HARNESS_SPEEDUP}x, e2e tok/s ${AK_HIPFIRE_TOK_S}"
    else
        AK_DECISION="REJECT"
        log_warn "  REJECTED: e2e ${AK_HIPFIRE_TOK_S} tok/s below threshold vs baseline ${baseline_tok_s}"
        log_info "  Rolling back: git checkout kernels/src/${target}.gfx1201.hip"
        git -C "$REPO_ROOT" checkout -- "kernels/src/${target}.gfx1201.hip" 2>/dev/null || true
        AK_PROMOTED=0
    fi

    return 0
}

# ── write_phase14_report ─────────────────────────────────────────────────────
write_phase14_report() {
    local output_dir="${1:-$LAB_DIR/reports}"
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local md_file="$output_dir/phase14_kernel_authoring_${ts}.md"
    local json_file="$output_dir/phase14_kernel_authoring_${ts}.json"

    mkdir -p "$output_dir"

    cat > "$md_file" <<REPORT
# Phase 14 — Kernel Authoring Run Report
Generated: $(date -u +"%Y-%m-%d %H:%M:%S UTC")
Arch: $ARCH
Accept threshold: ${ACCEPT_MIN_SPEEDUP}x

## Summary

This report was generated by \`author_kernel.sh\` Phase 14 kernel-authoring mode.
See \`tools/autokernel-rdna/kernel_lab/\` for candidate files and harness results.

## Targets (by decode runtime share)

| Target | Decode Share | Notes |
|--------|-------------|-------|
| gemv_hfq4g256 | 20% | Dominant GEMV; BW-ceiling on gfx1201. Try LDS staging. |
| gemv_hfq6g256 | 18% | No gfx12 variant exists yet — any win is net new. |
| gemv_hfq4g256_residual | 12% | Avoid prefetch (Phase 10 regression). |

## Strategies

$(for target in gemv_hfq4g256 gemv_hfq6g256 gemv_hfq4g256_residual; do
    echo "### $target"
    local strats_var="AUTHOR_STRATEGIES_${target}"
    # safely iterate if defined
    if declare -p "$strats_var" &>/dev/null; then
        eval "for s in \"\${${strats_var}[@]}\"; do echo \"- \$s\"; done"
    fi
    echo ""
done)

## Known Negative Results

- 8x group unroll + packed uint32 loads on gemv_hfq4g256: +0.1% (Phase 12). BW ceiling hit.
- s_prefetch in gemv_hfq4g256_residual: regression (Phase 10). Avoid.

## Next Steps

1. Use this harness infrastructure to evaluate LLM-generated candidates.
2. Promote winners to \`kernels/src/\`. Verified candidates stay even if e2e bench is neutral.
3. Run \`./scripts/coherence-gate-dflash.sh\` after any promotion to hipfire.
REPORT

    # JSON summary
    python3 - <<PYEOF
import json, datetime
report = {
    "generated": datetime.datetime.utcnow().isoformat() + "Z",
    "arch": "$ARCH",
    "accept_min_speedup": float("$ACCEPT_MIN_SPEEDUP"),
    "harness_trials": $HARNESS_TRIALS,
    "targets": [
        {"name": "gemv_hfq4g256",          "decode_share": 0.20, "spec": "kernel_lab/generated/gemv_hfq4g256/target_spec.json"},
        {"name": "gemv_hfq6g256",           "decode_share": 0.18, "spec": "kernel_lab/generated/gemv_hfq6g256/target_spec.json"},
        {"name": "gemv_hfq4g256_residual",  "decode_share": 0.12, "spec": "kernel_lab/generated/gemv_hfq4g256_residual/target_spec.json"}
    ]
}
print(json.dumps(report, indent=2))
PYEOF
    > "$json_file"

    log_ok "Phase 14 report written:"
    log_ok "  $md_file"
    log_ok "  $json_file"
}

# ── Standalone entry point ───────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    TARGET="${TARGET:-gemv_hfq4g256}"
    ITER="${ITER:-1}"
    STRATEGY="${STRATEGY:-author_gemv_lds_staging_x}"

    author_kernel_iteration "$TARGET" "$ITER" "$STRATEGY"

    echo ""
    echo "=== Result ==="
    echo "AK_CORRECTNESS:     $AK_CORRECTNESS"
    echo "AK_HARNESS_SPEEDUP: $AK_HARNESS_SPEEDUP"
    echo "AK_PROMOTED:        $AK_PROMOTED"
    echo "AK_HIPFIRE_TOK_S:   $AK_HIPFIRE_TOK_S"
    echo "AK_DECISION:        $AK_DECISION"
    echo "AK_CANDIDATE_PATH:  $AK_CANDIDATE_PATH"
fi
