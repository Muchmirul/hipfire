#!/usr/bin/env bash
# tools/autokernel-rdna/run.sh — AutoKernel-style optimization loop for hipfire on RDNA.
#
# Adapts the AutoKernel approach (https://github.com/RightNow-AI/autokernel) to
# hipfire's Rust + HIP architecture targeting AMD Radeon RX 9070 XT / gfx1201.
#
# Usage:
#   ./tools/autokernel-rdna/run.sh baseline   --arch gfx1201 --model qwen3.5:9b
#   ./tools/autokernel-rdna/run.sh profile    --arch gfx1201 --model qwen3.5:9b
#   ./tools/autokernel-rdna/run.sh experiment --kernel <name> --arch gfx1201
#   ./tools/autokernel-rdna/run.sh orchestrate --arch gfx1201 --model qwen3.5:9b
#
# Hard rules (per docs/methodology/perf-benchmarking.md + AGENTS.md):
#   - Never claim speedup from a single noisy run.
#   - Always fresh-process benchmarking (rm stale bench binary before reruns).
#   - Byte-identical prompts; MD5 recorded.
#   - Run coherence-gate-dflash.sh after any kernel/dispatch/fusion change.
#   - Revert on correctness failure, even if tok/s improves.
#   - Never bypass gates with --no-verify.
#   - No Python in the hot path (this script is offline tooling only).

set -uo pipefail
IFS=$'\n\t'

TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TOOL_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# ── Paths ────────────────────────────────────────────────────────────────
RESULTS_TSV="$TOOL_DIR/results.tsv"
BASELINES_DIR="$TOOL_DIR/baselines"
REPORTS_DIR="$TOOL_DIR/reports"
EXPERIMENTS_DIR="$TOOL_DIR/experiments"
KERNELS_SRC="$REPO_ROOT/kernels/src"
BENCH_EXE="$REPO_ROOT/target/release/examples/bench_qwen35_mq4"
BENCH_PROMPT="${HIPFIRE_BENCH_PROMPT:-$REPO_ROOT/benchmarks/prompts/lru_cache_pep8_strict.txt}"
MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
KERNEL_CACHE="${HIPFIRE_KERNEL_CACHE:-$HOME/.hipfire/bin/kernels}"

# ── Defaults ─────────────────────────────────────────────────────────────
OPT_ARCH="gfx1201"
OPT_MODEL="qwen3.5:9b"
OPT_KERNEL=""
OPT_ALLOW_OTHER_ARCH=0
OPT_TRIALS=3
OPT_SPEEDUP_THRESHOLD="1.08"
OPT_VERBOSE=0
OPT_COHERENCE=1
OPT_NOTES=""

# ── Colours (when stderr is a tty) ───────────────────────────────────────
if [ -t 2 ]; then
    RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'; BLU='\033[0;34m'; RST='\033[0m'
else
    RED=''; GRN=''; YLW=''; BLU=''; RST=''
fi
log()  { echo -e "${BLU}[autokernel]${RST} $*" >&2; }
ok()   { echo -e "${GRN}[autokernel OK]${RST} $*" >&2; }
warn() { echo -e "${YLW}[autokernel WARN]${RST} $*" >&2; }
err()  { echo -e "${RED}[autokernel ERR]${RST} $*" >&2; }
die()  { err "$*"; exit 1; }

# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

# Detect active GPU arch (gfx1201, gfx1100, etc.)
detect_arch() {
    local arch=""
    for probe in amdgpu-arch offload-arch \
                 /opt/rocm/bin/amdgpu-arch /opt/rocm/bin/offload-arch \
                 /opt/rocm/llvm/bin/amdgpu-arch; do
        if command -v "$probe" >/dev/null 2>&1 || [ -x "$probe" ]; then
            arch="$("$probe" 2>/dev/null | head -1)"
            if [ -n "$arch" ]; then echo "$arch"; return 0; fi
        fi
    done
    # Fallback: KFD topology
    for node_props in /sys/class/kfd/kfd/topology/nodes/*/properties; do
        [ -f "$node_props" ] || continue
        local ver
        ver=$(awk '/gfx_target_version/ {print $2; exit}' "$node_props" 2>/dev/null || true)
        case "$ver" in
            90006)          echo "gfx906";  return 0 ;;
            100100)         echo "gfx1010"; return 0 ;;
            100300|100302)  echo "gfx1030"; return 0 ;;
            110000|110001)  echo "gfx1100"; return 0 ;;
            110501)         echo "gfx1151"; return 0 ;;
            120000)         echo "gfx1200"; return 0 ;;
            120001)         echo "gfx1201"; return 0 ;;
        esac
    done
    echo ""
}

# Detect GPU name from sysfs / rocm-smi
detect_gpu_name() {
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --showproductname 2>/dev/null | grep -oP 'Card series:\s*\K.*' | head -1 | xargs || true
    fi
    local name=""
    for f in /sys/class/drm/card*/device/product_name; do
        [ -f "$f" ] && name=$(cat "$f" 2>/dev/null | head -1) && [ -n "$name" ] && echo "$name" && return
    done
    echo "unknown"
}

# Detect ROCm version
detect_rocm_version() {
    if [ -f /opt/rocm/.info/version ]; then cat /opt/rocm/.info/version; return; fi
    if [ -f /opt/rocm/version.txt ];    then cat /opt/rocm/version.txt;   return; fi
    if command -v rocm-smi >/dev/null 2>&1; then
        rocm-smi --version 2>/dev/null | grep -oP 'ROCm\s+\K[\d.]+' | head -1 || true
    fi
    echo "unknown"
}

# Detect hipcc version
detect_hipcc_version() {
    if command -v hipcc >/dev/null 2>&1; then
        hipcc --version 2>&1 | head -1 || echo "unknown"
    else
        echo "not found"
    fi
}

# MD5 of a file
file_md5() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" 2>/dev/null | cut -d' ' -f1
    else
        md5 -q "$1" 2>/dev/null || echo "unknown"
    fi
}

# ISO timestamp
ts_now() { date -u +%Y%m%d-%H%M%S; }
ts_iso()  { date -u +%Y-%m-%dT%H:%M:%SZ; }

# Current git commit SHA
git_sha() { git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown"; }
git_status_clean() { [ -z "$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null)" ]; }

# Parse tok/s values from bench_qwen35_mq4 output.
# Looks for lines like: "gen_tok_s: 132.5" or "prefill_tok_s: 1663.2"
parse_gen_tok_s()     { grep -oP 'gen_tok_s:\s*\K[\d.]+' "$1" 2>/dev/null | tail -1 || echo "0"; }
parse_prefill_tok_s() { grep -oP 'prefill_tok_s:\s*\K[\d.]+' "$1" 2>/dev/null | tail -1 || echo "0"; }
parse_vram_mb()       { grep -oP 'vram_mb:\s*\K[\d.]+' "$1" 2>/dev/null | tail -1 || echo "0"; }

# Compute median of a space-separated list of numbers using awk
median() {
    local nums=("$@")
    local n="${#nums[@]}"
    [ "$n" -eq 0 ] && echo "0" && return
    printf '%s\n' "${nums[@]}" | sort -n | awk -v n="$n" '
        NR == int((n+1)/2) { low = $1 }
        NR == int((n+2)/2) { high = $1 }
        END { print (low + high) / 2 }
    '
}

# Compute stddev of a space-separated list of numbers using awk
stddev() {
    local nums=("$@")
    local n="${#nums[@]}"
    [ "$n" -lt 2 ] && echo "0" && return
    printf '%s\n' "${nums[@]}" | awk -v n="$n" '
        { sum += $1; sumsq += $1*$1 }
        END { mean = sum/n; print sqrt(sumsq/n - mean*mean) }
    '
}

# Float comparison: is A >= B?
float_ge() { awk "BEGIN{exit !($1 >= $2)}"; }

# Float division
float_div() { awk "BEGIN{printf \"%.4f\", $1/$2}"; }

# Append a row to results.tsv (all fields tab-separated, no trailing newline issues)
log_result() {
    local fields=("$@")
    local row
    row=$(printf '%s\t' "${fields[@]}")
    row="${row%	}"  # strip trailing tab
    echo -e "$row" >> "$RESULTS_TSV"
}

# Rebuild bench_qwen35_mq4 from clean state.
# Removes stale binary first per docs/methodology/perf-benchmarking.md.
rebuild_bench() {
    log "Rebuilding bench_qwen35_mq4 from clean (removing stale binary)..."
    rm -f "$BENCH_EXE"
    if ! cargo build --release --example bench_qwen35_mq4 --features deltanet \
            -p hipfire-runtime 2>&1; then
        err "Build failed"
        return 1
    fi
    ok "bench_qwen35_mq4 built"
}

# Run bench_qwen35_mq4 once and save output to a temp file.
# Prints the path to the output file.
run_bench_once() {
    local model_arg="${1:-}"
    local outfile
    outfile="$(mktemp /tmp/autokernel-bench-XXXXXX.txt)"
    local extra_args=()
    [ -n "$model_arg" ] && extra_args+=("--model" "$model_arg")
    if "$BENCH_EXE" "${extra_args[@]}" >"$outfile" 2>&1; then
        echo "$outfile"
        return 0
    else
        echo "$outfile"
        return 1
    fi
}

# Run N fresh-process trials and compute median gen_tok_s and prefill_tok_s.
# Outputs: <median_gen> <median_prefill> <stddev_gen> <stddev_prefill> <vram_mb>
run_bench_trials() {
    local n="$1"
    local gen_vals=() prefill_vals=() vram_vals=()
    local i status
    for ((i=1; i<=n; i++)); do
        log "  trial $i/$n..."
        local outfile
        outfile="$(run_bench_once)" || { warn "trial $i failed (non-zero exit)"; continue; }
        local g p v
        g="$(parse_gen_tok_s "$outfile")"
        p="$(parse_prefill_tok_s "$outfile")"
        v="$(parse_vram_mb "$outfile")"
        rm -f "$outfile"
        if [ "$OPT_VERBOSE" -eq 1 ]; then
            log "    gen_tok_s=$g prefill_tok_s=$p vram_mb=$v"
        fi
        gen_vals+=("$g")
        prefill_vals+=("$p")
        vram_vals+=("$v")
    done
    local med_gen med_prefill sd_gen sd_prefill med_vram
    med_gen="$(median "${gen_vals[@]}")"
    med_prefill="$(median "${prefill_vals[@]}")"
    sd_gen="$(stddev "${gen_vals[@]}")"
    sd_prefill="$(stddev "${prefill_vals[@]}")"
    med_vram="$(median "${vram_vals[@]}")"
    echo "$med_gen $med_prefill $sd_gen $sd_prefill $med_vram"
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — BASELINE
# ═══════════════════════════════════════════════════════════════════════════
cmd_baseline() {
    local ts; ts="$(ts_now)"
    log "=== AutoKernel BASELINE for $OPT_ARCH / $OPT_MODEL ==="

    # ── Environment verification ────────────────────────────────────────
    local detected_arch
    detected_arch="$(detect_arch)"

    if [ -z "$detected_arch" ]; then
        die "Could not detect GPU arch. Is ROCm installed and the GPU visible?"
    fi

    if [ "$detected_arch" != "$OPT_ARCH" ]; then
        if [ "$OPT_ALLOW_OTHER_ARCH" -eq 1 ]; then
            warn "Active GPU is $detected_arch, not $OPT_ARCH — proceeding (--allow-other-arch)"
        else
            die "Active GPU arch is $detected_arch, expected $OPT_ARCH. Pass --allow-other-arch to override."
        fi
    else
        ok "GPU arch: $detected_arch"
    fi

    local gpu_name rocm_ver hipcc_ver git_sha_val git_dirty
    gpu_name="$(detect_gpu_name)"
    rocm_ver="$(detect_rocm_version)"
    hipcc_ver="$(detect_hipcc_version)"
    git_sha_val="$(git_sha)"
    git_dirty="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null | wc -l | xargs)"

    log "GPU:    $gpu_name ($detected_arch)"
    log "ROCm:   $rocm_ver"
    log "hipcc:  $hipcc_ver"
    log "commit: $git_sha_val (dirty=$git_dirty)"

    # ── Prompt MD5 ──────────────────────────────────────────────────────
    if [ ! -f "$BENCH_PROMPT" ]; then
        die "Bench prompt not found: $BENCH_PROMPT"
    fi
    local prompt_md5; prompt_md5="$(file_md5 "$BENCH_PROMPT")"
    local prompt_name; prompt_name="$(basename "$BENCH_PROMPT")"
    log "Prompt: $prompt_name (md5=$prompt_md5)"

    # ── Build ────────────────────────────────────────────────────────────
    rebuild_bench || die "Cannot proceed without bench binary"

    # ── Collect HIPFIRE_* env vars ───────────────────────────────────────
    local hipfire_env
    hipfire_env="$(env | grep '^HIPFIRE_' | sort | tr '\n' ' ' || true)"

    # ── Run baseline benchmark ($OPT_TRIALS trials) ─────────────────────
    log "Running baseline ($OPT_TRIALS trials, DPM warmup is built into bench)..."
    local bench_results
    read -r med_gen med_prefill sd_gen sd_prefill med_vram <<< \
        "$(run_bench_trials "$OPT_TRIALS")"

    ok "Baseline decode:  $med_gen tok/s  (σ=$sd_gen)"
    ok "Baseline prefill: $med_prefill tok/s"
    ok "VRAM:             $med_vram MB"

    # ── Write JSON ───────────────────────────────────────────────────────
    local out_json="$BASELINES_DIR/${ts}.json"
    cat > "$out_json" <<EOF
{
  "timestamp": "$(ts_iso)",
  "arch": "$detected_arch",
  "target_arch": "$OPT_ARCH",
  "gpu_name": "$gpu_name",
  "rocm_version": "$rocm_ver",
  "hipcc_version": "$hipcc_ver",
  "git_sha": "$git_sha_val",
  "git_dirty_files": $git_dirty,
  "model": "$OPT_MODEL",
  "bench_prompt": "$prompt_name",
  "prompt_md5": "$prompt_md5",
  "kernel_cache_path": "$KERNEL_CACHE",
  "models_dir": "$MODELS_DIR",
  "hipfire_env": "$hipfire_env",
  "trials": $OPT_TRIALS,
  "baseline_decode_tok_s": $med_gen,
  "baseline_prefill_tok_s": $med_prefill,
  "decode_stddev": $sd_gen,
  "prefill_stddev": $sd_prefill,
  "vram_mb": $med_vram
}
EOF
    ok "Baseline saved: $out_json"
    echo "$out_json"
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — PROFILE (bottleneck ranking)
# ═══════════════════════════════════════════════════════════════════════════
cmd_profile() {
    local ts; ts="$(ts_now)"
    log "=== AutoKernel PROFILE for $OPT_ARCH / $OPT_MODEL ==="

    local detected_arch; detected_arch="$(detect_arch)"
    if [ -n "$detected_arch" ] && [ "$detected_arch" != "$OPT_ARCH" ] && [ "$OPT_ALLOW_OTHER_ARCH" -eq 0 ]; then
        die "Active GPU is $detected_arch, expected $OPT_ARCH. Pass --allow-other-arch to override."
    fi

    rebuild_bench || die "Cannot profile without bench binary"

    # ── Single bench run to gather timing signal ─────────────────────────
    log "Running profile bench pass (1 trial for timing signal)..."
    local outfile; outfile="$(run_bench_once)"
    local gen_tok_s prefill_tok_s
    gen_tok_s="$(parse_gen_tok_s "$outfile")"
    prefill_tok_s="$(parse_prefill_tok_s "$outfile")"
    log "Bench signal: decode=$gen_tok_s tok/s  prefill=$prefill_tok_s tok/s"

    # ── Kernel inventory for gfx1201 ─────────────────────────────────────
    # Classify kernels present in kernels/src/ by type and decode-path relevance.
    # Estimate time fraction using architecture knowledge + relative op counts.
    # For a 9B model (Qwen 3.5, 28 layers):
    #   decode dominant ops (memory-bandwidth bound, single token):
    #     - GEMV for QKV, O, gate/up, down, residual (~70% of decode)
    #     - RMSNorm, RoPE (~5%)
    #     - FlashAttention (KV cache read, softmax) (~15%)
    #     - KV cache quant/write (~5%)
    #     - misc (~5%)
    #   prefill dominant ops (compute bound, full sequence):
    #     - GEMM for QKV, O, gate/up, down (~75% of prefill)
    #     - FlashAttention full seq (~20%)
    #     - misc (~5%)

    # Check which gfx12 / gfx1201 variants already exist
    local existing_gfx12 existing_gfx1201
    existing_gfx12=$(ls "$KERNELS_SRC"/*.gfx12.hip 2>/dev/null | xargs -I{} basename {} .hip | sed 's/\.gfx12$//' | sort)
    existing_gfx1201=$(ls "$KERNELS_SRC"/*.gfx1201.hip 2>/dev/null | xargs -I{} basename {} .hip | sed 's/\.gfx1201$//' | sort)

    # ── Candidate kernels with Amdahl estimates ───────────────────────────
    # Format: name|decode_frac|prefill_frac|call_freq|bound_type|gfx12_exists|gfx1201_exists|priority_note
    # decode_frac + prefill_frac are fractions of total time (0.0–1.0).
    # Amdahl end-to-end decode impact = decode_frac * (1 - 1/expected_speedup)
    declare -A KERNEL_META
    #                        name                              dec_f  pre_f  freq  bound   note
    KERNEL_META["gemv_hfq4g256"]="decode:0.20|prefill:0.01|freq:high|bound:mem|note:Main decode GEMV for HFQ4 weights. gfx1201 variant exists — verify tuning."
    KERNEL_META["gemv_hfq6g256"]="decode:0.18|prefill:0.01|freq:high|bound:mem|note:Main decode GEMV for HFQ6 weights. gfx1201 variant exists."
    KERNEL_META["gemv_hfq4g256_residual"]="decode:0.12|prefill:0.00|freq:high|bound:mem|note:Fused residual GEMV for HFQ4. gfx1100 variant — gfx12 opportunity."
    KERNEL_META["gemv_hfq6g256_residual"]="decode:0.10|prefill:0.00|freq:high|bound:mem|note:Fused residual GEMV for HFQ6. gfx1100 variant — gfx12 opportunity."
    KERNEL_META["attention_flash_asym3_tile"]="decode:0.12|prefill:0.08|freq:high|bound:mem|note:FlashAttention asym3 KV tile path. No arch variant — gfx12 opportunity."
    KERNEL_META["fused_gate_up_hfq4g256"]="decode:0.06|prefill:0.02|freq:high|bound:mem|note:Fused gate+up SwiGLU GEMV decode path."
    KERNEL_META["fused_gate_up_hfq6g256"]="decode:0.05|prefill:0.02|freq:high|bound:mem|note:Fused gate+up for HFQ6."
    KERNEL_META["gemm_gate_up_hfq4g256_wmma"]="decode:0.00|prefill:0.15|freq:med|bound:compute|note:Prefill SwiGLU WMMA. gfx12 variant exists."
    KERNEL_META["gemm_gate_up_hfq6g256_wmma"]="decode:0.00|prefill:0.12|freq:med|bound:compute|note:Prefill SwiGLU WMMA HFQ6. gfx12 variant exists."
    KERNEL_META["gemm_qkvza_hfq4g256_wmma"]="decode:0.00|prefill:0.12|freq:med|bound:compute|note:Prefill QKV+ZA WMMA HFQ4. gfx12 variant exists."
    KERNEL_META["gemm_hfq4g256_residual_wmma"]="decode:0.00|prefill:0.08|freq:med|bound:compute|note:Prefill residual WMMA HFQ4. gfx12 variant exists."
    KERNEL_META["fused_qkv_hfq4g256"]="decode:0.08|prefill:0.01|freq:high|bound:mem|note:Fused QKV GEMV decode. No gfx12 variant."
    KERNEL_META["fused_qkvza_hfq4g256"]="decode:0.07|prefill:0.01|freq:high|bound:mem|note:Fused QKV+ZA GEMV decode."
    KERNEL_META["attention_flash_asym3_tile_batched"]="decode:0.04|prefill:0.04|freq:med|bound:mem|note:Batched flash-attn asym3."
    KERNEL_META["rmsnorm"]="decode:0.03|prefill:0.03|freq:high|bound:mem|note:RMSNorm — low absolute time but many invocations."
    KERNEL_META["fused_rmsnorm_mq_rotate"]="decode:0.04|prefill:0.03|freq:high|bound:mem|note:Fused RMSNorm+MQ rotate."

    # ── Score each kernel by Amdahl decode impact ────────────────────────
    # Expected max speedup from tuning decode GEMV on gfx1201: ~1.3x on that kernel
    # Amdahl end-to-end = 1 / ((1 - f) + f/s)  where f=fraction, s=kernel speedup
    local expected_kernel_speedup=1.30
    local report_md="$REPORTS_DIR/profile_${ts}.md"
    local report_json="$REPORTS_DIR/profile_${ts}.json"

    {
        echo "# AutoKernel Profile — $OPT_ARCH / $OPT_MODEL"
        echo ""
        echo "Generated: $(ts_iso)"
        echo "Bench signal: decode=$gen_tok_s tok/s | prefill=$prefill_tok_s tok/s"
        echo ""
        echo "## Kernel Optimization Candidates (ranked by Amdahl decode impact)"
        echo ""
        echo "| Rank | Kernel | Dec% | Pre% | Bound | E2E Gain (1.3x) | gfx12 | gfx1201 | Note |"
        echo "|------|--------|------|------|-------|-----------------|-------|---------|------|"
    } > "$report_md"

    # Build ranked list
    local ranked=()
    for kname in "${!KERNEL_META[@]}"; do
        local meta="${KERNEL_META[$kname]}"
        local dec_f; dec_f=$(echo "$meta" | grep -oP 'decode:\K[\d.]+')
        local pre_f; pre_f=$(echo "$meta" | grep -oP 'prefill:\K[\d.]+')
        # Amdahl decode impact score
        local e2e_gain
        e2e_gain=$(awk -v f="$dec_f" -v s="$expected_kernel_speedup" \
            'BEGIN{e2e=1/((1-f)+f/s); printf "%.4f", e2e}')
        ranked+=("${e2e_gain}|${kname}")
    done
    IFS=$'\n' sorted_ranked=($(sort -t'|' -k1 -rn <<< "${ranked[*]}")); unset IFS

    local rank=1
    local json_entries=()
    for entry in "${sorted_ranked[@]}"; do
        local e2e_gain="${entry%%|*}"
        local kname="${entry#*|}"
        local meta="${KERNEL_META[$kname]}"
        local dec_f; dec_f=$(echo "$meta" | grep -oP 'decode:\K[\d.]+')
        local pre_f; pre_f=$(echo "$meta" | grep -oP 'prefill:\K[\d.]+')
        local bound;  bound=$(echo "$meta"  | grep -oP 'bound:\K\w+')
        local note;   note=$(echo "$meta"   | grep -oP 'note:\K.*')
        local has_gfx12="no"
        local has_gfx1201="no"
        echo "$existing_gfx12" | grep -qx "$kname"   && has_gfx12="yes"
        echo "$existing_gfx1201" | grep -qx "$kname" && has_gfx1201="yes"

        local dec_pct; dec_pct=$(awk "BEGIN{printf \"%d\", $dec_f*100}")
        local pre_pct; pre_pct=$(awk "BEGIN{printf \"%d\", $pre_f*100}")

        echo "| $rank | \`$kname\` | ${dec_pct}% | ${pre_pct}% | $bound | ${e2e_gain}x | $has_gfx12 | $has_gfx1201 | $note |" >> "$report_md"

        json_entries+=("{\"rank\":$rank,\"kernel\":\"$kname\",\"decode_frac\":$dec_f,\"prefill_frac\":$pre_f,\"bound_type\":\"$bound\",\"amdahl_e2e_gain\":$e2e_gain,\"gfx12_variant\":$( [[ "$has_gfx12" == "yes" ]] && echo true || echo false),\"gfx1201_variant\":$( [[ "$has_gfx1201" == "yes" ]] && echo true || echo false),\"note\":\"$(echo "$note" | sed 's/"/\\"/g')\"}")
        rank=$((rank + 1))
    done

    {
        echo ""
        echo "## Notes"
        echo ""
        echo "- E2E Gain column assumes 1.3x speedup on that kernel in isolation."
        echo "- Decode fractions estimated from hipfire architecture + op counts for Qwen3.5 9B."
        echo "- Run \`./tools/autokernel-rdna/run.sh experiment --kernel <name>\` to optimize a specific kernel."
        echo "- Run \`./tools/autokernel-rdna/run.sh orchestrate\` to let the scheduler pick automatically."
        echo ""
        echo "## rocprof Integration (optional)"
        echo ""
        echo "If \`rocprof\` is available, run:"
        echo "\`\`\`bash"
        echo "rocprof --stats $BENCH_EXE 2>&1 | tee $REPORTS_DIR/rocprof_${ts}.csv"
        echo "\`\`\`"
        echo "Then update the decode_frac estimates above from the per-kernel GPU time column."
    } >> "$report_md"

    # Write JSON
    local json_arr; json_arr=$(printf '%s,' "${json_entries[@]}")
    json_arr="[${json_arr%,}]"
    cat > "$report_json" <<EOF
{
  "timestamp": "$(ts_iso)",
  "arch": "$OPT_ARCH",
  "model": "$OPT_MODEL",
  "bench_decode_tok_s": $gen_tok_s,
  "bench_prefill_tok_s": $prefill_tok_s,
  "candidates": $json_arr
}
EOF

    ok "Profile report: $report_md"
    ok "Profile JSON:   $report_json"
    echo "$report_json"
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — EXPERIMENT (single kernel optimization attempt)
# ═══════════════════════════════════════════════════════════════════════════
cmd_experiment() {
    local ts; ts="$(ts_now)"
    [ -z "$OPT_KERNEL" ] && die "--kernel <name> is required for experiment"

    log "=== AutoKernel EXPERIMENT: $OPT_KERNEL on $OPT_ARCH ==="

    # ── Load baseline ───────────────────────────────────────────────────
    local baseline_json
    baseline_json="$(ls -t "$BASELINES_DIR"/*.json 2>/dev/null | head -1 || true)"
    if [ -z "$baseline_json" ]; then
        warn "No baseline found — running baseline first..."
        baseline_json="$(cmd_baseline)"
    fi

    local baseline_decode baseline_prefill baseline_vram
    baseline_decode=$(grep -oP '"baseline_decode_tok_s":\s*\K[\d.]+' "$baseline_json" || echo "0")
    baseline_prefill=$(grep -oP '"baseline_prefill_tok_s":\s*\K[\d.]+' "$baseline_json" || echo "0")
    baseline_vram=$(grep -oP '"vram_mb":\s*\K[\d.]+' "$baseline_json" || echo "0")
    log "Baseline: decode=$baseline_decode prefill=$baseline_prefill vram=$baseline_vram MB"

    # ── Git branch ──────────────────────────────────────────────────────
    local base_sha; base_sha="$(git_sha)"
    local branch="autokernel/gfx1201-${OPT_KERNEL}-$(date +%Y%m%d)"
    local existing_branch
    existing_branch=$(git -C "$REPO_ROOT" branch --list "$branch" | xargs)
    if [ -n "$existing_branch" ]; then
        branch="${branch}-$(date +%H%M%S)"
    fi
    log "Creating branch: $branch"
    git -C "$REPO_ROOT" checkout -b "$branch" 2>&1 | head -3

    # ── Determine kernel variant file ───────────────────────────────────
    # Prefer .gfx12.hip (covers gfx1200+gfx1201) unless strictly gfx1201-only.
    local src_base="$KERNELS_SRC/${OPT_KERNEL}"
    local variant_file=""
    local variant_scope=""

    # Check if a gfx1201-specific variant makes sense vs gfx12-wide
    # Default: use gfx12-wide (safer, covers both 9070 and 9070 XT)
    local prefer_gfx12=1

    # If a gfx1201 variant already exists for this kernel, use it
    if [ -f "${src_base}.gfx1201.hip" ]; then
        variant_file="${src_base}.gfx1201.hip"
        variant_scope="gfx1201"
    elif [ -f "${src_base}.gfx12.hip" ]; then
        variant_file="${src_base}.gfx12.hip"
        variant_scope="gfx12"
    elif [ "$prefer_gfx12" -eq 1 ]; then
        variant_file="${src_base}.gfx12.hip"
        variant_scope="gfx12"
        log "Creating new gfx12 variant: $(basename "$variant_file")"
        # Bootstrap from the existing generic kernel
        if [ -f "${src_base}.hip" ]; then
            cp "${src_base}.hip" "$variant_file"
            log "  Bootstrapped from ${OPT_KERNEL}.hip"
        else
            warn "No base ${OPT_KERNEL}.hip found — variant file will be empty"
            touch "$variant_file"
        fi
    else
        variant_file="${src_base}.gfx1201.hip"
        variant_scope="gfx1201"
        if [ -f "${src_base}.hip" ]; then
            cp "${src_base}.hip" "$variant_file"
        else
            touch "$variant_file"
        fi
    fi
    local variant_basename; variant_basename="$(basename "$variant_file")"
    log "Kernel variant file: kernels/src/$variant_basename"

    # ── Record experiment start ─────────────────────────────────────────
    local exp_dir="$EXPERIMENTS_DIR/${ts}-${OPT_KERNEL}"
    mkdir -p "$exp_dir"
    cp "$variant_file" "$exp_dir/original_${variant_basename}" 2>/dev/null || true

    # ── Agent modification hook ─────────────────────────────────────────
    # This is the file the agent/human modifies. For automated runs, check
    # if a patch file exists in experiments/<kernel>.patch; if not, remind
    # the user to modify the file and press Enter.
    local patch_file="$EXPERIMENTS_DIR/${OPT_KERNEL}.patch"
    if [ -f "$patch_file" ]; then
        log "Applying patch from $patch_file..."
        if ! git -C "$REPO_ROOT" apply "$patch_file" 2>&1; then
            err "Patch failed to apply"
            git -C "$REPO_ROOT" checkout "$variant_file" 2>/dev/null || true
            git -C "$REPO_ROOT" checkout master 2>/dev/null || true
            git -C "$REPO_ROOT" branch -D "$branch" 2>/dev/null || true
            _log_failed_experiment "$ts" "$base_sha" "" "SKIP" "SKIP" "patch_failed" \
                "$baseline_decode" "0" "$baseline_prefill" "0" "0" "0" "$variant_basename" \
                "Patch file apply failed" "REVERT"
            return 1
        fi
    else
        log ""
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log "  Modify the kernel variant, then return here to continue."
        log "  File: kernels/src/$variant_basename"
        log ""
        log "  RDNA4/gfx1201 optimization hints:"
        log "    - wave32 is native; avoid wave64 constructs"
        log "    - WMMA builtins available: __builtin_amdgcn_wmma_*"
        log "    - LDS bank width = 32 bytes; stride multiples of 32 cause conflicts"
        log "    - Vectorize loads: float4 for 128-bit aligned, float2 for 64-bit"
        log "    - Tune: block size, unroll factor, VGPRs, LDS usage"
        log "    - For memory-bound decode: minimize global reads, maximize reuse"
        log "    - For compute-bound prefill: maximize WMMA utilization"
        log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        log ""
        if [ -t 0 ]; then
            read -r -p "[autokernel] Press Enter when you've modified the kernel file (or Ctrl-C to cancel)..." </dev/tty
        else
            warn "Non-interactive mode and no patch file provided. Proceeding with unmodified kernel."
        fi
    fi

    # Save modified version
    cp "$variant_file" "$exp_dir/candidate_${variant_basename}" 2>/dev/null || true
    local change_summary="gfx12 variant for ${OPT_KERNEL} targeting $OPT_ARCH"

    # ── Build ────────────────────────────────────────────────────────────
    log "Building from clean state..."
    local build_status="PASS"
    rebuild_bench || { build_status="FAIL"; }

    if [ "$build_status" = "FAIL" ]; then
        err "Build failed — reverting"
        git -C "$REPO_ROOT" checkout -- "$variant_file" 2>/dev/null || true
        git -C "$REPO_ROOT" checkout master 2>/dev/null || true
        git -C "$REPO_ROOT" branch -D "$branch" 2>/dev/null || true
        _log_failed_experiment "$ts" "$base_sha" "" "SKIP" "SKIP" "FAIL" \
            "$baseline_decode" "0" "$baseline_prefill" "0" "0" "0" "$variant_basename" \
            "Build failed" "REVERT"
        return 1
    fi

    # ── Correctness tests ────────────────────────────────────────────────
    log "Running correctness tests..."
    local correctness_status="PASS"
    if [ -x "./scripts/test-kernels.sh" ]; then
        if ! ./scripts/test-kernels.sh 2>&1 | tail -20; then
            correctness_status="FAIL"
        fi
    else
        warn "test-kernels.sh not found; skipping kernel unit tests"
    fi

    # ── Coherence gate ───────────────────────────────────────────────────
    local coherence_status="SKIP"
    if [ "$OPT_COHERENCE" -eq 1 ]; then
        log "Running coherence gate (--fast)..."
        coherence_status="PASS"
        if ! ./scripts/coherence-gate-dflash.sh --fast 2>&1 | tail -30; then
            coherence_status="FAIL"
        fi
    fi

    # ── Gate on correctness and coherence ───────────────────────────────
    if [ "$correctness_status" = "FAIL" ] || [ "$coherence_status" = "FAIL" ]; then
        err "Correctness/coherence gate FAILED — reverting"
        git -C "$REPO_ROOT" checkout -- "$variant_file" 2>/dev/null || true
        git -C "$REPO_ROOT" checkout master 2>/dev/null || true
        git -C "$REPO_ROOT" branch -D "$branch" 2>/dev/null || true
        _log_failed_experiment "$ts" "$base_sha" "" "$correctness_status" "$coherence_status" "$build_status" \
            "$baseline_decode" "0" "$baseline_prefill" "0" "0" "0" "$variant_basename" \
            "$change_summary" "REVERT"
        return 1
    fi

    # ── Fresh-process benchmark ($OPT_TRIALS trials) ────────────────────
    log "Running $OPT_TRIALS fresh-process benchmark trials..."
    local bench_out
    read -r med_gen med_prefill sd_gen sd_prefill med_vram <<< \
        "$(run_bench_trials "$OPT_TRIALS")"

    local candidate_sha; candidate_sha="$(git_sha)"

    # ── Compare to baseline ──────────────────────────────────────────────
    local decode_speedup prefill_speedup
    decode_speedup=$(float_div "$med_gen" "$baseline_decode" 2>/dev/null || echo "1.0000")
    prefill_speedup=$(float_div "$med_prefill" "$baseline_prefill" 2>/dev/null || echo "1.0000")

    # Amdahl estimated end-to-end (decode is dominant for interactive use)
    local e2e_estimated
    e2e_estimated="$decode_speedup"   # simple proxy; real Amdahl needs fraction

    log "Results:"
    log "  decode:  baseline=$baseline_decode  candidate=$med_gen  speedup=${decode_speedup}x"
    log "  prefill: baseline=$baseline_prefill  candidate=$med_prefill  speedup=${prefill_speedup}x"
    log "  vram:    baseline=$baseline_vram  candidate=$med_vram MB"

    # ── Decision ─────────────────────────────────────────────────────────
    local decision revert_reason=""
    if float_ge "$decode_speedup" "$OPT_SPEEDUP_THRESHOLD"; then
        decision="KEEP"
        ok "Speedup ${decode_speedup}x exceeds threshold ${OPT_SPEEDUP_THRESHOLD}x — KEEPING"
    else
        decision="REVERT"
        revert_reason="decode speedup ${decode_speedup}x below threshold ${OPT_SPEEDUP_THRESHOLD}x"
        warn "Speedup ${decode_speedup}x below threshold ${OPT_SPEEDUP_THRESHOLD}x — REVERTING"
        git -C "$REPO_ROOT" checkout -- "$variant_file" 2>/dev/null || true
        git -C "$REPO_ROOT" checkout master 2>/dev/null || true
        git -C "$REPO_ROOT" branch -D "$branch" 2>/dev/null || true
    fi

    # ── Append to results.tsv ────────────────────────────────────────────
    local exp_id; exp_id="$(wc -l < "$RESULTS_TSV" | xargs)"  # line count = next id
    local prompt_md5; prompt_md5="$(file_md5 "$BENCH_PROMPT")"
    local prompt_name; prompt_name="$(basename "$BENCH_PROMPT")"

    log_result \
        "$exp_id" "$(ts_iso)" "$base_sha" "$candidate_sha" \
        "$OPT_ARCH" "$(detect_gpu_name)" "$(detect_rocm_version)" \
        "$OPT_MODEL" "auto" "asym3" \
        "$prompt_name" "$prompt_md5" \
        "$OPT_KERNEL" "kernels/src/$variant_basename" "$change_summary" \
        "$correctness_status" "$coherence_status" "$build_status" \
        "$baseline_decode" "$med_gen" "$decode_speedup" \
        "$baseline_prefill" "$med_prefill" "$prefill_speedup" \
        "$e2e_estimated" "$decode_speedup" \
        "$OPT_TRIALS" "$sd_gen" "$med_vram" \
        "$decision" "$revert_reason" "${OPT_NOTES:-}"

    ok "Result logged to $RESULTS_TSV (id=$exp_id)"
    [ "$decision" = "KEEP" ] && return 0 || return 1
}

# Helper: log a failed/skipped experiment row
_log_failed_experiment() {
    local ts="$1" base_sha="$2" cand_sha="${3:-}" corr="$4" coher="$5" build="$6"
    local bl_dec="$7" cand_dec="$8" bl_pre="$9" cand_pre="${10}"
    local e2e="${11}" meas="${12}" variant="${13}" summary="${14}"
    local decision="${15}" revert="${16:-}"
    local exp_id; exp_id="$(wc -l < "$RESULTS_TSV" | xargs)"
    local prompt_md5; prompt_md5="$(file_md5 "$BENCH_PROMPT")"
    local prompt_name; prompt_name="$(basename "$BENCH_PROMPT")"
    log_result \
        "$exp_id" "$(ts_iso)" "$base_sha" "${cand_sha:-$base_sha}" \
        "$OPT_ARCH" "$(detect_gpu_name)" "$(detect_rocm_version)" \
        "$OPT_MODEL" "auto" "asym3" \
        "$prompt_name" "$prompt_md5" \
        "$OPT_KERNEL" "kernels/src/$variant" "$summary" \
        "$corr" "$coher" "$build" \
        "$bl_dec" "$cand_dec" "1.0000" \
        "$bl_pre" "$cand_pre" "1.0000" \
        "$e2e" "$meas" \
        "$OPT_TRIALS" "0" "0" \
        "$decision" "$revert" "${OPT_NOTES:-}"
}

# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — ORCHESTRATE (Amdahl-driven multi-kernel scheduler)
# ═══════════════════════════════════════════════════════════════════════════
cmd_orchestrate() {
    local ts; ts="$(ts_now)"
    log "=== AutoKernel ORCHESTRATE for $OPT_ARCH / $OPT_MODEL ==="

    # ── Load or run profile ──────────────────────────────────────────────
    local profile_json
    profile_json="$(ls -t "$REPORTS_DIR"/profile_*.json 2>/dev/null | head -1 || true)"
    if [ -z "$profile_json" ]; then
        log "No profile found — running profile step first..."
        profile_json="$(cmd_profile)"
    fi
    log "Using profile: $profile_json"

    # ── Load baseline ────────────────────────────────────────────────────
    local baseline_json
    baseline_json="$(ls -t "$BASELINES_DIR"/*.json 2>/dev/null | head -1 || true)"
    if [ -z "$baseline_json" ]; then
        log "No baseline found — running baseline step first..."
        baseline_json="$(cmd_baseline)"
    fi

    # ── Extract ranked candidates from profile JSON ──────────────────────
    # Parse kernel names in rank order from JSON
    local candidates
    candidates=$(python3 -c "
import json, sys
with open('$profile_json') as f: d=json.load(f)
for c in sorted(d['candidates'], key=lambda x: -x['amdahl_e2e_gain']):
    print(c['kernel'])
" 2>/dev/null || grep -oP '"kernel":"[^"]+' "$profile_json" | cut -d'"' -f4)

    if [ -z "$candidates" ]; then
        die "Could not parse candidates from $profile_json"
    fi

    local final_report="$REPORTS_DIR/final_${ts}.md"
    {
        echo "# AutoKernel Final Report — $OPT_ARCH / $OPT_MODEL"
        echo ""
        echo "Generated: $(ts_iso)"
        echo "Profile: $profile_json"
        echo ""
        echo "## Optimization Run"
        echo ""
    } > "$final_report"

    local attempted=0 accepted=0 rejected=0
    local already_tried=()

    # ── Optimization loop ────────────────────────────────────────────────
    for kname in $candidates; do
        # Skip if already tried this kernel (from results.tsv)
        local prev_result
        prev_result=$(grep "	${kname}	" "$RESULTS_TSV" 2>/dev/null | tail -1 || true)
        if [ -n "$prev_result" ]; then
            local prev_decision; prev_decision=$(echo "$prev_result" | awk -F'\t' '{print $30}')
            if [ "$prev_decision" = "KEEP" ]; then
                log "Skipping $kname — already accepted"
                already_tried+=("$kname")
                continue
            fi
            local fail_count; fail_count=$(grep "	${kname}	" "$RESULTS_TSV" 2>/dev/null | wc -l | xargs)
            if [ "$fail_count" -ge 3 ]; then
                log "Skipping $kname — $fail_count failed attempts (diminishing returns)"
                already_tried+=("$kname")
                continue
            fi
        fi

        log "Next candidate: $kname"
        OPT_KERNEL="$kname"
        attempted=$((attempted + 1))

        if cmd_experiment; then
            accepted=$((accepted + 1))
            echo "- **KEPT** \`$kname\` — experiment passed all gates" >> "$final_report"
        else
            rejected=$((rejected + 1))
            echo "- **REVERTED** \`$kname\` — speedup insufficient or gate failure" >> "$final_report"
        fi

        # Stop after too many consecutive failures (guard against thrashing)
        if [ "$attempted" -ge 8 ] && [ "$accepted" -eq 0 ]; then
            warn "8 attempts with no acceptance — stopping to avoid thrashing"
            break
        fi
    done

    # ── Final report ─────────────────────────────────────────────────────
    {
        echo ""
        echo "## Summary"
        echo ""
        echo "- Attempted: $attempted"
        echo "- Accepted:  $accepted"
        echo "- Reverted:  $rejected"
        echo ""
        echo "## results.tsv"
        echo ""
        echo "Full experiment log: $RESULTS_TSV"
        echo ""
        echo "## Remaining Opportunities"
        echo ""
        echo "Kernels not yet attempted or still improvable:"
        for kname in $candidates; do
            local in_tried=0
            for t in "${already_tried[@]:-}"; do [[ "$t" == "$kname" ]] && in_tried=1; done
            [ "$in_tried" -eq 0 ] && echo "- \`$kname\`"
        done
    } >> "$final_report"

    ok "Final report: $final_report"
    cat "$final_report"
}

# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════
usage() {
    cat >&2 <<'EOF'
Usage:
  ./tools/autokernel-rdna/run.sh <command> [options]

Commands:
  baseline    Capture environment + run baseline benchmark
  profile     Profile kernels and rank optimization candidates
  experiment  Run a single kernel optimization experiment
  orchestrate Run the full multi-kernel optimization loop (Amdahl scheduler)

Global options:
  --arch <gfx>          Target arch (default: gfx1201)
  --model <name>        hipfire model tag (default: qwen3.5:9b)
  --allow-other-arch    Allow running on non-target arch (testing only)
  --trials <n>          Number of fresh-process benchmark trials (default: 3)
  --speedup-threshold   Minimum speedup to accept (default: 1.08)
  --verbose             Print full bench output for each trial
  --no-coherence        Skip coherence gate (NOT RECOMMENDED)
  --notes <text>        Freeform notes appended to results.tsv

experiment options:
  --kernel <name>       Kernel base name (e.g. gemv_hfq4g256)

Examples:
  ./tools/autokernel-rdna/run.sh baseline --arch gfx1201 --model qwen3.5:9b
  ./tools/autokernel-rdna/run.sh profile  --arch gfx1201 --model qwen3.5:9b
  ./tools/autokernel-rdna/run.sh experiment --kernel gemv_hfq4g256 --arch gfx1201
  ./tools/autokernel-rdna/run.sh orchestrate --arch gfx1201 --model qwen3.5:9b
EOF
}

COMMAND="${1:-}"
shift || true

while [ $# -gt 0 ]; do
    case "$1" in
        --arch)               OPT_ARCH="$2";              shift 2 ;;
        --model)              OPT_MODEL="$2";             shift 2 ;;
        --kernel)             OPT_KERNEL="$2";            shift 2 ;;
        --allow-other-arch)   OPT_ALLOW_OTHER_ARCH=1;    shift   ;;
        --trials)             OPT_TRIALS="$2";            shift 2 ;;
        --speedup-threshold)  OPT_SPEEDUP_THRESHOLD="$2"; shift 2 ;;
        --verbose)            OPT_VERBOSE=1;              shift   ;;
        --no-coherence)       OPT_COHERENCE=0;            shift   ;;
        --notes)              OPT_NOTES="$2";             shift 2 ;;
        -h|--help)            usage; exit 0 ;;
        *)                    die "Unknown option: $1" ;;
    esac
done

case "$COMMAND" in
    baseline)   cmd_baseline   ;;
    profile)    cmd_profile    ;;
    experiment) cmd_experiment ;;
    orchestrate) cmd_orchestrate ;;
    ""|help|-h|--help) usage; exit 0 ;;
    *)          die "Unknown command: $COMMAND. Use baseline|profile|experiment|orchestrate" ;;
esac
