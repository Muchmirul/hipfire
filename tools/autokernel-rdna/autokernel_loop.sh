#!/usr/bin/env bash
# tools/autokernel-rdna/autokernel_loop.sh
#
# Phase 13 — Autonomous AutoKernel-style optimization loop for hipfire on gfx1201.
# Phase 14 — Real kernel-authoring mode (AUTHOR_KERNEL=1): generates new HIP kernel
#             candidates via author_kernel.sh, evaluates in isolated harnesses, promotes
#             winners to kernels/src/ before wiring into hipfire.
# Primary target: Qwen3.5-27B decode throughput.
#
# Usage (Phase 13 — sed-mutation loop):
#   ARCH=gfx1201 MODEL=qwen3.5:27b MAX_ITERS=100 ./tools/autokernel-rdna/autokernel_loop.sh
#
# Usage (Phase 14 — kernel authoring mode):
#   ARCH=gfx1201 MODEL=qwen3.5:27b AUTHOR_KERNEL=1 MAX_ITERS=50 \
#     ACCEPT_MIN_SPEEDUP=1.005 ./tools/autokernel-rdna/autokernel_loop.sh
#
# Overnight aggressive (Phase 13):
#   ARCH=gfx1201 MODEL=qwen3.5:27b MAX_ITERS=300 ACCEPT_MIN_SPEEDUP=1.005 AUTOCOMMIT=1 \
#     ./tools/autokernel-rdna/autokernel_loop.sh
#
# Stop safely (any time):
#   touch .autokernel_stop          — checked after every iteration
#   Ctrl-C / kill -TERM             — caught; writes final report before exit
#
# Hard rules:
#   - Never keep a candidate that fails correctness.
#   - Always fresh-process benchmarking (binary removed before each rebuild).
#   - Byte-identical prompts; MD5 recorded in every TSV row.
#   - No Python in the inference hot path; this script is offline tooling.
#   - Revert on correctness failure even if tok/s improves.

set -uo pipefail
IFS=$'\n\t'

# ─────────────────────────────────────────────────────────────────────────────
# Configuration — all overridable via environment variables
# ─────────────────────────────────────────────────────────────────────────────
ARCH="${ARCH:-gfx1201}"
MODEL="${MODEL:-qwen3.5:27b}"
MAX_ITERS="${MAX_ITERS:-100}"
ACCEPT_MIN_SPEEDUP="${ACCEPT_MIN_SPEEDUP:-1.01}"
AUTOCOMMIT="${AUTOCOMMIT:-0}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-1200}"       # 20 min per operation
MAX_ATTEMPTS_PER_TARGET="${MAX_ATTEMPTS_PER_TARGET:-5}"
BENCH_TRIALS="${BENCH_TRIALS:-3}"
ALLOW_OTHER_ARCH="${ALLOW_OTHER_ARCH:-0}"
RUN_FINAL_VALIDATE="${RUN_FINAL_VALIDATE:-0}"
CRASH_STRATEGY_LIMIT="${CRASH_STRATEGY_LIMIT:-3}"

# Phase 14 — Real kernel-authoring mode
# Set AUTHOR_KERNEL=1 to generate new .hip kernel candidates and evaluate them
# in isolated harnesses before wiring into hipfire. Uses author_kernel.sh.
AUTHOR_KERNEL="${AUTHOR_KERNEL:-0}"
KERNEL_LAB_DIR="${KERNEL_LAB_DIR:-}"   # set in main() after TOOL_DIR is resolved

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TOOL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$TOOL_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Add cargo and ROCm to PATH if not already present
export PATH="/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.cargo/bin:/opt/rocm/bin:${PATH}"

WORKSPACE="$TOOL_DIR/workspace"
CANDIDATES_DIR="$WORKSPACE/candidates"
ACCEPTED_DIR="$WORKSPACE/accepted"
REJECTED_DIR="$WORKSPACE/rejected"
STATE_FILE="$WORKSPACE/orchestration_state.json"
PLAN_FILE="$WORKSPACE/optimization_plan.json"
PROFILE_FILE="$WORKSPACE/profile_report.json"
RESULTS_TSV="$TOOL_DIR/results.tsv"
REPORTS_DIR="$TOOL_DIR/reports"
BASELINES_DIR="$TOOL_DIR/baselines"
KERNELS_SRC="$REPO_ROOT/kernels/src"
BENCH_EXE="$REPO_ROOT/target/release/examples/bench_qwen35_mq4"
BENCH_PROMPT="${HIPFIRE_BENCH_PROMPT:-$REPO_ROOT/benchmarks/prompts/lru_cache_pep8_strict.txt}"
STOP_FLAG="$REPO_ROOT/.autokernel_stop"

# ─────────────────────────────────────────────────────────────────────────────
# Colours
# ─────────────────────────────────────────────────────────────────────────────
if [ -t 2 ]; then
    RED='\033[0;31m'; GRN='\033[0;32m'; YLW='\033[0;33m'
    BLU='\033[0;34m'; CYN='\033[0;36m'; RST='\033[0m'
else
    RED=''; GRN=''; YLW=''; BLU=''; CYN=''; RST=''
fi
log()  { echo -e "${BLU}[loop]${RST} $*" >&2; }
ok()   { echo -e "${GRN}[loop OK]${RST} $*" >&2; }
warn() { echo -e "${YLW}[loop WARN]${RST} $*" >&2; }
err()  { echo -e "${RED}[loop ERR]${RST} $*" >&2; }
hdr()  { echo -e "${CYN}══ $* ══${RST}" >&2; }

# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
ts_now()  { date -u +%Y%m%d-%H%M%S; }
ts_iso()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
git_sha() { git -C "$REPO_ROOT" rev-parse HEAD 2>/dev/null || echo "unknown"; }

file_md5() {
    if command -v md5sum >/dev/null 2>&1; then
        md5sum "$1" 2>/dev/null | cut -d' ' -f1
    else
        md5 -q "$1" 2>/dev/null || echo "unknown"
    fi
}

float_ge()  { awk "BEGIN{exit !($1 >= $2)}"; }
float_div() { awk "BEGIN{printf \"%.4f\", ($2 != 0) ? $1/$2 : 0}"; }

median() {
    local nums=("$@")
    local n="${#nums[@]}"
    [ "$n" -eq 0 ] && echo "0" && return
    printf '%s\n' "${nums[@]}" | sort -n | awk -v n="$n" '
        NR == int((n+1)/2) { low = $1 }
        NR == int((n+2)/2) { high = $1 }
        END { print (low + high) / 2 }'
}

stddev() {
    local nums=("$@")
    local n="${#nums[@]}"
    [ "$n" -lt 2 ] && echo "0" && return
    printf '%s\n' "${nums[@]}" | awk -v n="$n" '
        { sum += $1; sumsq += $1*$1 }
        END { mean=sum/n; print sqrt(sumsq/n - mean*mean) }'
}

parse_gen_tok_s()     { grep -oP 'gen_tok_s=\K[\d.]+' "$1" 2>/dev/null | tail -1 || echo "0"; }
parse_prefill_tok_s() { grep -oP 'prefill_tok_s=\K[\d.]+' "$1" 2>/dev/null | tail -1 || echo "0"; }
parse_vram_mb()       { grep -oP 'vram_mb=\K[\d.]+' "$1" 2>/dev/null | tail -1 || echo "0"; }

# ─────────────────────────────────────────────────────────────────────────────
# Model resolution
# ─────────────────────────────────────────────────────────────────────────────
if [ -d "$HOME/.hipfire/models" ] && ls "$HOME/.hipfire/models"/*.mq4 &>/dev/null 2>&1; then
    MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
elif [ -d "/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.hipfire/models" ]; then
    MODELS_DIR="${HIPFIRE_MODELS_DIR:-/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.hipfire/models}"
else
    MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
fi

resolve_model_path() {
    local tag="$1"
    local size; size=$(echo "$tag" | grep -oP '\d+b' | head -1)
    local candidates=(
        "$MODELS_DIR/qwen3.5-27b.mq4"
        "$MODELS_DIR/qwen35-27b.mq4"
        "$MODELS_DIR/qwen3.5-9b.mq4"
        "$MODELS_DIR/qwen35-9b.mq4"
    )
    for f in "$MODELS_DIR"/*.mq4; do
        [ -f "$f" ] && candidates+=("$f")
    done
    for c in "${candidates[@]}"; do
        if [ -f "$c" ]; then
            if [ -n "$size" ] && echo "$c" | grep -qi "$size"; then
                echo "$c"; return 0
            fi
        fi
    done
    for c in "${candidates[@]}"; do [ -f "$c" ] && echo "$c" && return 0; done
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Bench runner
# ─────────────────────────────────────────────────────────────────────────────
run_bench_once() {
    local outfile; outfile="$(mktemp /tmp/autokernel-bench-XXXXXX.txt)"
    local model_path; model_path="$(resolve_model_path "$MODEL")"
    if [ -z "$model_path" ]; then
        warn "No model file found for $MODEL in $MODELS_DIR"
        echo "$outfile"; return 1
    fi
    if timeout "$TIMEOUT_SECONDS" "$BENCH_EXE" "$model_path" --gen 80 --warmup 10 \
            >"$outfile" 2>&1; then
        echo "$outfile"; return 0
    else
        echo "$outfile"; return 1
    fi
}

run_bench_trials() {
    local n="$1"
    local gen_vals=() prefill_vals=() vram_vals=()
    for ((i=1; i<=n; i++)); do
        log "  bench trial $i/$n..."
        local outfile; outfile="$(run_bench_once)" || { warn "trial $i failed"; continue; }
        gen_vals+=("$(parse_gen_tok_s "$outfile")")
        prefill_vals+=("$(parse_prefill_tok_s "$outfile")")
        vram_vals+=("$(parse_vram_mb "$outfile")")
        rm -f "$outfile"
    done
    local med_gen med_pre sd_gen sd_pre med_vram
    med_gen="$(median "${gen_vals[@]:-0}")"
    med_pre="$(median "${prefill_vals[@]:-0}")"
    sd_gen="$(stddev "${gen_vals[@]:-0}")"
    sd_pre="$(stddev "${prefill_vals[@]:-0}")"
    med_vram="$(median "${vram_vals[@]:-0}")"
    echo "$med_gen $med_pre $sd_gen $sd_pre $med_vram"
}

# ─────────────────────────────────────────────────────────────────────────────
# Build
# ─────────────────────────────────────────────────────────────────────────────
rebuild_bench() {
    log "Building bench_qwen35_mq4 (removing stale binary first)..."
    rm -f "$BENCH_EXE"
    timeout "$((TIMEOUT_SECONDS * 2))" \
        cargo build --release --example bench_qwen35_mq4 \
        --features deltanet -p hipfire-runtime 2>&1
}

# ─────────────────────────────────────────────────────────────────────────────
# Orchestration state — JSON read/write via python3 (offline tooling)
# ─────────────────────────────────────────────────────────────────────────────
init_state() {
    mkdir -p "$WORKSPACE" "$CANDIDATES_DIR" "$ACCEPTED_DIR" "$REJECTED_DIR"
    if [ ! -f "$STATE_FILE" ]; then
        log "Initializing orchestration state..."
        python3 - <<PYEOF
import json, os, sys

plan_file = "$PLAN_FILE"
state_file = "$STATE_FILE"

target_queue = []
if os.path.exists(plan_file):
    try:
        plan = json.load(open(plan_file))
        target_queue = [t["target"] for t in plan.get("targets", [])]
    except Exception:
        pass

if not target_queue:
    # Default priority queue for 27B decode optimization on gfx1201.
    # Order by Amdahl decode impact (highest first).
    target_queue = [
        "gemv_hfq4g256",
        "gemv_hfq6g256",
        "gemv_hfq4g256_residual",
        "gemv_hfq6g256_residual",
        "fused_qkv_hfq4g256",
        "fused_gate_up_hfq4g256",
        "attention_flash_asym3_tile",
        "fused_rmsnorm_mq_rotate",
        "rmsnorm",
    ]

state = {
    "version": 1,
    "current_target": target_queue[0] if target_queue else "",
    "target_queue": target_queue,
    "accepted_candidates": [],
    "rejected_candidates": [],
    "best_tok_s": 0.0,
    "baseline_tok_s": 0.0,
    "best_speedup": 1.0,
    "attempts_per_target": {},
    "strategies_tried": {},
    "crash_count": 0,
    "timeout_count": 0,
    "plateau_count": 0,
    "last_decision": "INIT",
    "next_action": "CONTINUE",
    "total_iterations": 0,
}
with open(state_file, 'w') as f:
    json.dump(state, f, indent=2)
print("OK: state initialized")
PYEOF
    fi
}

get_state() {
    python3 -c "
import json, sys
try:
    d = json.load(open('$STATE_FILE'))
    v = d.get('$1', '')
    print(v if not isinstance(v, (list, dict)) else '')
except:
    print('')
" 2>/dev/null || echo ""
}

update_state() {
    # update_state key=value [key=value ...]
    python3 - "$STATE_FILE" "$@" <<'PYEOF'
import json, sys

state_file = sys.argv[1]
updates = {}
for arg in sys.argv[2:]:
    if '=' in arg:
        k, v = arg.split('=', 1)
        try:    v = int(v)
        except:
            try:    v = float(v)
            except:
                if v == 'true':  v = True
                elif v == 'false': v = False
        updates[k] = v

try:
    with open(state_file) as f:
        state = json.load(f)
    state.update(updates)
    with open(state_file, 'w') as f:
        json.dump(state, f, indent=2)
except Exception as e:
    print(f"update_state error: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
}

append_accepted() {
    python3 - "$STATE_FILE" "$1" <<'PYEOF'
import json, sys
state_file, candidate = sys.argv[1], sys.argv[2]
with open(state_file) as f:
    d = json.load(f)
d.setdefault('accepted_candidates', [])
if candidate not in d['accepted_candidates']:
    d['accepted_candidates'].append(candidate)
with open(state_file, 'w') as f:
    json.dump(d, f, indent=2)
PYEOF
}

append_rejected() {
    python3 - "$STATE_FILE" "$1" <<'PYEOF'
import json, sys
state_file, candidate = sys.argv[1], sys.argv[2]
with open(state_file) as f:
    d = json.load(f)
d.setdefault('rejected_candidates', [])
if candidate not in d['rejected_candidates']:
    d['rejected_candidates'].append(candidate)
with open(state_file, 'w') as f:
    json.dump(d, f, indent=2)
PYEOF
}

increment_attempts() {
    python3 - "$STATE_FILE" "$1" <<'PYEOF'
import json, sys
state_file, target = sys.argv[1], sys.argv[2]
with open(state_file) as f:
    d = json.load(f)
d.setdefault('attempts_per_target', {})
d['attempts_per_target'][target] = d['attempts_per_target'].get(target, 0) + 1
with open(state_file, 'w') as f:
    json.dump(d, f, indent=2)
PYEOF
}

record_strategy_tried() {
    python3 - "$STATE_FILE" "$1" "$2" <<'PYEOF'
import json, sys
state_file, target, strategy = sys.argv[1], sys.argv[2], sys.argv[3]
with open(state_file) as f:
    d = json.load(f)
d.setdefault('strategies_tried', {})
d['strategies_tried'].setdefault(target, [])
if strategy not in d['strategies_tried'][target]:
    d['strategies_tried'][target].append(strategy)
with open(state_file, 'w') as f:
    json.dump(d, f, indent=2)
PYEOF
}

get_attempts() {
    python3 -c "
import json
try:
    d = json.load(open('$STATE_FILE'))
    print(d.get('attempts_per_target', {}).get('$1', 0))
except:
    print(0)
" 2>/dev/null || echo "0"
}

get_strategies_tried() {
    python3 -c "
import json
try:
    d = json.load(open('$STATE_FILE'))
    tried = d.get('strategies_tried', {}).get('$1', [])
    for s in tried: print(s)
except:
    pass
" 2>/dev/null || true
}

advance_target() {
    python3 - "$STATE_FILE" "$MAX_ATTEMPTS_PER_TARGET" <<'PYEOF'
import json, sys
state_file = sys.argv[1]
max_attempts = int(sys.argv[2])

with open(state_file) as f:
    d = json.load(f)

queue = d.get('target_queue', [])
current = d.get('current_target', '')

# Rotate: move current to end, pick next under max_attempts
if current in queue:
    idx = queue.index(current)
    new_queue = queue[idx+1:] + queue[:idx] + [current]
else:
    new_queue = queue

new_target = ''
for t in new_queue:
    attempts = d.get('attempts_per_target', {}).get(t, 0)
    if attempts < max_attempts:
        new_target = t
        break

if not new_target:
    d['next_action'] = 'DONE'
    d['current_target'] = ''
else:
    d['current_target'] = new_target
    d['target_queue'] = new_queue
    d['plateau_count'] = d.get('plateau_count', 0) + 1
    d['next_action'] = 'NEXT'

with open(state_file, 'w') as f:
    json.dump(d, f, indent=2)
print(new_target)
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Strategy library
# ─────────────────────────────────────────────────────────────────────────────
# Returns the next untried strategy name for a given kernel target.
pick_strategy() {
    local target="$1"
    local already_tried
    already_tried="$(get_strategies_tried "$target" 2>/dev/null | sort | tr '\n' '|')"

    local strategies=()

    # Phase 14: author-kernel mode — return LLM-authoring strategy names.
    # author_kernel.sh must have been sourced before pick_strategy is called.
    if [ "${AUTHOR_KERNEL:-0}" = "1" ]; then
        local strats_var="AUTHOR_STRATEGIES_$(echo "$target" | tr '-' '_')"
        if declare -p "$strats_var" &>/dev/null 2>&1; then
            eval "strategies=(\"\${${strats_var}[@]}\")"
        else
            # Fallback authoring strategies if target-specific list is not defined
            strategies=("author_gemv_lds_staging_x" "author_gemv_uint4_loads"
                        "author_gemv_8acc_ilp" "author_gemv_scale_broadcast")
        fi
        for s in "${strategies[@]}"; do
            if ! echo "|${already_tried}" | grep -qF "|${s}|"; then
                echo "$s"; return 0
            fi
        done
        echo "EXHAUSTED"; return 1
    fi

    case "$target" in
        gemv_hfq4g256|gemv_hfq6g256)
            # Pure memory-bandwidth GEMV: tune launch bounds, try small occupancy changes
            strategies=("launch_w32_o12" "launch_w32_o8" "launch_w32_o16"
                        "launch_w32_o24" "unroll_pragma8" "unroll_pragma16"
                        "restrict_qualifiers")
            ;;
        gemv_hfq4g256_residual|gemv_hfq6g256_residual)
            strategies=("launch_w32_o12" "launch_w32_o8" "launch_w32_o16"
                        "launch_w32_o24" "unroll_pragma8" "restrict_qualifiers")
            ;;
        fused_qkv_hfq4g256|fused_qkvza_hfq4g256|fused_qkv*)
            strategies=("launch_w32_o12" "launch_w32_o8" "launch_w32_o16"
                        "unroll_pragma8" "restrict_qualifiers")
            ;;
        fused_gate_up_hfq4g256|fused_gate_up_hfq6g256|fused_gate_up*)
            strategies=("launch_w32_o12" "launch_w32_o8" "launch_w32_o16"
                        "unroll_pragma8" "restrict_qualifiers")
            ;;
        attention_flash_asym3_tile|attention_flash*)
            # Attention: has arithmetic component; more latitude for tuning
            strategies=("launch_w32_o8" "launch_w32_o12" "launch_w32_o16"
                        "launch_w64_o4" "unroll_pragma4" "restrict_qualifiers")
            ;;
        rmsnorm|fused_rmsnorm*)
            strategies=("launch_w32_o16" "launch_w32_o24" "unroll_pragma4"
                        "restrict_qualifiers" "launch_w32_o8")
            ;;
        *)
            strategies=("launch_w32_o12" "launch_w32_o8" "launch_w32_o16"
                        "unroll_pragma8" "restrict_qualifiers")
            ;;
    esac

    for s in "${strategies[@]}"; do
        if ! echo "|${already_tried}" | grep -qF "|${s}|"; then
            echo "$s"; return 0
        fi
    done
    echo "EXHAUSTED"; return 1
}

# Apply a named strategy mutation to a kernel source file.
# Returns 0 if a change was made, 1 if the pattern was not applicable.
apply_strategy() {
    local strategy="$1"
    local variant_file="$2"

    case "$strategy" in
        launch_w32_o8)
            if grep -q '__launch_bounds__' "$variant_file"; then
                sed -i 's/__launch_bounds__([0-9][0-9]*, *[0-9][0-9]*)/__launch_bounds__(32, 8)/g' "$variant_file"
            else
                sed -i 's/^__global__ void /__attribute__((launch_bounds(32, 8))) __global__ void /g' "$variant_file"
            fi
            return 0
            ;;
        launch_w32_o12)
            if grep -q '__launch_bounds__' "$variant_file"; then
                sed -i 's/__launch_bounds__([0-9][0-9]*, *[0-9][0-9]*)/__launch_bounds__(32, 12)/g' "$variant_file"
            else
                sed -i 's/^__global__ void /__attribute__((launch_bounds(32, 12))) __global__ void /g' "$variant_file"
            fi
            return 0
            ;;
        launch_w32_o16)
            if grep -q '__launch_bounds__' "$variant_file"; then
                sed -i 's/__launch_bounds__([0-9][0-9]*, *[0-9][0-9]*)/__launch_bounds__(32, 16)/g' "$variant_file"
            else
                sed -i 's/^__global__ void /__attribute__((launch_bounds(32, 16))) __global__ void /g' "$variant_file"
            fi
            return 0
            ;;
        launch_w32_o24)
            if grep -q '__launch_bounds__' "$variant_file"; then
                sed -i 's/__launch_bounds__([0-9][0-9]*, *[0-9][0-9]*)/__launch_bounds__(32, 24)/g' "$variant_file"
            else
                sed -i 's/^__global__ void /__attribute__((launch_bounds(32, 24))) __global__ void /g' "$variant_file"
            fi
            return 0
            ;;
        launch_w64_o4)
            if grep -q '__launch_bounds__' "$variant_file"; then
                sed -i 's/__launch_bounds__([0-9][0-9]*, *[0-9][0-9]*)/__launch_bounds__(64, 4)/g' "$variant_file"
                return 0
            fi
            return 1
            ;;
        unroll_pragma4)
            # Insert #pragma unroll 4 before the first inner for-loop using int g or int k
            if grep -qP 'for\s*\(int\s+(g|k|i)\s*=' "$variant_file"; then
                # Remove any existing pragma unroll first (idempotent)
                sed -i '/#pragma unroll/d' "$variant_file"
                sed -i '/for\s*(int\s*\(g\|k\|i\)\s*=/i\#pragma unroll 4' "$variant_file"
                return 0
            fi
            return 1
            ;;
        unroll_pragma8)
            if grep -qP 'for\s*\(int\s+(g|k|i)\s*=' "$variant_file"; then
                sed -i '/#pragma unroll/d' "$variant_file"
                sed -i '/for\s*(int\s*\(g\|k\|i\)\s*=/i\#pragma unroll 8' "$variant_file"
                return 0
            fi
            return 1
            ;;
        unroll_pragma16)
            if grep -qP 'for\s*\(int\s+(g|k|i)\s*=' "$variant_file"; then
                sed -i '/#pragma unroll/d' "$variant_file"
                sed -i '/for\s*(int\s*\(g\|k\|i\)\s*=/i\#pragma unroll 16' "$variant_file"
                return 0
            fi
            return 1
            ;;
        restrict_qualifiers)
            # Add __restrict__ to pointer parameters — enables better load coalescing
            # Only meaningful if not already present
            if grep -q '__restrict__' "$variant_file"; then
                return 1  # Already applied
            fi
            # Add to const pointer params in kernel signatures
            sed -i 's/const \(float\|half\|uint8_t\|uint32_t\|unsigned char\)\s*\*/const \1* __restrict__/g' \
                "$variant_file" 2>/dev/null || true
            # Check if we changed anything
            if grep -q '__restrict__' "$variant_file"; then
                return 0
            fi
            return 1
            ;;
        *)
            warn "Unknown strategy: $strategy — skipping"
            return 1
            ;;
    esac
}

# ─────────────────────────────────────────────────────────────────────────────
# TSV logging — appends one row to results.tsv
# Phase 13 schema extends Phase 12 with 7 additional columns at the end.
# Phase 14 schema extends Phase 13 with 9 additional columns (author-kernel mode).
# ─────────────────────────────────────────────────────────────────────────────
log_tsv_row() {
    # Variadic: 39 fields (Phase 13) or 48 fields (Phase 14). Trailing fields blank for older rows.
    local fields=("$@")
    local row; row=$(printf '%s\t' "${fields[@]}")
    row="${row%$'\t'}"
    echo "$row" >> "$RESULTS_TSV"
}

# ─────────────────────────────────────────────────────────────────────────────
# Baseline capture
# ─────────────────────────────────────────────────────────────────────────────
ensure_baseline() {
    # Returns cached or freshly-measured baseline decode tok/s.
    # Bug fix: must select baseline matching current MODEL size, not just newest file.
    local _model_sz; _model_sz=$(echo "$MODEL" | grep -oP '\d+b' | head -1 || echo "")
    local latest_json=""
    while IFS= read -r _f; do
        [[ -f "$_f" ]] || continue
        local _bm
        _bm=$(python3 -c "import json; print(json.load(open('$_f')).get('model',''))" 2>/dev/null || echo "")
        if [[ -n "$_model_sz" && "$_bm" == *"$_model_sz"* ]]; then
            latest_json="$_f"
            break
        fi
    done < <(ls -t "$BASELINES_DIR"/*.json 2>/dev/null)
    if [ -n "$latest_json" ]; then
        local bl
        bl="$(grep -oP '"baseline_decode_tok_s":\s*\K[\d.]+' "$latest_json" 2>/dev/null || echo "0")"
        if float_ge "$bl" "1.0"; then
            log "Using model-matched baseline ($MODEL): $bl tok/s ($latest_json)"
            echo "$bl"; return 0
        fi
    fi
    log "No cached baseline — running baseline via run.sh..."
    local bl_json
    bl_json="$("$TOOL_DIR/run.sh" baseline \
        --arch "$ARCH" \
        --model "$MODEL" \
        --trials "$BENCH_TRIALS" \
        $( [ "${ALLOW_OTHER_ARCH:-0}" = "1" ] && echo "--allow-other-arch" ) \
        2>&1 | tail -1)"
    if [ -f "$bl_json" ]; then
        local bl; bl="$(grep -oP '"baseline_decode_tok_s":\s*\K[\d.]+' "$bl_json" || echo "0")"
        echo "$bl"; return 0
    fi
    warn "run.sh baseline failed — falling back to direct bench..."
    rebuild_bench >/dev/null 2>&1 || true
    local bench_out; bench_out="$(run_bench_trials "$BENCH_TRIALS")"
    IFS=' ' read -r med_gen _rest <<< "$bench_out"
    echo "${med_gen:-0}"
}

# ─────────────────────────────────────────────────────────────────────────────
# Candidate workspace artifact writer
# ─────────────────────────────────────────────────────────────────────────────
write_candidate_metadata() {
    local cand_dir="$1" iter="$2" target="$3" strategy="$4" variant_file="$5"
    local base_sha="$6" cand_sha="$7" decision="$8" revert_reason="$9"
    local baseline_tok_s="${10}" cand_tok_s="${11}" decode_speedup="${12}"
    local build_status="${13}" correctness_status="${14}" bench_status="${15}"

    cat > "$cand_dir/metadata.json" <<EOF
{
  "iteration": $iter,
  "target": "$target",
  "strategy": "$strategy",
  "variant_file": "kernels/src/$(basename "$variant_file")",
  "timestamp": "$(ts_iso)",
  "base_sha": "$base_sha",
  "candidate_sha": "$cand_sha",
  "model": "$MODEL",
  "arch": "$ARCH",
  "baseline_tok_s": $baseline_tok_s,
  "candidate_tok_s": $cand_tok_s,
  "decode_speedup": $decode_speedup,
  "accept_threshold": $ACCEPT_MIN_SPEEDUP,
  "build_status": "$build_status",
  "correctness_status": "$correctness_status",
  "bench_status": "$bench_status",
  "decision": "$decision",
  "revert_reason": "$revert_reason"
}
EOF
}

# ─────────────────────────────────────────────────────────────────────────────
# Final report
# ─────────────────────────────────────────────────────────────────────────────
write_final_report() {
    local ts; ts="$(ts_now)"
    local report_md="$REPORTS_DIR/phase13_autokernel_loop_${ts}.md"
    local report_json="$REPORTS_DIR/phase13_autokernel_loop_${ts}.json"
    local accepted_count rejected_count total_iters baseline best_tok best_speedup
    local crash_count timeout_count plateau_count

    accepted_count="$(python3 -c "
import json; d=json.load(open('$STATE_FILE'))
print(len(d.get('accepted_candidates',[])))
" 2>/dev/null || echo "0")"

    rejected_count="$(python3 -c "
import json; d=json.load(open('$STATE_FILE'))
print(len(d.get('rejected_candidates',[])))
" 2>/dev/null || echo "0")"

    total_iters="$(get_state total_iterations)"
    baseline="$(get_state baseline_tok_s)"
    best_tok="$(get_state best_tok_s)"
    best_speedup="$(get_state best_speedup)"
    crash_count="$(get_state crash_count)"
    timeout_count="$(get_state timeout_count)"
    plateau_count="$(get_state plateau_count)"

    local next_target
    next_target="$(python3 -c "
import json
d=json.load(open('$STATE_FILE'))
q=d.get('target_queue',[])
att=d.get('attempts_per_target',{})
for t in q:
    if att.get(t,0) < $MAX_ATTEMPTS_PER_TARGET:
        print(t); break
else:
    print('all-exhausted')
" 2>/dev/null || echo "unknown")"

    cat > "$report_md" <<MDEOF
# AutoKernel Loop Report — Phase 13

Generated: $(ts_iso)

## Run Summary

| Field | Value |
|-------|-------|
| Model | \`$MODEL\` |
| Arch | $ARCH |
| Total iterations | $total_iters |
| Baseline tok/s | $baseline |
| Best tok/s | $best_tok |
| Best speedup | ${best_speedup}x |
| Accepted candidates | $accepted_count |
| Rejected/reverted | $rejected_count |
| Crashes | $crash_count |
| Timeouts | $timeout_count |
| Targets advanced (plateau) | $plateau_count |

## Accepted Candidates

$(python3 -c "
import json
d=json.load(open('$STATE_FILE'))
acc=d.get('accepted_candidates',[])
print('\n'.join(f'- \`{a}\`' for a in acc) if acc else '_(none)_')
" 2>/dev/null || echo "_(none)_")

## Rejected / Exhausted Candidates

$(python3 -c "
import json
d=json.load(open('$STATE_FILE'))
rej=d.get('rejected_candidates',[])
lines='\n'.join(f'- {r}' for r in rej[:30])
extra=len(rej)-30
if extra>0: lines+=f'\n- ...and {extra} more'
print(lines if rej else '_(none)_')
" 2>/dev/null || echo "_(none)_")

## Next Recommended Target

\`$next_target\`

## How to Rerun

\`\`\`bash
ARCH=$ARCH MODEL=$MODEL MAX_ITERS=$MAX_ITERS ACCEPT_MIN_SPEEDUP=$ACCEPT_MIN_SPEEDUP \\
  ./tools/autokernel-rdna/autokernel_loop.sh
\`\`\`

## How to Rollback Accepted Changes

\`\`\`bash
# Show accepted commits:
git log --oneline --grep='autokernel: optimize'
# Revert one:
git revert <sha>
# Or revert all kernel source files to a known-good SHA:
git checkout <base-sha> -- kernels/src/
\`\`\`

## Final Validation

$([ "$accepted_count" -gt "0" ] && \
    echo "At least one candidate was accepted. Run final validation:" || \
    echo "No accepted candidates — skip validation.")

\`\`\`bash
TRIALS=5 MODEL=$MODEL ./tools/autokernel-rdna/phase11_validate.sh
\`\`\`

## Full Results Log

\`tools/autokernel-rdna/results.tsv\`

MDEOF

    python3 - <<PYEOF
import json, sys

with open("$STATE_FILE") as f:
    d = json.load(f)

report = {
    "timestamp": "$(ts_iso)",
    "model": "$MODEL",
    "arch": "$ARCH",
    "max_iters": $MAX_ITERS,
    "accept_min_speedup": $ACCEPT_MIN_SPEEDUP,
    "total_iterations": d.get("total_iterations", 0),
    "baseline_tok_s": d.get("baseline_tok_s", 0),
    "best_tok_s": d.get("best_tok_s", 0),
    "best_speedup": d.get("best_speedup", 1.0),
    "accepted_candidates": d.get("accepted_candidates", []),
    "rejected_count": len(d.get("rejected_candidates", [])),
    "crash_count": d.get("crash_count", 0),
    "timeout_count": d.get("timeout_count", 0),
    "plateau_count": d.get("plateau_count", 0),
    "next_recommended_target": "$next_target",
    "rerun_command": "ARCH=$ARCH MODEL=$MODEL MAX_ITERS=$MAX_ITERS ACCEPT_MIN_SPEEDUP=$ACCEPT_MIN_SPEEDUP ./tools/autokernel-rdna/autokernel_loop.sh",
}
with open("$report_json", "w") as f:
    json.dump(report, f, indent=2)
print("$report_json")
PYEOF

    ok "Final report: $report_md"
    ok "Final JSON:   $report_json"

    if [ "${accepted_count:-0}" -gt "0" ]; then
        echo "" >&2
        echo "  ┌────────────────────────────────────────────────────────────────────┐" >&2
        echo "  │  Candidates accepted. Run final validation when ready:             │" >&2
        echo "  │                                                                    │" >&2
        echo "  │  TRIALS=5 MODEL=$MODEL ./tools/autokernel-rdna/phase11_validate.sh │" >&2
        echo "  └────────────────────────────────────────────────────────────────────┘" >&2
        echo "" >&2
        if [ "${RUN_FINAL_VALIDATE:-0}" = "1" ]; then
            log "RUN_FINAL_VALIDATE=1 — running phase11_validate.sh..."
            TRIALS=5 MODEL="$MODEL" "$TOOL_DIR/phase11_validate.sh" \
                || warn "Final validation completed with warnings"
        fi
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
# Signal handling — clean stop on Ctrl-C / SIGTERM
# ─────────────────────────────────────────────────────────────────────────────
LOOP_RUNNING=0
_cleanup() {
    if [ "$LOOP_RUNNING" = "1" ]; then
        echo "" >&2
        warn "Signal received — finishing current iteration then writing final report..."
        LOOP_RUNNING=2
    fi
}
trap _cleanup SIGINT SIGTERM

# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
main() {
    hdr "AutoKernel Loop — Phase 13/14  (AUTHOR_KERNEL=${AUTHOR_KERNEL})"
    log "ARCH=$ARCH  MODEL=$MODEL  MAX_ITERS=$MAX_ITERS"
    log "ACCEPT_MIN_SPEEDUP=$ACCEPT_MIN_SPEEDUP  AUTOCOMMIT=$AUTOCOMMIT  TIMEOUT=${TIMEOUT_SECONDS}s"
    echo "" >&2

    # Resolve KERNEL_LAB_DIR now that TOOL_DIR is available
    KERNEL_LAB_DIR="${KERNEL_LAB_DIR:-$TOOL_DIR/kernel_lab}"

    # Phase 14: source author_kernel.sh to make author_kernel_iteration() available
    if [ "${AUTHOR_KERNEL:-0}" = "1" ]; then
        local author_script="$TOOL_DIR/author_kernel.sh"
        if [ ! -f "$author_script" ]; then
            err "AUTHOR_KERNEL=1 but author_kernel.sh not found at: $author_script"
            exit 1
        fi
        # shellcheck source=author_kernel.sh
        source "$author_script"
        log "Phase 14 author-kernel mode ACTIVE. kernel_lab: $KERNEL_LAB_DIR"
    fi

    # ── Setup ────────────────────────────────────────────────────────────────
    mkdir -p "$WORKSPACE" "$CANDIDATES_DIR" "$ACCEPTED_DIR" "$REJECTED_DIR" "$REPORTS_DIR"
    init_state

    # Check for DONE state from a previous interrupted run
    local next_action; next_action="$(get_state next_action)"
    if [ "$next_action" = "DONE" ]; then
        warn "Orchestration state is DONE (all targets exhausted)."
        warn "To reset: rm $STATE_FILE"
        write_final_report
        exit 0
    fi

    # ── Verify prompt ────────────────────────────────────────────────────────
    if [ ! -f "$BENCH_PROMPT" ]; then
        err "Bench prompt not found: $BENCH_PROMPT"
        exit 1
    fi
    local prompt_md5; prompt_md5="$(file_md5 "$BENCH_PROMPT")"
    local prompt_name; prompt_name="$(basename "$BENCH_PROMPT")"

    # ── Initial build ────────────────────────────────────────────────────────
    log "Initial build to verify clean state..."
    if ! rebuild_bench 2>&1 | tail -3; then
        err "Initial build failed — cannot proceed. Check cargo/ROCm setup."
        exit 1
    fi

    # ── Baseline ────────────────────────────────────────────────────────────
    local baseline_tok_s
    baseline_tok_s="$(ensure_baseline)"
    if ! float_ge "${baseline_tok_s:-0}" "1.0"; then
        err "Baseline measurement failed (got '${baseline_tok_s}' tok/s). Check model path."
        exit 1
    fi
    ok "Baseline: $baseline_tok_s tok/s  (model=$MODEL, arch=$ARCH)"

    # Persist baseline in state if not already set
    local st_baseline; st_baseline="$(get_state baseline_tok_s)"
    if ! float_ge "${st_baseline:-0}" "1.0"; then
        update_state "baseline_tok_s=$baseline_tok_s" "best_tok_s=$baseline_tok_s"
    fi
    local current_best; current_best="$(get_state best_tok_s)"
    if ! float_ge "${current_best:-0}" "1.0"; then
        current_best="$baseline_tok_s"
        update_state "best_tok_s=$baseline_tok_s"
    fi

    # ── Main optimization loop ───────────────────────────────────────────────
    LOOP_RUNNING=1
    local iter=0
    local consecutive_failures=0

    while [ "$iter" -lt "$MAX_ITERS" ] && [ "$LOOP_RUNNING" = "1" ]; do

        # ── Stop flag ────────────────────────────────────────────────────────
        if [ -f "$STOP_FLAG" ]; then
            warn "Stop flag detected (.autokernel_stop) — exiting loop cleanly"
            rm -f "$STOP_FLAG"
            break
        fi

        iter=$((iter + 1))
        local ts_iter; ts_iter="$(ts_now)"
        local base_sha; base_sha="$(git_sha)"

        # ── Pick current target ──────────────────────────────────────────────
        local current_target; current_target="$(get_state current_target)"
        if [ -z "$current_target" ]; then
            warn "No current target — advancing..."
            current_target="$(advance_target)"
            [ -z "$current_target" ] && { warn "No more targets available — done"; break; }
        fi

        # Check if this target has hit max attempts
        local attempts; attempts="$(get_attempts "$current_target")"
        if [ "$attempts" -ge "$MAX_ATTEMPTS_PER_TARGET" ]; then
            log "Target '$current_target' exhausted ($attempts/$MAX_ATTEMPTS_PER_TARGET) — advancing"
            current_target="$(advance_target)"
            update_state "current_target=$current_target"
            [ -z "$current_target" ] && { warn "All targets exhausted — done"; break; }
            attempts="$(get_attempts "$current_target")"
        fi

        hdr "Iter $iter/$MAX_ITERS — target: $current_target  attempt $((attempts+1))/$MAX_ATTEMPTS_PER_TARGET"

        # ── Pick strategy ────────────────────────────────────────────────────
        local strategy
        strategy="$(pick_strategy "$current_target")"
        if [ "$strategy" = "EXHAUSTED" ]; then
            log "All strategies exhausted for '$current_target' — advancing"
            increment_attempts "$current_target"  # force max
            current_target="$(advance_target)"
            update_state "current_target=$current_target"
            consecutive_failures=$((consecutive_failures + 1))
            [ -z "$current_target" ] && break
            continue
        fi
        log "Strategy: $strategy"

        # ── Phase 14: author-kernel mode ─────────────────────────────────────
        if [ "${AUTHOR_KERNEL:-0}" = "1" ]; then
            # author_kernel_iteration() sets: AK_CORRECTNESS, AK_HARNESS_SPEEDUP,
            # AK_PROMOTED, AK_HIPFIRE_TOK_S, AK_DECISION, AK_CANDIDATE_PATH
            author_kernel_iteration "$current_target" "$iter" "$strategy"

            local ak_spec_path="kernel_lab/generated/${current_target}/target_spec.json"
            local ak_cand_path="${AK_CANDIDATE_PATH:-}"
            local ak_hipfire_before="$baseline_tok_s"
            local ak_hipfire_after="${AK_HIPFIRE_TOK_S:-0}"
            local ak_speedup="${AK_HARNESS_SPEEDUP:-0}"

            # TSV row (48 fields: 39 Phase-13 + 9 Phase-14)
            log_tsv_row \
                "exp-ph14-$(printf '%04d' $iter)" "$(ts_iso)" "$base_sha" "$(git_sha)" \
                "$ARCH" "" "" "$MODEL" "mq4" "q8" \
                "$prompt_name" "$prompt_md5" \
                "$current_target" "" "$strategy" \
                "SKIP" "SKIP" "SKIP" \
                "$baseline_tok_s" "${ak_hipfire_after}" "0" \
                "0" "0" "0" "0" "0" \
                "0" "0" "0" \
                "$AK_DECISION" "" "" \
                "$iter" "$current_target" "$strategy" "${ak_cand_path}" \
                "${AK_CORRECTNESS:-FAIL}" "$baseline_tok_s" "$current_best" \
                "${ak_hipfire_after}" "$ak_speedup" "1.0" \
                "1" "$ak_spec_path" "${ak_cand_path}" \
                "${AK_CORRECTNESS:-FAIL}" "$ak_speedup" \
                "${AK_PROMOTED:-0}" "$ak_hipfire_before" "$ak_hipfire_after" \
                "$AK_DECISION"

            record_strategy_tried "$current_target" "$strategy"
            increment_attempts "$current_target"
            update_state "total_iterations=$iter"

            if [ "${AK_DECISION:-SKIP}" = "ACCEPT" ]; then
                current_best="${ak_hipfire_after:-$current_best}"
                update_state "best_tok_s=$current_best" \
                             "best_speedup=$(float_div "$current_best" "$baseline_tok_s")" \
                             "next_action=CONTINUE" "last_decision=KEEP"
                consecutive_failures=0
                if [ "${AUTOCOMMIT:-0}" = "1" ]; then
                    git -C "$REPO_ROOT" add "kernels/src/${current_target}.gfx1201.hip" 2>/dev/null || true
                    git -C "$REPO_ROOT" commit -m "autokernel(ph14): author ${current_target} for $ARCH

Strategy: $strategy
Harness speedup: ${ak_speedup}x
E2E tok/s: ${ak_hipfire_before} -> ${ak_hipfire_after}
Iteration: $iter" 2>/dev/null || warn "git commit failed"
                fi
            else
                consecutive_failures=$((consecutive_failures + 1))
            fi

            local new_attempts_ak; new_attempts_ak="$(get_attempts "$current_target")"
            if [ "$new_attempts_ak" -ge "$MAX_ATTEMPTS_PER_TARGET" ]; then
                log "Max attempts for '$current_target' (author mode) — advancing"
                current_target="$(advance_target)"
                update_state "current_target=$current_target"
                [ -z "$current_target" ] && { warn "All targets exhausted"; break; }
            fi
            [ "$LOOP_RUNNING" = "2" ] && break
            continue
        fi

        # ── Resolve kernel variant file ──────────────────────────────────────
        local src_base="$KERNELS_SRC/$current_target"
        local variant_file=""
        if   [ -f "${src_base}.gfx1201.hip" ]; then
            variant_file="${src_base}.gfx1201.hip"
        elif [ -f "${src_base}.gfx12.hip" ]; then
            variant_file="${src_base}.gfx12.hip"
        elif [ -f "${src_base}.hip" ]; then
            # Bootstrap a new gfx12 variant from the base
            variant_file="${src_base}.gfx12.hip"
            cp "${src_base}.hip" "$variant_file"
            log "Created gfx12 variant bootstrapped from $(basename "${src_base}.hip")"
        else
            warn "No source file for kernel '$current_target' — skipping"
            record_strategy_tried "$current_target" "$strategy"
            increment_attempts "$current_target"
            continue
        fi
        local variant_basename; variant_basename="$(basename "$variant_file")"

        # ── Create candidate workspace directory ─────────────────────────────
        local cand_dir="$CANDIDATES_DIR/${iter}_${current_target}"
        mkdir -p "$cand_dir"
        cp "$variant_file" "$cand_dir/original_${variant_basename}"

        # ── Apply mutation ───────────────────────────────────────────────────
        cp "$variant_file" "${variant_file}.bak"
        if ! apply_strategy "$strategy" "$variant_file"; then
            warn "Strategy '$strategy' not applicable to $variant_basename — skipping"
            cp "${variant_file}.bak" "$variant_file"; rm -f "${variant_file}.bak"
            record_strategy_tried "$current_target" "$strategy"
            increment_attempts "$current_target"
            continue
        fi

        # Check if anything actually changed (avoid no-op iterations)
        if diff -q "$cand_dir/original_${variant_basename}" "$variant_file" >/dev/null 2>&1; then
            warn "Strategy '$strategy' produced no diff — skipping"
            cp "${variant_file}.bak" "$variant_file"; rm -f "${variant_file}.bak"
            record_strategy_tried "$current_target" "$strategy"
            increment_attempts "$current_target"
            continue
        fi

        # Save candidate snapshot and patch
        cp "$variant_file" "$cand_dir/candidate_${variant_basename}"
        diff "$cand_dir/original_${variant_basename}" \
             "$cand_dir/candidate_${variant_basename}" \
             > "$cand_dir/patch.diff" 2>/dev/null || true

        cat > "$cand_dir/candidate_notes.md" <<EOF
# Candidate: iter ${iter} — ${current_target} — ${strategy}

- Target: $current_target
- Strategy: $strategy
- File: kernels/src/$variant_basename
- Timestamp: $(ts_iso)
- Base SHA: $base_sha
- Model: $MODEL
- Arch: $ARCH
- Baseline: $baseline_tok_s tok/s
- Accept threshold: $ACCEPT_MIN_SPEEDUP x
EOF

        # ── Build ────────────────────────────────────────────────────────────
        local build_status="PASS"
        local build_log="$cand_dir/build.log"
        log "Building..."
        if ! rebuild_bench >"$build_log" 2>&1; then
            build_status="FAIL"
            err "Build FAILED — reverting"
            cp "${variant_file}.bak" "$variant_file"; rm -f "${variant_file}.bak"
            record_strategy_tried "$current_target" "$strategy"
            increment_attempts "$current_target"
            update_state "crash_count=$(($(get_state crash_count)+1))"
            consecutive_failures=$((consecutive_failures+1))
            cp -r "$cand_dir" "$REJECTED_DIR/" 2>/dev/null || true
            append_rejected "${current_target}:${strategy}:iter${iter}"
            write_candidate_metadata "$cand_dir" "$iter" "$current_target" "$strategy" \
                "$variant_file" "$base_sha" "$base_sha" "REVERT" "build_failed" \
                "$baseline_tok_s" "0" "0" "$build_status" "SKIP" "SKIP"
            log_tsv_row \
                "exp-ph13-$(printf '%04d' $iter)" "$(ts_iso)" "$base_sha" "$base_sha" \
                "$ARCH" "" "" "$MODEL" "mq4" "q8" \
                "$prompt_name" "$prompt_md5" \
                "$current_target" "kernels/src/$variant_basename" "$strategy" \
                "SKIP" "SKIP" "FAIL" \
                "$baseline_tok_s" "0" "0" \
                "0" "0" "0" "0" "0" \
                "0" "0" "0" \
                "REVERT" "build_failed" "" \
                "$iter" "$current_target" "$strategy" "kernels/src/$variant_basename" \
                "FAIL" "$baseline_tok_s" "$current_best" "0" "1.0" "1.0"
            continue
        fi

        # ── Quick correctness gate ────────────────────────────────────────────
        local correctness_status="SKIP"
        local correctness_log="$cand_dir/correctness.log"
        if [ -x "$REPO_ROOT/scripts/test-kernels.sh" ]; then
            log "Running correctness gate..."
            correctness_status="PASS"
            if ! timeout "$TIMEOUT_SECONDS" "$REPO_ROOT/scripts/test-kernels.sh" \
                    >"$correctness_log" 2>&1; then
                correctness_status="FAIL"
            fi
        else
            echo "test-kernels.sh not found — SKIP" > "$correctness_log"
        fi

        if [ "$correctness_status" = "FAIL" ]; then
            err "Correctness gate FAILED — reverting (hard rule: never keep a failing candidate)"
            cp "${variant_file}.bak" "$variant_file"; rm -f "${variant_file}.bak"
            record_strategy_tried "$current_target" "$strategy"
            increment_attempts "$current_target"
            consecutive_failures=$((consecutive_failures+1))
            cp -r "$cand_dir" "$REJECTED_DIR/" 2>/dev/null || true
            append_rejected "${current_target}:${strategy}:iter${iter}"
            write_candidate_metadata "$cand_dir" "$iter" "$current_target" "$strategy" \
                "$variant_file" "$base_sha" "$base_sha" "REVERT" "correctness_gate_failed" \
                "$baseline_tok_s" "0" "0" "$build_status" "FAIL" "SKIP"
            log_tsv_row \
                "exp-ph13-$(printf '%04d' $iter)" "$(ts_iso)" "$base_sha" "$base_sha" \
                "$ARCH" "" "" "$MODEL" "mq4" "q8" \
                "$prompt_name" "$prompt_md5" \
                "$current_target" "kernels/src/$variant_basename" "$strategy" \
                "FAIL" "SKIP" "PASS" \
                "$baseline_tok_s" "0" "0" \
                "0" "0" "0" "0" "0" \
                "0" "0" "0" \
                "REVERT" "correctness_gate_failed" "" \
                "$iter" "$current_target" "$strategy" "kernels/src/$variant_basename" \
                "FAIL" "$baseline_tok_s" "$current_best" "0" "1.0" "1.0"
            continue
        fi

        # ── Benchmark ────────────────────────────────────────────────────────
        local bench_status="PASS"
        local bench_log="$cand_dir/benchmark.log"
        log "Benchmarking ($BENCH_TRIALS trials)..."
        {
            echo "model=$MODEL  arch=$ARCH  strategy=$strategy"
            echo "trials=$BENCH_TRIALS  baseline=$baseline_tok_s"
        } > "$bench_log"

        local bench_out
        bench_out="$(run_bench_trials "$BENCH_TRIALS")"
        local med_gen med_pre sd_gen sd_pre med_vram
        IFS=' ' read -r med_gen med_pre sd_gen sd_pre med_vram <<< "$bench_out"

        echo "result: decode=$med_gen prefill=$med_pre stddev=$sd_gen vram=$med_vram" >> "$bench_log"

        if ! float_ge "${med_gen:-0}" "1.0"; then
            bench_status="FAIL"
            warn "Benchmark produced 0 or error (med_gen='$med_gen') — reverting"
            cp "${variant_file}.bak" "$variant_file"; rm -f "${variant_file}.bak"
            record_strategy_tried "$current_target" "$strategy"
            increment_attempts "$current_target"
            update_state "timeout_count=$(($(get_state timeout_count)+1))"
            consecutive_failures=$((consecutive_failures+1))
            cp -r "$cand_dir" "$REJECTED_DIR/" 2>/dev/null || true
            append_rejected "${current_target}:${strategy}:iter${iter}"
            write_candidate_metadata "$cand_dir" "$iter" "$current_target" "$strategy" \
                "$variant_file" "$base_sha" "$base_sha" "REVERT" "benchmark_failed" \
                "$baseline_tok_s" "0" "0" "$build_status" "$correctness_status" "FAIL"
            log_tsv_row \
                "exp-ph13-$(printf '%04d' $iter)" "$(ts_iso)" "$base_sha" "$base_sha" \
                "$ARCH" "" "" "$MODEL" "mq4" "q8" \
                "$prompt_name" "$prompt_md5" \
                "$current_target" "kernels/src/$variant_basename" "$strategy" \
                "$correctness_status" "SKIP" "$build_status" \
                "$baseline_tok_s" "0" "0" \
                "0" "0" "0" "0" "0" \
                "0" "0" "0" \
                "REVERT" "benchmark_failed" "" \
                "$iter" "$current_target" "$strategy" "kernels/src/$variant_basename" \
                "FAIL" "$baseline_tok_s" "$current_best" "0" "1.0" "1.0"
            continue
        fi

        # ── Accept / Reject decision ─────────────────────────────────────────
        local decode_speedup; decode_speedup="$(float_div "$med_gen" "$baseline_tok_s")"
        local speedup_vs_best; speedup_vs_best="$(float_div "$med_gen" "$current_best")"
        local prefill_speedup; prefill_speedup="$(float_div "${med_pre:-0}" "${med_pre:-1}")"
        local cand_sha; cand_sha="$(git_sha)"

        log "decode=$med_gen tok/s  baseline=$baseline_tok_s  speedup=${decode_speedup}x  vs_best=${speedup_vs_best}x"

        local decision revert_reason=""
        if float_ge "$decode_speedup" "$ACCEPT_MIN_SPEEDUP"; then
            decision="KEEP"
            ok "ACCEPTED: ${decode_speedup}x >= threshold ${ACCEPT_MIN_SPEEDUP}x"
            current_best="$med_gen"
            update_state "best_tok_s=$med_gen" "best_speedup=$decode_speedup" \
                         "next_action=CONTINUE" "last_decision=KEEP"
            append_accepted "${current_target}:${strategy}:iter${iter}"
            cp -r "$cand_dir" "$ACCEPTED_DIR/" 2>/dev/null || true
            rm -f "${variant_file}.bak"
            consecutive_failures=0

            if [ "${AUTOCOMMIT:-0}" = "1" ]; then
                log "AUTOCOMMIT=1 — committing..."
                git -C "$REPO_ROOT" add "kernels/src/$variant_basename" 2>/dev/null || true
                git -C "$REPO_ROOT" commit -m "autokernel: optimize $current_target for $ARCH

Strategy: $strategy
Speedup: ${decode_speedup}x (${baseline_tok_s} -> ${med_gen} tok/s)
Model: $MODEL / Iteration: $iter" 2>/dev/null \
                    || warn "git commit failed (files not staged or nothing changed)"
            fi
        else
            decision="REVERT"
            revert_reason="decode ${decode_speedup}x below threshold ${ACCEPT_MIN_SPEEDUP}x"
            warn "REJECTED: $revert_reason"
            cp "${variant_file}.bak" "$variant_file"; rm -f "${variant_file}.bak"
            update_state "last_decision=REVERT"
            consecutive_failures=$((consecutive_failures+1))
            cp -r "$cand_dir" "$REJECTED_DIR/" 2>/dev/null || true
            append_rejected "${current_target}:${strategy}:iter${iter}"
        fi

        # ── Write candidate metadata ──────────────────────────────────────────
        write_candidate_metadata "$cand_dir" "$iter" "$current_target" "$strategy" \
            "$variant_file" "$base_sha" "$cand_sha" "$decision" "$revert_reason" \
            "$baseline_tok_s" "${med_gen:-0}" "$decode_speedup" \
            "$build_status" "$correctness_status" "$bench_status"

        # ── TSV row ───────────────────────────────────────────────────────────
        log_tsv_row \
            "exp-ph13-$(printf '%04d' $iter)" "$(ts_iso)" "$base_sha" "$cand_sha" \
            "$ARCH" "" "" "$MODEL" "mq4" "q8" \
            "$prompt_name" "$prompt_md5" \
            "$current_target" "kernels/src/$variant_basename" "$strategy" \
            "$correctness_status" "SKIP" "$build_status" \
            "$baseline_tok_s" "${med_gen:-0}" "$decode_speedup" \
            "${med_pre:-0}" "${med_pre:-0}" "1.0000" \
            "$decode_speedup" "$decode_speedup" \
            "$BENCH_TRIALS" "${sd_gen:-0}" "${med_vram:-0}" \
            "$decision" "$revert_reason" "" \
            "$iter" "$current_target" "$strategy" "kernels/src/$variant_basename" \
            "$bench_status" "$baseline_tok_s" "$current_best" \
            "${med_gen:-0}" "$decode_speedup" "$speedup_vs_best"

        # ── Update orchestration state ────────────────────────────────────────
        record_strategy_tried "$current_target" "$strategy"
        increment_attempts "$current_target"
        update_state "total_iterations=$iter"

        # Advance target if max attempts hit
        local new_attempts; new_attempts="$(get_attempts "$current_target")"
        if [ "$new_attempts" -ge "$MAX_ATTEMPTS_PER_TARGET" ]; then
            log "Max attempts for '$current_target' — advancing target"
            current_target="$(advance_target)"
            update_state "current_target=$current_target"
            [ -z "$current_target" ] && { warn "All targets exhausted"; break; }
        fi

        # Anti-thrash: too many consecutive failures → advance target
        if [ "$consecutive_failures" -ge "$MAX_ATTEMPTS_PER_TARGET" ]; then
            warn "Too many consecutive failures ($consecutive_failures) — advancing target"
            consecutive_failures=0
            current_target="$(advance_target)"
            update_state "current_target=$current_target"
            [ -z "$current_target" ] && break
        fi

        # Check for stop signal (set by SIGINT handler)
        [ "$LOOP_RUNNING" = "2" ] && break

    done  # ── end main loop

    LOOP_RUNNING=0
    update_state "total_iterations=$iter"
    hdr "Loop complete — $iter iterations"
    write_final_report
}

main "$@"
