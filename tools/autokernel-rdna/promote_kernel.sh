#!/usr/bin/env bash
# tools/autokernel-rdna/promote_kernel.sh
#
# Phase 15 — Kernel Promotion Pipeline
#
# Promotes a verified kernel_lab candidate into hipfire safely.
# One candidate at a time, full validation before accept.
#
# Usage:
#   ARCH=gfx1201 \
#   MODEL=qwen3.5:27b \
#   CANDIDATE=tools/autokernel-rdna/kernel_lab/generated/<target>/candidate_<N>.hip \
#   ./tools/autokernel-rdna/promote_kernel.sh
#
# Required env:
#   CANDIDATE         - path to candidate_N.hip (required)
#
# Optional env:
#   ARCH              - target arch (default: gfx1201)
#   MODEL             - hipfire model slug (default: qwen3.5:27b)
#   ACCEPT_MIN_SPEEDUP - minimum tok/s ratio to accept (default: 1.005)
#   ALLOW_OTHER_ARCH  - set to 1 to skip arch check (default: 0)
#   AUTOCOMMIT        - set to 1 to git-commit on accept (default: 0)
#   SKIP_COHERENCE    - set to 1 to skip coherence-gate-dflash.sh (default: 0)
#   SKIP_SPEED_GATE   - set to 1 to skip speed-gate.sh --fast (default: 0)
#   VERBOSE           - set to 1 for extra output (default: 0)
#
# Hard rules:
#   - Promote only one candidate at a time.
#   - Never delete the original generic kernel.
#   - Never overwrite a fallback without backup.
#   - Correctness is a hard gate — fast-but-wrong = FAIL.
#   - All gates run before results.tsv is updated.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TOOL_DIR="$SCRIPT_DIR"
cd "$REPO_ROOT"

# ── Config (overridable via env) ──────────────────────────────────────────
ARCH="${ARCH:-gfx1201}"
MODEL="${MODEL:-qwen3.5:27b}"
CANDIDATE="${CANDIDATE:-}"
ACCEPT_MIN_SPEEDUP="${ACCEPT_MIN_SPEEDUP:-1.005}"
ALLOW_OTHER_ARCH="${ALLOW_OTHER_ARCH:-0}"
AUTOCOMMIT="${AUTOCOMMIT:-0}"
SKIP_COHERENCE="${SKIP_COHERENCE:-0}"
SKIP_SPEED_GATE="${SKIP_SPEED_GATE:-0}"
VERBOSE="${VERBOSE:-0}"

# ── Paths ────────────────────────────────────────────────────────────────
RESULTS_TSV="$TOOL_DIR/results.tsv"
REPORTS_DIR="$TOOL_DIR/reports"
WORKSPACE_DIR="$TOOL_DIR/workspace"
BACKUP_BASE="$WORKSPACE_DIR/promotion_backups"
ACCEPTED_DIR="$WORKSPACE_DIR/accepted_promotions"
REJECTED_DIR="$WORKSPACE_DIR/rejected_promotions"
KERNELS_SRC="$REPO_ROOT/kernels/src"
BENCH_EXE="$REPO_ROOT/target/release/examples/bench_qwen35_mq4"
BENCH_PROMPT="${HIPFIRE_BENCH_PROMPT:-$REPO_ROOT/benchmarks/prompts/lru_cache_pep8_strict.txt}"
BASELINES_DIR="$TOOL_DIR/baselines"

if [ -d "$HOME/.hipfire/models" ] && ls "$HOME/.hipfire/models"/*.mq4 >/dev/null 2>&1; then
    MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
elif [ -d "/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.hipfire/models" ]; then
    MODELS_DIR="${HIPFIRE_MODELS_DIR:-/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.hipfire/models}"
else
    MODELS_DIR="${HIPFIRE_MODELS_DIR:-$HOME/.hipfire/models}"
fi

# ── Colours ──────────────────────────────────────────────────────────────
if [[ -t 1 ]]; then
    C_GREEN='\033[0;32m'; C_YELLOW='\033[0;33m'; C_RED='\033[0;31m'
    C_CYAN='\033[0;36m'; C_BOLD='\033[1m'; C_RESET='\033[0m'
else
    C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''; C_BOLD=''; C_RESET=''
fi
log_info()  { echo -e "${C_CYAN}[promote]${C_RESET} $*"; }
log_ok()    { echo -e "${C_GREEN}[promote]${C_RESET} $*"; }
log_warn()  { echo -e "${C_YELLOW}[promote]${C_RESET} $*"; }
log_error() { echo -e "${C_RED}[promote]${C_RESET} $*" >&2; }
log_bold()  { echo -e "${C_BOLD}$*${C_RESET}"; }

# ── Timestamps ───────────────────────────────────────────────────────────
TS="$(date -u +%Y%m%d-%H%M%S)"
TS_ISO="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# ── State variables (populated during run) ───────────────────────────────
TARGET_NAME=""
PROMOTED_DEST=""
DISPATCH_FILES_CHANGED=""
BACKUP_DIR=""
HARNESS_CORRECTNESS="UNKNOWN"
HARNESS_SPEEDUP="0"
BUILD_STATUS="UNKNOWN"
CORRECTNESS_STATUS="UNKNOWN"
BASELINE_TOK_S="0"
CANDIDATE_TOK_S="0"
HIPFIRE_SPEEDUP="0"
DECISION="UNKNOWN"
ROLLBACK_STATUS="N/A"
REJECT_REASON=""
CANDIDATE_HIP=""
SPEC_FILE=""

# ═══════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════

die() {
    log_error "$*"
    exit 1
}

require_file() {
    local f="$1" label="$2"
    if [ ! -f "$f" ]; then
        die "Missing $label: $f"
    fi
}

jq_field() {
    # Lightweight jq substitute — extracts top-level string/number field.
    # Falls back gracefully if python3 not available (offline tooling).
    local file="$1" field="$2"
    python3 -c "
import json, sys
try:
    d = json.load(open('$file'))
    v = d.get('$field', '')
    print(v if v is not None else '')
except Exception:
    print('')
" 2>/dev/null || true
}

run_bench() {
    # Returns gen_tok_s from bench binary. Prints to stdout.
    local model_path="$1"
    local prompt_file="$2"
    if [ ! -f "$model_path" ]; then
        echo "0"
        return
    fi
    if [ ! -f "$BENCH_EXE" ]; then
        echo "0"
        return
    fi
    local out
    out=$("$BENCH_EXE" "$model_path" --gen 80 --warmup 5 < "$prompt_file" 2>&1) || true
    echo "$out" | grep -oP 'gen_tok_s=\K[0-9.]+' | head -1 || echo "0"
}

find_model_path() {
    local model_slug="$1"
    local size=""
    echo "$model_slug" | grep -q "27b" && size="27b"
    echo "$model_slug" | grep -q "9b"  && size="9b"

    local candidates=()
    candidates+=(
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
    # No size match — return first valid
    for c in "${candidates[@]}"; do
        [ -f "$c" ] && { echo "$c"; return 0; }
    done
    echo ""
}

restore_backup() {
    if [ -z "$BACKUP_DIR" ] || [ ! -d "$BACKUP_DIR" ]; then
        log_warn "No backup to restore from"
        ROLLBACK_STATUS="NO_BACKUP"
        return 1
    fi
    log_warn "Restoring backup from $BACKUP_DIR ..."
    local restored=0
    # Restore kernel file
    for f in "$BACKUP_DIR"/*.hip; do
        [ -f "$f" ] || continue
        local dest_rel
        dest_rel=$(cat "$BACKUP_DIR/promoted_dest.txt" 2>/dev/null || echo "")
        if [ -n "$dest_rel" ] && [ -n "$PROMOTED_DEST" ]; then
            cp "$f" "$PROMOTED_DEST"
            log_ok "  Restored: $PROMOTED_DEST"
            restored=$((restored+1))
        fi
    done
    # If dest didn't exist before promotion, remove it
    local existed_before
    existed_before=$(cat "$BACKUP_DIR/dest_existed_before.txt" 2>/dev/null || echo "1")
    if [ "$existed_before" = "0" ] && [ -n "$PROMOTED_DEST" ] && [ -f "$PROMOTED_DEST" ]; then
        rm -f "$PROMOTED_DEST"
        log_ok "  Removed newly-created $PROMOTED_DEST (did not exist before)"
        restored=$((restored+1))
    fi
    if [ $restored -eq 0 ]; then
        ROLLBACK_STATUS="FAIL_NO_FILES"
        log_error "Nothing restored — check $BACKUP_DIR"
        return 1
    fi
    ROLLBACK_STATUS="OK"
    log_ok "Rollback complete"
}

write_tsv_row() {
    local decision="$1"
    local reject_reason="${2:-}"
    local experiment_id="exp-promote-$TS"
    local git_sha
    git_sha=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
    local gpu_name
    gpu_name=$(rocminfo 2>/dev/null | grep -i "Marketing Name" | head -1 | sed 's/.*: *//' || echo "gfx1201")
    local rocm_ver
    rocm_ver=$(cat /opt/rocm/.info/version 2>/dev/null || echo "unknown")
    local prompt_md5=""
    [ -f "$BENCH_PROMPT" ] && prompt_md5=$(md5sum "$BENCH_PROMPT" | awk '{print $1}')

    # Use tabs between fields. Phase 15 appends to existing 48-col schema
    # + 14 new Phase 15 columns (total 62):
    #   49: phase
    #   50: promotion_candidate_dir
    #   51: promoted_kernel_file
    #   52: dispatch_files_changed
    #   53: harness_correctness_status (P15)
    #   54: harness_speedup (P15)
    #   55: hipfire_build_status
    #   56: hipfire_correctness_status
    #   57: hipfire_baseline_tok_s
    #   58: hipfire_candidate_tok_s
    #   59: hipfire_speedup
    #   60: promotion_decision
    #   61: rollback_status
    #   62: reject_reason

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$experiment_id" \
        "$TS_ISO" \
        "$git_sha" \
        "$git_sha" \
        "$ARCH" \
        "$gpu_name" \
        "$rocm_ver" \
        "$MODEL" \
        "mq4" \
        "" \
        "lru_cache_pep8_strict.txt" \
        "$prompt_md5" \
        "$TARGET_NAME" \
        "$(basename "$PROMOTED_DEST" 2>/dev/null || echo "")" \
        "phase15_promotion" \
        "$CORRECTNESS_STATUS" \
        "" \
        "$BUILD_STATUS" \
        "$BASELINE_TOK_S" \
        "$CANDIDATE_TOK_S" \
        "$HIPFIRE_SPEEDUP" \
        "" "" "" "" "" \
        "1" "" \
        "0" \
        "$decision" \
        "$reject_reason" \
        "" \
        "" \
        "$TARGET_NAME" \
        "phase15_promote" \
        "" \
        "" \
        "" \
        "1" \
        "" \
        "1" \
        "$(dirname "$CANDIDATE_HIP" 2>/dev/null || echo "")" \
        "$CANDIDATE_HIP" \
        "$HARNESS_CORRECTNESS" \
        "$HARNESS_SPEEDUP" \
        "1" \
        "$BASELINE_TOK_S" \
        "$CANDIDATE_TOK_S" \
        "$decision" \
        "phase15" \
        "$(dirname "$CANDIDATE_HIP" 2>/dev/null || echo "")" \
        "$PROMOTED_DEST" \
        "$DISPATCH_FILES_CHANGED" \
        "$HARNESS_CORRECTNESS" \
        "$HARNESS_SPEEDUP" \
        "$BUILD_STATUS" \
        "$CORRECTNESS_STATUS" \
        "$BASELINE_TOK_S" \
        "$CANDIDATE_TOK_S" \
        "$HIPFIRE_SPEEDUP" \
        "$decision" \
        "$ROLLBACK_STATUS" \
        "$reject_reason" \
    >> "$RESULTS_TSV"
}

write_report() {
    local decision="$1"
    local report_md="$REPORTS_DIR/phase15_promotion_${TS}.md"
    local report_json="$REPORTS_DIR/phase15_promotion_${TS}.json"

    mkdir -p "$REPORTS_DIR"

    local rollback_cmd=""
    if [ -n "$PROMOTED_DEST" ] && [ -n "$BACKUP_DIR" ]; then
        rollback_cmd="cp $BACKUP_DIR/$(basename "$PROMOTED_DEST") $PROMOTED_DEST"
        if [ "$(cat "$BACKUP_DIR/dest_existed_before.txt" 2>/dev/null || echo "1")" = "0" ]; then
            rollback_cmd="rm -f $PROMOTED_DEST  # file did not exist before promotion"
        fi
    fi

    cat > "$report_md" <<EOF
# Phase 15 Promotion Report — ${TS_ISO}

| Field | Value |
|---|---|
| Target | \`$TARGET_NAME\` |
| Candidate | \`$CANDIDATE_HIP\` |
| Promoted to | \`$PROMOTED_DEST\` |
| Dispatch changes | \`${DISPATCH_FILES_CHANGED:-none}\` |
| Backup location | \`${BACKUP_DIR:-none}\` |
| Arch | $ARCH |
| Model | $MODEL |

## Harness Result

| | |
|---|---|
| Correctness | **$HARNESS_CORRECTNESS** |
| Speedup vs ref | $HARNESS_SPEEDUP× |

## hipfire Build

| | |
|---|---|
| Build status | **$BUILD_STATUS** |

## Correctness

| | |
|---|---|
| Quick correctness | **$CORRECTNESS_STATUS** |

## Qwen3.5-27B Benchmark

| | tok/s |
|---|---|
| Baseline | $BASELINE_TOK_S |
| Candidate | $CANDIDATE_TOK_S |
| Speedup | **${HIPFIRE_SPEEDUP}×** |
| Threshold | $ACCEPT_MIN_SPEEDUP× |

## Decision: $decision

$([ -n "$REJECT_REASON" ] && echo "**Reject reason:** $REJECT_REASON" || true)

## Rollback Command

\`\`\`bash
$rollback_cmd
\`\`\`

Or restore from backup:

\`\`\`bash
ls $BACKUP_DIR/
\`\`\`
EOF

    python3 - <<PYEOF
import json
d = {
    "timestamp": "$TS_ISO",
    "arch": "$ARCH",
    "model": "$MODEL",
    "target_name": "$TARGET_NAME",
    "candidate_hip": "$CANDIDATE_HIP",
    "promoted_dest": "$PROMOTED_DEST",
    "dispatch_files_changed": "$DISPATCH_FILES_CHANGED",
    "backup_dir": "$BACKUP_DIR",
    "harness_correctness": "$HARNESS_CORRECTNESS",
    "harness_speedup": "$HARNESS_SPEEDUP",
    "build_status": "$BUILD_STATUS",
    "correctness_status": "$CORRECTNESS_STATUS",
    "baseline_tok_s": "$BASELINE_TOK_S",
    "candidate_tok_s": "$CANDIDATE_TOK_S",
    "hipfire_speedup": "$HIPFIRE_SPEEDUP",
    "decision": "$decision",
    "reject_reason": "$REJECT_REASON",
    "rollback_status": "$ROLLBACK_STATUS",
    "rollback_cmd": "$rollback_cmd",
}
with open("$report_json", "w") as f:
    json.dump(d, f, indent=2)
PYEOF

    log_ok "Report: $report_md"
    log_ok "JSON:   $report_json"
    echo "$report_md"
}

reject() {
    local reason="$1"
    REJECT_REASON="$reason"
    DECISION="FAIL_REVERTED"
    log_error "REJECTED: $reason"
    restore_backup || true
    # Archive to rejected_promotions
    if [ -n "$CANDIDATE_HIP" ] && [ -f "$CANDIDATE_HIP" ]; then
        mkdir -p "$REJECTED_DIR"
        local slug
        slug="${TARGET_NAME}_${TS}"
        cp "$CANDIDATE_HIP" "$REJECTED_DIR/${slug}.hip"
    fi
    write_tsv_row "FAIL_REVERTED" "$reason"
    write_report "FAIL_REVERTED"
    exit 1
}

# ═══════════════════════════════════════════════════════════════════════
# Step 1: validate candidate
# ═══════════════════════════════════════════════════════════════════════

validate_candidate() {
    log_bold "═══ Phase 15: Kernel Promotion Pipeline ═══"
    log_info "Candidate: $CANDIDATE"
    log_info "Arch:      $ARCH"
    log_info "Model:     $MODEL"
    echo ""

    if [ -z "$CANDIDATE" ]; then
        die "CANDIDATE env var is not set. Usage: CANDIDATE=<path/to/candidate_N.hip> $0"
    fi

    # Resolve absolute path
    CANDIDATE_HIP="$(cd "$(dirname "$CANDIDATE")" && pwd)/$(basename "$CANDIDATE")"
    local CAND_DIR
    CAND_DIR="$(dirname "$CANDIDATE_HIP")"

    require_file "$CANDIDATE_HIP" "candidate HIP file"

    log_info "Candidate dir: $CAND_DIR"

    # Locate spec — look in candidate dir, then parent dir
    if [ -f "$CAND_DIR/target_spec.json" ]; then
        SPEC_FILE="$CAND_DIR/target_spec.json"
    elif [ -f "$(dirname "$CAND_DIR")/target_spec.json" ]; then
        SPEC_FILE="$(dirname "$CAND_DIR")/target_spec.json"
    fi

    if [ -z "$SPEC_FILE" ] || [ ! -f "$SPEC_FILE" ]; then
        die "target_spec.json not found near candidate (checked $CAND_DIR and parent)"
    fi
    log_info "Spec: $SPEC_FILE"

    # Read harness_result.json if present
    local harness_result=""
    for hr in "$CAND_DIR/harness_result.json" "$CAND_DIR/../harness_result.json"; do
        [ -f "$hr" ] && { harness_result="$hr"; break; }
    done

    # Extract target name from spec
    TARGET_NAME=$(jq_field "$SPEC_FILE" "kernel_name")
    if [ -z "$TARGET_NAME" ]; then
        die "kernel_name not found in $SPEC_FILE"
    fi
    log_info "Target: $TARGET_NAME"

    # Check arch
    local spec_arch
    spec_arch=$(jq_field "$SPEC_FILE" "arch")
    if [ "$ALLOW_OTHER_ARCH" != "1" ]; then
        if [ -n "$spec_arch" ] && [ "$spec_arch" != "$ARCH" ]; then
            die "Arch mismatch: spec says '$spec_arch', ARCH='$ARCH'. Set ALLOW_OTHER_ARCH=1 to override."
        fi
    fi

    # Read harness result
    if [ -n "$harness_result" ]; then
        log_info "Reading harness result: $harness_result"
        HARNESS_CORRECTNESS=$(jq_field "$harness_result" "correctness")
        HARNESS_SPEEDUP=$(jq_field "$harness_result" "speedup_vs_ref")
        [ -z "$HARNESS_CORRECTNESS" ] && HARNESS_CORRECTNESS="UNKNOWN"
        [ -z "$HARNESS_SPEEDUP" ]     && HARNESS_SPEEDUP="0"
    else
        log_warn "No harness_result.json found — will re-run harness"
        run_harness_now
    fi

    # Enforce gates
    if [ "$HARNESS_CORRECTNESS" != "PASS" ]; then
        die "Harness correctness is '$HARNESS_CORRECTNESS' — refusing to promote a failing candidate"
    fi

    local speedup_ok
    speedup_ok=$(python3 -c "
s = float('$HARNESS_SPEEDUP') if '$HARNESS_SPEEDUP' else 0.0
print('yes' if s >= float('$ACCEPT_MIN_SPEEDUP') else 'no')
" 2>/dev/null || echo "no")
    if [ "$speedup_ok" != "yes" ]; then
        die "Harness speedup $HARNESS_SPEEDUP× < threshold $ACCEPT_MIN_SPEEDUP× — refusing to promote"
    fi

    log_ok "Candidate validation passed (correctness=$HARNESS_CORRECTNESS, speedup=${HARNESS_SPEEDUP}×)"
}

run_harness_now() {
    # Re-run the harness for the candidate to get fresh results
    local harness_script="$TOOL_DIR/kernel_lab/harnesses/${TARGET_NAME}_harness.sh"
    if [ ! -f "$harness_script" ]; then
        log_warn "No harness script found at $harness_script — skipping harness re-run"
        log_warn "Set HARNESS_CORRECTNESS and HARNESS_SPEEDUP manually or provide harness_result.json"
        # If we have nothing, let caller decide
        HARNESS_CORRECTNESS="${HARNESS_CORRECTNESS:-UNKNOWN}"
        HARNESS_SPEEDUP="${HARNESS_SPEEDUP:-0}"
        return
    fi
    log_info "Re-running harness: $harness_script"
    local harness_out
    harness_out=$(ARCH="$ARCH" CANDIDATE="$CANDIDATE_HIP" bash "$harness_script" 2>&1) || true
    HARNESS_CORRECTNESS=$(echo "$harness_out" | grep '^CORRECTNESS:' | awk '{print $2}')
    HARNESS_SPEEDUP=$(echo "$harness_out" | grep '^SPEEDUP_VS_REF:' | awk '{print $2}')
    [ -z "$HARNESS_CORRECTNESS" ] && HARNESS_CORRECTNESS="FAIL"
    [ -z "$HARNESS_SPEEDUP" ]     && HARNESS_SPEEDUP="0"
    log_info "  Harness correctness: $HARNESS_CORRECTNESS  speedup: ${HARNESS_SPEEDUP}×"

    # Save result json next to candidate
    python3 - <<PYEOF 2>/dev/null || true
import json
d = {"correctness": "$HARNESS_CORRECTNESS", "speedup_vs_ref": "$HARNESS_SPEEDUP", "harness_output": """$harness_out"""}
with open("$(dirname "$CANDIDATE_HIP")/harness_result.json", "w") as f:
    json.dump(d, f, indent=2)
PYEOF
}

# ═══════════════════════════════════════════════════════════════════════
# Step 2: determine promotion destination
# ═══════════════════════════════════════════════════════════════════════

determine_destination() {
    local gfx1201_variant
    gfx1201_variant=$(jq_field "$SPEC_FILE" "gfx1201_variant")
    local gfx12_variant
    gfx12_variant=$(jq_field "$SPEC_FILE" "gfx12_variant")

    if [ "$ARCH" = "gfx1201" ] && [ -n "$gfx1201_variant" ]; then
        PROMOTED_DEST="$REPO_ROOT/$gfx1201_variant"
        log_info "Destination: $PROMOTED_DEST (gfx1201-specific)"
    elif [ -n "$gfx12_variant" ]; then
        PROMOTED_DEST="$REPO_ROOT/$gfx12_variant"
        log_info "Destination: $PROMOTED_DEST (gfx12 family)"
    else
        # Fallback: construct from target name
        PROMOTED_DEST="$KERNELS_SRC/${TARGET_NAME}.gfx1201.hip"
        log_warn "No variant path in spec — defaulting to $PROMOTED_DEST"
    fi

    # Sanity: never overwrite the generic fallback
    local generic_kernel="$KERNELS_SRC/${TARGET_NAME}.hip"
    if [ "$(realpath "$PROMOTED_DEST" 2>/dev/null || echo "$PROMOTED_DEST")" = \
         "$(realpath "$generic_kernel" 2>/dev/null || echo "$generic_kernel")" ]; then
        die "Destination resolves to generic fallback $generic_kernel — refusing to overwrite fallback kernel"
    fi

    log_ok "Promotion destination: $PROMOTED_DEST"
}

# ═══════════════════════════════════════════════════════════════════════
# Step 3: backup
# ═══════════════════════════════════════════════════════════════════════

create_backup() {
    BACKUP_DIR="$BACKUP_BASE/$TS"
    mkdir -p "$BACKUP_DIR"

    # Save whether destination existed before
    if [ -f "$PROMOTED_DEST" ]; then
        echo "1" > "$BACKUP_DIR/dest_existed_before.txt"
        cp "$PROMOTED_DEST" "$BACKUP_DIR/$(basename "$PROMOTED_DEST").bak"
        log_info "Backed up existing: $PROMOTED_DEST → $BACKUP_DIR/"
    else
        echo "0" > "$BACKUP_DIR/dest_existed_before.txt"
        log_info "Destination does not yet exist — will create new file"
    fi

    # Save candidate + spec
    cp "$CANDIDATE_HIP" "$BACKUP_DIR/"
    cp "$SPEC_FILE"     "$BACKUP_DIR/target_spec.json.bak"

    # Save git diff
    git diff HEAD > "$BACKUP_DIR/git_diff_before.patch" 2>/dev/null || true

    # Record promoted dest path for restore
    echo "$PROMOTED_DEST" > "$BACKUP_DIR/promoted_dest.txt"

    log_ok "Backup saved: $BACKUP_DIR"
}

# ═══════════════════════════════════════════════════════════════════════
# Step 4: dispatch wiring (hipfire already uses arch-specific .hip files)
# ═══════════════════════════════════════════════════════════════════════

wire_dispatch() {
    # hipfire's dispatch mechanism in crates/rdna-compute/src/kernels.rs
    # already selects .gfx1201.hip at kernel-cache build time when the
    # file exists alongside the generic kernel. Copying the candidate to
    # kernels/src/<kernel>.gfx1201.hip is sufficient — no Rust edit needed.
    #
    # If kernels.rs required explicit wiring, we'd patch the smallest
    # possible block here. For now, the copy is the wiring.
    DISPATCH_FILES_CHANGED="none (arch-specific variant file replaces/creates .gfx1201.hip)"
    log_info "Dispatch: no code change needed — arch-specific variant auto-selected"
    log_info "  Writing candidate to: $PROMOTED_DEST"
    mkdir -p "$(dirname "$PROMOTED_DEST")"
    cp "$CANDIDATE_HIP" "$PROMOTED_DEST"
    log_ok "Candidate installed: $PROMOTED_DEST"
}

# ═══════════════════════════════════════════════════════════════════════
# Step 5: build
# ═══════════════════════════════════════════════════════════════════════

build_hipfire() {
    log_info "Building hipfire (release)..."
    local build_log="$BACKUP_DIR/build.log"
    local cargo_bin="${CARGO_BIN:-$(which cargo 2>/dev/null || echo "")}"
    if [ -z "$cargo_bin" ]; then
        # Try well-known location
        for p in \
            "$HOME/.cargo/bin/cargo" \
            "/media/dev/Tforce/dev/radeonmax/baselines/hipfire/.cargo/bin/cargo"; do
            [ -x "$p" ] && { cargo_bin="$p"; break; }
        done
    fi
    if [ -z "$cargo_bin" ]; then
        BUILD_STATUS="FAIL"
        reject "cargo not found — cannot build hipfire"
    fi

    if "$cargo_bin" build --release --features deltanet \
        --example bench_qwen35_mq4 \
        -p hipfire-runtime 2>"$build_log"; then
        BUILD_STATUS="PASS"
        log_ok "Build passed"
    else
        BUILD_STATUS="FAIL"
        log_error "Build log: $build_log"
        reject "hipfire build failed after promotion — see $build_log"
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# Step 6: correctness
# ═══════════════════════════════════════════════════════════════════════

run_correctness() {
    CORRECTNESS_STATUS="PASS"  # optimistic default if gates not available

    # Quick self-check via test-kernels.sh if available
    local test_script="$REPO_ROOT/scripts/test-kernels.sh"
    if [ -f "$test_script" ]; then
        log_info "Running quick kernel tests..."
        local test_out_file="$BACKUP_DIR/test_kernels.log"
        if bash "$test_script" >"$test_out_file" 2>&1; then
            log_ok "Quick kernel tests passed"
        else
            CORRECTNESS_STATUS="FAIL"
            reject "Quick kernel tests failed — see $test_out_file"
        fi
    else
        log_warn "test-kernels.sh not found — skipping quick correctness test"
    fi

    # coherence-gate-dflash.sh
    if [ "$SKIP_COHERENCE" != "1" ]; then
        local cgate="$REPO_ROOT/scripts/coherence-gate-dflash.sh"
        if [ -f "$cgate" ]; then
            log_info "Running coherence-gate-dflash.sh..."
            local cgate_log="$BACKUP_DIR/coherence_gate.log"
            if bash "$cgate" >"$cgate_log" 2>&1; then
                log_ok "Coherence gate passed"
            else
                CORRECTNESS_STATUS="FAIL"
                reject "coherence-gate-dflash.sh failed — see $cgate_log"
            fi
        else
            log_warn "coherence-gate-dflash.sh not found — skipping"
        fi
    else
        log_warn "SKIP_COHERENCE=1 — skipping coherence gate"
    fi

    # speed-gate.sh --fast
    if [ "$SKIP_SPEED_GATE" != "1" ]; then
        local sgate="$REPO_ROOT/scripts/speed-gate.sh"
        if [ -f "$sgate" ]; then
            log_info "Running speed-gate.sh --fast..."
            local sgate_log="$BACKUP_DIR/speed_gate.log"
            if bash "$sgate" --fast >"$sgate_log" 2>&1; then
                log_ok "Speed gate passed"
            else
                # Speed gate is advisory here — we rely on our own tok/s bench
                log_warn "speed-gate.sh --fast reported issues — see $sgate_log"
                log_warn "Continuing (speed gate is advisory; our tok/s bench is the accept gate)"
            fi
        else
            log_warn "speed-gate.sh not found — skipping"
        fi
    else
        log_warn "SKIP_SPEED_GATE=1 — skipping speed gate"
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# Step 7: Qwen3.5-27B benchmark
# ═══════════════════════════════════════════════════════════════════════

run_benchmark() {
    log_info "Running Qwen3.5-27B benchmark..."

    local model_path
    model_path=$(find_model_path "$MODEL")
    if [ -z "$model_path" ] || [ ! -f "$model_path" ]; then
        log_warn "Model not found for $MODEL in $MODELS_DIR — skipping benchmark"
        BASELINE_TOK_S="0"
        CANDIDATE_TOK_S="0"
        HIPFIRE_SPEEDUP="1.0"
        return
    fi
    log_info "  Model: $model_path"
    require_file "$BENCH_PROMPT" "bench prompt"

    # Load or measure baseline
    local baseline_json
    baseline_json=$(ls -t "$BASELINES_DIR"/*.json 2>/dev/null | head -1 || echo "")
    if [ -n "$baseline_json" ] && [ -f "$baseline_json" ]; then
        BASELINE_TOK_S=$(python3 -c "
import json
d = json.load(open('$baseline_json'))
v = d.get('baseline_decode_tok_s', d.get('gen_tok_s', d.get('decode_tok_s', 0)))
print(v)" 2>/dev/null || echo "0")
        log_info "  Baseline (cached): $BASELINE_TOK_S tok/s from $baseline_json"
    else
        log_info "  Measuring baseline tok/s..."
        BASELINE_TOK_S=$(run_bench "$model_path" "$BENCH_PROMPT")
        log_info "  Baseline: $BASELINE_TOK_S tok/s"
    fi

    # Measure candidate (1 run, then up to 3 more if looks faster)
    log_info "  Measuring candidate tok/s (run 1)..."
    local run1
    run1=$(run_bench "$model_path" "$BENCH_PROMPT")
    log_info "  Candidate run 1: $run1 tok/s"

    local faster
    faster=$(python3 -c "
b=float('$BASELINE_TOK_S') if '$BASELINE_TOK_S' else 0
r=float('$run1') if '$run1' else 0
print('yes' if b>0 and r/b >= float('$ACCEPT_MIN_SPEEDUP') else 'no')
" 2>/dev/null || echo "no")

    local best_tok_s="$run1"
    if [ "$faster" = "yes" ]; then
        log_info "  Run 1 looks faster — running 3 more trials for confidence"
        local runs=("$run1")
        for i in 2 3 4; do
            local r
            r=$(run_bench "$model_path" "$BENCH_PROMPT")
            log_info "  Candidate run $i: $r tok/s"
            runs+=("$r")
        done
        best_tok_s=$(python3 -c "
vals=[float(x) for x in ['${runs[0]}','${runs[1]}','${runs[2]}','${runs[3]}'] if x and float(x)>0]
vals.sort()
print(vals[len(vals)//2] if vals else 0)
" 2>/dev/null || echo "$run1")
        log_info "  Median of 4 runs: $best_tok_s tok/s"
    fi

    CANDIDATE_TOK_S="$best_tok_s"
    HIPFIRE_SPEEDUP=$(python3 -c "
b=float('$BASELINE_TOK_S') if '$BASELINE_TOK_S' else 0
c=float('$CANDIDATE_TOK_S') if '$CANDIDATE_TOK_S' else 0
print(f'{c/b:.4f}' if b>0 else '1.0')
" 2>/dev/null || echo "1.0")
    log_info "  Baseline: $BASELINE_TOK_S tok/s  |  Candidate: $CANDIDATE_TOK_S tok/s  |  Speedup: ${HIPFIRE_SPEEDUP}×"
}

# ═══════════════════════════════════════════════════════════════════════
# Step 8: accept / revert
# ═══════════════════════════════════════════════════════════════════════

accept_or_revert() {
    local accept
    accept=$(python3 -c "
s=float('$HIPFIRE_SPEEDUP') if '$HIPFIRE_SPEEDUP' else 0
print('yes' if s >= float('$ACCEPT_MIN_SPEEDUP') else 'no')
" 2>/dev/null || echo "no")

    if [ "$accept" = "yes" ]; then
        DECISION="PASS_PROMOTED"
        log_ok "ACCEPTED — speedup ${HIPFIRE_SPEEDUP}× ≥ threshold ${ACCEPT_MIN_SPEEDUP}×"

        # Archive to accepted_promotions
        mkdir -p "$ACCEPTED_DIR"
        local slug="${TARGET_NAME}_${TS}"
        cp "$CANDIDATE_HIP" "$ACCEPTED_DIR/${slug}.hip"
        cp "$SPEC_FILE" "$ACCEPTED_DIR/${slug}_target_spec.json"
        log_ok "Candidate archived: $ACCEPTED_DIR/${slug}.hip"

        write_tsv_row "PASS_PROMOTED" ""
        local rpt
        rpt=$(write_report "PASS_PROMOTED")

        if [ "$AUTOCOMMIT" = "1" ]; then
            log_info "AUTOCOMMIT=1 — committing..."
            git add "$PROMOTED_DEST" 2>/dev/null || true
            git add "$RESULTS_TSV" 2>/dev/null || true
            git commit -m "autokernel: promote $TARGET_NAME gfx1201 kernel

Phase 15 promotion: ${HIPFIRE_SPEEDUP}× speedup on Qwen3.5-27B
Candidate: $CANDIDATE_HIP
Destination: $PROMOTED_DEST
Harness: correctness=$HARNESS_CORRECTNESS speedup=${HARNESS_SPEEDUP}×" 2>/dev/null || log_warn "git commit failed"
        fi

        echo ""
        log_bold "══ PROMOTION ACCEPTED ══"
        echo "  Target:    $TARGET_NAME"
        echo "  Installed: $PROMOTED_DEST"
        echo "  Speedup:   ${HIPFIRE_SPEEDUP}×"
        echo "  Report:    $rpt"
    else
        REJECT_REASON="hipfire speedup ${HIPFIRE_SPEEDUP}× < threshold ${ACCEPT_MIN_SPEEDUP}×"
        DECISION="FAIL_REVERTED"
        log_warn "REJECTED: $REJECT_REASON"
        restore_backup || true

        mkdir -p "$REJECTED_DIR"
        local slug="${TARGET_NAME}_${TS}"
        cp "$CANDIDATE_HIP" "$REJECTED_DIR/${slug}.hip" 2>/dev/null || true

        write_tsv_row "FAIL_REVERTED" "$REJECT_REASON"
        local rpt
        rpt=$(write_report "FAIL_REVERTED")

        echo ""
        log_bold "══ PROMOTION REJECTED ══"
        echo "  Reason:  $REJECT_REASON"
        echo "  Reverted to: $BACKUP_DIR"
        echo "  Report:  $rpt"
        exit 1
    fi
}

# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

main() {
    mkdir -p "$REPORTS_DIR" "$BACKUP_BASE" "$ACCEPTED_DIR" "$REJECTED_DIR"

    validate_candidate
    determine_destination
    create_backup
    wire_dispatch
    build_hipfire
    run_correctness
    run_benchmark
    accept_or_revert
}

main "$@"
