#!/usr/bin/env bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# bisect_metal.sh - Binary search for the tt-metal commit that introduced a regression.
#
# Usage:
#   ./tools/scripts/bisect_metal.sh <good_commit> <bad_commit> "<test_command>"
#
# Example:
#   CLAUDE_AUTO_FIX=1 ./tools/scripts/bisect_metal.sh 90c914ef258b cc45c083679 \
#     "source env/activate && pytest -svv 'test/python/golden/test_ttir_ops.py::test_collective_permute[f32-mesh_shape1-source_target_pairs1-30x60]'"
#   ./tools/scripts/bisect_metal.sh 90c914ef258b cc45c083679 \
#     "source env/activate && pytest -svv 'test/python/golden/test_ttir_ops.py::test_collective_permute[f32-mesh_shape1-source_target_pairs1-30x60]'"
#
# The script will:
#   1. Enumerate commits between good..bad in the tt-metal repo
#   2. Binary search: for each candidate, update tt-metal, rebuild, and run the test
#   3. If tt-mlir fails to build against a metal commit, optionally invoke
#      Claude Code to attempt an automatic fix
#   4. Report the first bad commit
#
# Environment variables:
#   TT_MLIR_ROOT      - Path to tt-mlir repo (default: script's grandparent dir)
#   BUILD_DIR         - Build directory (default: $TT_MLIR_ROOT/build)
#   CLAUDE_AUTO_FIX   - Set to "1" to enable Claude Code auto-fix on build breaks (default: 0)
#   BISECT_LOG        - Log file path (default: $TT_MLIR_ROOT/bisect_metal.log)
#   BUILD_TIMEOUT     - Timeout in seconds for the build step (default: 1800 = 30min)
#   TEST_TIMEOUT      - Timeout in seconds for the test step (default: 300 = 5min)
#   VERBOSE           - Set to "1" to print build output to terminal (default: 0, log-only)
#   RESET_DEVICE      - Set to "1" to run "tt-smi -r" after test failures (default: 0)

set -euo pipefail

# --- Configuration -----------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_MLIR_ROOT="${TT_MLIR_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

# Activate the project's virtual environment
# Temporarily disable nounset - env/activate checks unset vars with -z
set +u
cd "$TT_MLIR_ROOT"
source "$TT_MLIR_ROOT/env/activate"
set -u
BUILD_DIR="${BUILD_DIR:-$TT_MLIR_ROOT/build}"
CLAUDE_AUTO_FIX="${CLAUDE_AUTO_FIX:-0}"
BISECT_LOG="${BISECT_LOG:-$TT_MLIR_ROOT/bisect_metal.log}"
BUILD_TIMEOUT="${BUILD_TIMEOUT:-1800}"
TEST_TIMEOUT="${TEST_TIMEOUT:-300}"
VERBOSE="${VERBOSE:-0}"
RESET_DEVICE="${RESET_DEVICE:-0}"

TT_METAL_DIR="$TT_MLIR_ROOT/third_party/tt-metal/src/tt-metal"
THIRD_PARTY_CMAKE="$TT_MLIR_ROOT/third_party/CMakeLists.txt"

# Colors for terminal output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# --- Functions ---------------------------------------------------------------

log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    echo -e "$msg"
    echo "$msg" >> "$BISECT_LOG"
}

die() {
    log "${RED}ERROR: $*${NC}"
    exit 1
}

usage() {
    echo "Usage: $0 <good_commit> <bad_commit> <test_command>"
    echo ""
    echo "Arguments:"
    echo "  good_commit   - tt-metal commit hash where the test passes"
    echo "  bad_commit    - tt-metal commit hash where the test fails"
    echo "  test_command  - Shell command to run the test (exit 0 = pass, non-zero = fail)"
    echo ""
    echo "Environment variables:"
    echo "  CLAUDE_AUTO_FIX=1  - Enable Claude Code to auto-fix build breaks"
    echo "  BUILD_TIMEOUT=1800 - Build timeout in seconds (default: 30min)"
    echo "  TEST_TIMEOUT=300   - Test timeout in seconds (default: 5min)"
    echo "  VERBOSE=1          - Print build output to terminal (default: log-only)"
    echo "  RESET_DEVICE=1     - Run 'tt-smi -r' after test failures (default: off)"
    exit 1
}

# Update TT_METAL_VERSION in third_party/CMakeLists.txt
set_metal_version() {
    local commit="$1"
    log "${CYAN}Setting TT_METAL_VERSION to $commit${NC}"
    sed -i "s|^set(TT_METAL_VERSION \"[a-f0-9]*\")|set(TT_METAL_VERSION \"$commit\")|" "$THIRD_PARTY_CMAKE"
}

# Checkout tt-metal to a specific commit (faster than letting CMake re-fetch)
checkout_metal() {
    local commit="$1"
    log "${CYAN}Checking out tt-metal to $commit${NC}"
    git -C "$TT_METAL_DIR" fetch origin "$commit" --depth=1 2>/dev/null || \
        git -C "$TT_METAL_DIR" fetch origin 2>/dev/null || true
    git -C "$TT_METAL_DIR" checkout "$commit" --force 2>&1 || {
        log "${YELLOW}Direct checkout failed, trying fetch + checkout${NC}"
        git -C "$TT_METAL_DIR" fetch origin --unshallow 2>/dev/null || true
        git -C "$TT_METAL_DIR" checkout "$commit" --force
    }
    # Also update the CMakeLists so CMake doesn't try to re-fetch
    set_metal_version "$commit"
}

# Build tt-mlir. Returns 0 on success, 1 on failure.
# On failure, stores the build log in $TT_MLIR_ROOT/bisect_build.log
build_tt_mlir() {
    local build_log="$TT_MLIR_ROOT/bisect_build.log"
    log "${CYAN}Building tt-mlir...${NC}"
    log "${CYAN}Build progress: tail -f $build_log${NC}"

    # Reconfigure to pick up the new tt-metal (skip the download/update steps
    # since we already checked out manually)
    local build_ok=0
    if [ "$VERBOSE" = "1" ]; then
        timeout "$BUILD_TIMEOUT" cmake --build "$BUILD_DIR" -- -j$(nproc) 2>&1 | tee "$build_log" || build_ok=$?
    else
        timeout "$BUILD_TIMEOUT" cmake --build "$BUILD_DIR" -- -j$(nproc) > "$build_log" 2>&1 || build_ok=$?
    fi

    if [ "$build_ok" -ne 0 ]; then
        log "${RED}Build failed. See: $build_log${NC}"
        return 1
    fi

    log "${GREEN}Build succeeded.${NC}"
    return 0
}

# Attempt to fix a build break using Claude Code
try_claude_fix() {
    local build_log="$TT_MLIR_ROOT/bisect_build.log"
    local error_tail
    error_tail=$(tail -80 "$build_log" 2>/dev/null || echo "No build log available")

    local claude_log="$TT_MLIR_ROOT/bisect_claude.log"
    log "${YELLOW}Invoking Claude Code to fix build break...${NC}"
    log "${YELLOW}Claude progress: tail -f $claude_log${NC}"

    # Create a prompt file for claude
    local prompt_file
    prompt_file=$(mktemp /tmp/bisect_claude_prompt.XXXXXX)
    cat > "$prompt_file" <<PROMPT
The tt-mlir build is broken after switching to a different tt-metal commit during a bisect operation.
This is a temporary build fix needed only for bisect testing - make minimal changes to get it compiling.

Build error (last 80 lines):
$error_tail

Please fix the build error with minimal changes. Only modify tt-mlir files (not third_party/).
Focus on:
- API signature changes (added/removed/renamed parameters)
- Missing includes
- Type changes
- Renamed symbols

Do NOT modify files under third_party/.
After making changes, run: cmake --build $BUILD_DIR -- -j\$(nproc)
PROMPT

    # Run claude code with the prompt
    # Output streams to bisect_claude.log so you can: tail -f bisect_claude.log
    if command -v claude &>/dev/null; then
        local fix_result=0
        timeout 600 claude --permission-mode auto --output-format stream-json \
            --print "$(<"$prompt_file")" 2>&1 | tee "$claude_log" | tee -a "$BISECT_LOG" || fix_result=$?
        rm -f "$prompt_file"

        if [ "$fix_result" -eq 0 ]; then
            # Try building again after claude's fix
            log "${CYAN}Retrying build after Claude fix...${NC}"
            if build_tt_mlir; then
                log "${GREEN}Claude fix succeeded!${NC}"
                return 0
            fi
        fi
        log "${RED}Claude fix did not resolve the build break.${NC}"
    else
        log "${RED}claude command not found. Install Claude Code CLI to enable auto-fix.${NC}"
        rm -f "$prompt_file"
    fi
    return 1
}

# Run the test command. Returns 0 if test passes (commit is good), 1 if it fails (commit is bad).
run_test() {
    local test_cmd="$1"
    log "${CYAN}Running test: $test_cmd${NC}"

    if timeout "$TEST_TIMEOUT" bash -c "cd '$TT_MLIR_ROOT' && $test_cmd" 2>&1 | tee -a "$BISECT_LOG"; then
        log "${GREEN}Test PASSED${NC}"
        return 0
    else
        log "${RED}Test FAILED${NC}"
        if [ "$RESET_DEVICE" = "1" ]; then
            log "${YELLOW}Resetting device with tt-smi -r...${NC}"
            if ! tt-smi -r 2>&1 | tee -a "$BISECT_LOG"; then
                log "${RED}tt-smi -r failed${NC}"
            fi
        fi
        return 1
    fi
}

# --- Main --------------------------------------------------------------------

if [ "$#" -lt 3 ]; then
    usage
fi

GOOD_COMMIT="$1"
BAD_COMMIT="$2"
TEST_CMD="$3"

# Fail fast if the tt-mlir working tree has uncommitted changes, since the
# bisect loop resets the tree (git checkout -- .) after each iteration.
if ! git -C "$TT_MLIR_ROOT" diff --quiet HEAD 2>/dev/null || \
   ! git -C "$TT_MLIR_ROOT" diff --cached --quiet HEAD 2>/dev/null; then
    die "tt-mlir working tree has uncommitted changes. Please commit or stash them before running bisect."
fi

# Save original state to restore later
ORIGINAL_METAL_VERSION=$(grep -oP 'set\(TT_METAL_VERSION "\K[a-f0-9]+' "$THIRD_PARTY_CMAKE" | head -1)
ORIGINAL_BRANCH=$(git -C "$TT_MLIR_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo "unknown")

# Ensure we restore state on exit (normal, error, or interrupt)
cleanup() {
    local exit_code=$?
    log ""
    log "Restoring original TT_METAL_VERSION ($ORIGINAL_METAL_VERSION)..."
    set_metal_version "$ORIGINAL_METAL_VERSION" 2>/dev/null || true
    checkout_metal "$ORIGINAL_METAL_VERSION" 2>/dev/null || true
    git -C "$TT_MLIR_ROOT" checkout -- . 2>/dev/null || true
    if [ "$exit_code" -ne 0 ]; then
        log "${RED}Bisect exited early (code $exit_code). State has been restored.${NC}"
    fi
}
trap cleanup EXIT INT TERM

log "========================================================"
log "tt-metal bisect starting"
log "========================================================"
log "  Good commit : $GOOD_COMMIT"
log "  Bad commit  : $BAD_COMMIT"
log "  Test command: $TEST_CMD"
log "  tt-mlir root: $TT_MLIR_ROOT"
log "  Build dir   : $BUILD_DIR"
log "  Auto-fix    : $CLAUDE_AUTO_FIX"
log "  Log file    : $BISECT_LOG"
log "========================================================"

# Ensure tt-metal repo has the full history between good and bad
log "Ensuring tt-metal history is available..."
if git -C "$TT_METAL_DIR" rev-parse --is-shallow-repository | grep -q true; then
    log "tt-metal repo is shallow, unshallowing to get full history..."
    git -C "$TT_METAL_DIR" fetch --unshallow origin 2>&1 || true
fi
git -C "$TT_METAL_DIR" fetch origin 2>/dev/null || true

# Collect the list of commits between good and bad (oldest first)
mapfile -t COMMITS < <(git -C "$TT_METAL_DIR" rev-list --reverse "$GOOD_COMMIT..$BAD_COMMIT")
TOTAL=${#COMMITS[@]}

if [ "$TOTAL" -eq 0 ]; then
    die "No commits found between $GOOD_COMMIT and $BAD_COMMIT. Check commit order and ensure both exist in tt-metal."
fi

log "Found $TOTAL commits between good and bad."

# --- Reset device for clean state --------------------------------------------

log ""
log "Resetting device for clean state before bisect..."
if tt-smi -r 2>&1 | tee -a "$BISECT_LOG"; then
    log "${GREEN}Device reset successful.${NC}"
else
    log "${YELLOW}tt-smi -r failed, continuing anyway...${NC}"
fi

# --- Verify the bad commit actually fails ------------------------------------

log ""
log "Verifying bad commit (${BAD_COMMIT:0:12}) actually fails the test..."
checkout_metal "$BAD_COMMIT"
if build_tt_mlir && run_test "$TEST_CMD"; then
    git -C "$TT_MLIR_ROOT" checkout -- . 2>/dev/null || true
    die "Bad commit ${BAD_COMMIT:0:12} passed the test! Check that your good/bad commits are correct."
fi
git -C "$TT_MLIR_ROOT" checkout -- . 2>/dev/null || true
log "${GREEN}Confirmed: bad commit fails as expected.${NC}"

# --- Verify the good commit actually passes ----------------------------------

log ""
log "Verifying good commit (${GOOD_COMMIT:0:12}) actually passes the test..."
checkout_metal "$GOOD_COMMIT"
if build_tt_mlir && run_test "$TEST_CMD"; then
    git -C "$TT_MLIR_ROOT" checkout -- . 2>/dev/null || true
    log "${GREEN}Confirmed: good commit passes as expected.${NC}"
else
    git -C "$TT_MLIR_ROOT" checkout -- . 2>/dev/null || true
    die "Good commit ${GOOD_COMMIT:0:12} failed the test! Check that your good/bad commits are correct."
fi

# --- Binary search -----------------------------------------------------------

lo=0
hi=$((TOTAL - 1))
# Track the first known bad index (-1 = not yet found)
first_bad_idx=-1

while [ "$lo" -le "$hi" ]; do
    mid=$(( (lo + hi) / 2 ))
    commit="${COMMITS[$mid]}"
    remaining=$(( hi - lo + 1 ))

    # Compute log2 steps remaining using pure bash
    steps_left=0
    tmp_remaining=$remaining
    while [ "$tmp_remaining" -gt 1 ]; do
        tmp_remaining=$(( tmp_remaining / 2 ))
        steps_left=$(( steps_left + 1 ))
    done

    log ""
    log "========================================================"
    log "BISECT STEP: testing commit $((mid + 1))/$TOTAL (${commit:0:12})"
    log "  Range: [$lo..$hi] ($remaining commits remaining, ~$steps_left steps left)"
    log "========================================================"

    # 1. Checkout metal to this commit
    checkout_metal "$commit"

    # 2. Build tt-mlir
    build_ok=true
    if ! build_tt_mlir; then
        build_ok=false
        if [ "$CLAUDE_AUTO_FIX" = "1" ]; then
            if try_claude_fix; then
                build_ok=true
            fi
        fi
    fi

    if [ "$build_ok" = false ]; then
        log "${YELLOW}SKIP: Build failed for ${commit:0:12}, treating as bad (build break).${NC}"
        # Build break = bad, narrow right half
        first_bad_idx=$mid
        hi=$((mid - 1))

        # Restore tt-mlir source in case claude made changes
        git -C "$TT_MLIR_ROOT" checkout -- . 2>/dev/null || true
        continue
    fi

    # 3. Run the test
    if run_test "$TEST_CMD"; then
        # Test passed => this commit is good, search later
        lo=$((mid + 1))
    else
        # Test failed => this commit is bad, search earlier
        first_bad_idx=$mid
        hi=$((mid - 1))
    fi

    # Restore any source changes made by claude
    git -C "$TT_MLIR_ROOT" checkout -- . 2>/dev/null || true
done

# --- Report results ----------------------------------------------------------

if [ "$first_bad_idx" -eq -1 ]; then
    log ""
    log "========================================================"
    log "${GREEN}BISECT RESULT: No bad commit found — all commits in range passed!${NC}"
    log "========================================================"
else
    BAD_COMMIT_HASH="${COMMITS[$first_bad_idx]}"
    log ""
    log "========================================================"
    log "${RED}BISECT RESULT: First bad tt-metal commit:${NC}"
    log "========================================================"
    log ""
    git -C "$TT_METAL_DIR" log --oneline -1 "$BAD_COMMIT_HASH"
    log ""
    git -C "$TT_METAL_DIR" log --format=fuller -1 "$BAD_COMMIT_HASH" 2>&1 | tee -a "$BISECT_LOG"
    log ""
    log "========================================================"
fi

log "Bisect complete. Full log at: $BISECT_LOG"
