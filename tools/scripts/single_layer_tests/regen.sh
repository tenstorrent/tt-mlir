#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate single-block / single-layer model fixtures from a tt-xla sweep.
#
# tt-xla owns the generation side: a pytest sweep produces TTIR fixtures pinned
# to a specific tt-mlir SHA. To refresh fixtures we have to rebase the tt-mlir
# change we want validated onto that pinned SHA, hand the rebased SHA to
# tt-xla, run the sweep, and copy the result back. This script wraps that
# whole loop into one command.
#
# Steps:
#   1. Rebase tt-mlir HEAD onto tt-xla's pinned SHA in a throwaway worktree
#      (the working tree is never touched).
#   2. Push the rebased branch to 'origin' (must be tenstorrent/tt-mlir), or
#      skip the push with --local and pass TT_MLIR_LOCAL_PATH to tt-xla.
#   3. Invoke tt-xla/tests/benchmark/single_layer/regen.sh against that SHA.
#   4. Copy regenerated TTIRs into test/ttmlir/models/single_blocks_and_layers/
#      via the sibling update_models.sh. Lit test files themselves are NOT
#      touched — use update_lit_tests.sh to scaffold new ones.
#   5. Run llvm-lit + ttrt over the per-device build-tree mirror dirs (skip
#      with --skip-lit).
#
# The pushed branch is deleted on success by default; pass --keep-branch to
# keep it (e.g. for CI handoff). The branch is never deleted on failure
# regardless of flags. See ./README.md for the full workflow.

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTMLIR_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TTXLA_REMOTE="https://github.com/tenstorrent/tt-xla.git"

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------

SUBSET=""; SKIP_LIT=0; TTXLA_DIR=""; USE_LOCAL=0; KEEP_BRANCH=0
while (( $# )); do
    case "$1" in
        --subset)        SUBSET="$2"; shift 2 ;;
        --ttxla-dir)     TTXLA_DIR="$2"; shift 2 ;;
        --skip-lit)      SKIP_LIT=1; shift ;;
        --local)         USE_LOCAL=1; shift ;;
        --keep-branch)   KEEP_BRANCH=1; shift ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--subset name[,...]] [--ttxla-dir dir] [--skip-lit] [--local] [--keep-branch]

  --subset         tt-xla subset (single,llmbox,galaxy). Default: empty
                   (delegates to tt-xla's default, currently 'single').
  --ttxla-dir      tt-xla checkout. Default: ../tt-xla, else clone main into /tmp.
  --skip-lit       Don't run lit / ttrt at the end.
  --local          Skip the push; pass TT_MLIR_LOCAL_PATH=<this checkout>
                   to tt-xla instead. Use when 'origin' isn't
                   tenstorrent/tt-mlir or when offline.
  --keep-branch    Keep the pushed branch on success. Default: delete on
                   success. Failures always keep the branch.

Refreshes model fixtures only. To add lit test files for new fixtures, run
  tools/scripts/single_layer_tests/update_lit_tests.sh <fixture_name> [<name> ...]
(e.g. mistral_7b_1lyr_bs1_decode) — the basename of a file under
test/ttmlir/models/single_blocks_and_layers/.

HF_TOKEN is read from the environment; if unset, tt-xla's regen.sh prompts
(interactive only).
EOF
            exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

# -----------------------------------------------------------------------------
# Pre-flight: validate every external dependency before touching state.
# -----------------------------------------------------------------------------

# tt-xla's regen.sh clones tt-mlir from a fixed github URL, so the SHA we push
# must be reachable there. --local bypasses this by handing tt-xla a local path.
if (( ! USE_LOCAL )); then
    ORIGIN_URL="$(git -C "$TTMLIR_ROOT" remote get-url origin 2>/dev/null || true)"
    case "$ORIGIN_URL" in
        *tenstorrent/tt-mlir|*tenstorrent/tt-mlir.git) ;;
        *) echo "ERROR: 'origin' must point at tenstorrent/tt-mlir (got: '$ORIGIN_URL')." >&2
           echo "       Re-run with --local to skip the push." >&2
           exit 1 ;;
    esac
fi

# Resolve tt-xla checkout. Precedence:
#   1. --ttxla-dir <path>  — explicit override.
#   2. ../tt-xla           — sibling checkout (the common dev layout).
#   3. shallow clone of main into /tmp — last resort, throwaway.
SIBLING="$(cd "$TTMLIR_ROOT/.." && pwd)/tt-xla"
if [[ -n "$TTXLA_DIR" ]]; then
    [[ -f "$TTXLA_DIR/third_party/CMakeLists.txt" ]] \
        || { echo "ERROR: '$TTXLA_DIR' is not a tt-xla checkout." >&2; exit 1; }
    TTXLA_DIR="$(cd "$TTXLA_DIR" && pwd)"
elif [[ -f "$SIBLING/third_party/CMakeLists.txt" ]]; then
    TTXLA_DIR="$SIBLING"
else
    TTXLA_DIR="/tmp/ttxla-regen-$$/tt-xla"
    mkdir -p "$(dirname "$TTXLA_DIR")"
    git clone --depth=1 --branch main "$TTXLA_REMOTE" "$TTXLA_DIR"
fi

# tt-xla pins the tt-mlir SHA it builds against in its CMakeLists; that's the
# SHA we have to rebase onto so the rebased commit is a fast-forward of it.
BASE="$(sed -nE 's/.*set\(TT_MLIR_VERSION[^"]*"([0-9a-fA-F]+)".*/\1/p' \
    "$TTXLA_DIR/third_party/CMakeLists.txt" | head -n1)"
[[ -n "$BASE" ]] || { echo "ERROR: couldn't read TT_MLIR_VERSION from tt-xla." >&2; exit 1; }

# -----------------------------------------------------------------------------
# Branch identity + worktree setup
# -----------------------------------------------------------------------------

cd "$TTMLIR_ROOT"
ORIG_HEAD="$(git rev-parse HEAD)"
ORIG_LABEL="$(git branch --show-current || true)"
ORIG_LABEL="${ORIG_LABEL:-detached-${ORIG_HEAD:0:8}}"
# Strip characters that aren't valid in a git ref name.
ORIG_LABEL="${ORIG_LABEL//[^A-Za-z0-9._\/-]/-}"
# Branch name encodes both endpoints so concurrent regens don't collide and so
# operators can tell at a glance which HEAD was rebased onto which base.
REGEN_BRANCH="regen/${ORIG_LABEL}-on-${BASE:0:8}"
WORKTREE="/tmp/ttmlir-regen-$$"

# The rebase happens in a throwaway worktree from ORIG_HEAD, so anything not
# committed is silently dropped from the regen — call that out up front.
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "WARNING: uncommitted changes will not be rebased."
fi
# Best-effort cleanup of stale state from a prior interrupted run.
git worktree remove --force "$WORKTREE" 2>/dev/null || true
git branch -D "$REGEN_BRANCH" 2>/dev/null || true

cat <<EOF
=============================================
tt-mlir HEAD:   $ORIG_HEAD ($ORIG_LABEL)
tt-xla BASE:    $BASE
regen branch:   $REGEN_BRANCH
regen worktree: $WORKTREE
tt-xla dir:     $TTXLA_DIR
Subset:         ${SUBSET:-<default>}
Mode:           $( (( USE_LOCAL )) && echo "local (TT_MLIR_LOCAL_PATH)" || echo "push (origin/$REGEN_BRANCH)" )
=============================================
EOF

# Cleanup trap.
#   - The local worktree is always removed.
#   - The pushed branch is deleted on success unless --keep-branch is set;
#     on failure we always leave it so users can inspect or re-trigger CI
#     from it.
# BUILD_RC starts at 1 so any early failure (between push and the tt-xla
# invocation) keeps the branch around.
PUSHED=0; BUILD_RC=1
cleanup() {
    git worktree remove --force "$WORKTREE" 2>/dev/null || true
    if (( PUSHED )); then
        if (( ! KEEP_BRANCH && BUILD_RC == 0 )); then
            git push origin --delete "$REGEN_BRANCH" 2>/dev/null \
                && echo "Cleaned up remote branch: $REGEN_BRANCH" \
                || echo "WARNING: failed to delete remote branch: $REGEN_BRANCH"
        else
            echo "Branch on origin: $REGEN_BRANCH"
            echo "  Delete with: git push origin --delete $REGEN_BRANCH"
        fi
    fi
}

git worktree add -b "$REGEN_BRANCH" "$WORKTREE" "$ORIG_HEAD" \
    || { echo "ERROR: git worktree add failed." >&2; exit 1; }
trap cleanup EXIT

# -----------------------------------------------------------------------------
# Step 1: Rebase tt-mlir HEAD onto tt-xla's pinned SHA
# -----------------------------------------------------------------------------

# Conflicts here mean the operator's branch needs a manual rebase; we leave the
# worktree intact and disarm the trap so the user can finish it by hand.
if ! git -C "$WORKTREE" rebase "$BASE"; then
    trap - EXIT
    echo "ERROR: rebase conflicts. Inspect/abort in $WORKTREE." >&2
    exit 1
fi

REBASED_SHA="$(git -C "$WORKTREE" rev-parse HEAD)"
echo "Rebased HEAD: $REBASED_SHA"

# -----------------------------------------------------------------------------
# Step 2: Hand the rebased SHA to tt-xla (push, or --local)
# -----------------------------------------------------------------------------

if (( USE_LOCAL )); then
    # Worktrees share git objects with the parent checkout, so pointing tt-xla
    # at the parent via TT_MLIR_LOCAL_PATH is enough to fetch the rebased SHA.
    TT_MLIR_LOCAL_PATH_ARG="$TTMLIR_ROOT"
else
    echo "Pushing $REGEN_BRANCH to origin..."
    if ! git -C "$WORKTREE" push --force origin "$REGEN_BRANCH"; then
        echo "ERROR: push failed. Re-run with --local to skip the push." >&2
        exit 1
    fi
    PUSHED=1
    TT_MLIR_LOCAL_PATH_ARG=""
fi

# -----------------------------------------------------------------------------
# Step 3: Invoke tt-xla's single-layer regen against the rebased SHA
# -----------------------------------------------------------------------------

# Best-effort: a non-zero exit warns but doesn't abort, so any partial fixture
# set we did get can still be copied to the models tree (step 4).
BUILD_RC=0
TT_MLIR_COMMIT_OVERRIDE="$REBASED_SHA" \
TT_MLIR_LOCAL_PATH="$TT_MLIR_LOCAL_PATH_ARG" \
SUBSET="$SUBSET" \
    "$TTXLA_DIR/tests/benchmark/single_layer/regen.sh" || BUILD_RC=$?
(( BUILD_RC == 0 )) || echo "WARNING: tt-xla regen.sh exited $BUILD_RC."

# -----------------------------------------------------------------------------
# Step 4: Copy regenerated TTIRs into the models tree
# -----------------------------------------------------------------------------

TTIRS_DIR="$TTXLA_DIR/tests/benchmark/single_layer/generated_${REBASED_SHA:0:8}/ttir"
"$SCRIPT_DIR/update_models.sh" "$TTIRS_DIR" \
    || echo "WARNING: update_models.sh exited non-zero."

# -----------------------------------------------------------------------------
# Step 5: Smoke-test via llvm-lit + ttrt over the build-tree mirror
# -----------------------------------------------------------------------------

# Mirrors the path CI takes for these tests (op_model_ttrt.sh): lit drives
# ttmlir-opt + ttmlir-translate to produce .ttnn flatbuffers under the build
# tree, then ttrt runs them on silicon. Failures here are report-only — they
# don't fail the regen, since the fresh fixtures are already on disk.
if (( SKIP_LIT )); then
    echo "--skip-lit: skipping lit / ttrt run."
else
    LIT_BIN="$(command -v llvm-lit || true)"
    [[ -z "$LIT_BIN" && -x "$TTMLIR_ROOT/build/bin/llvm-lit" ]] && \
        LIT_BIN="$TTMLIR_ROOT/build/bin/llvm-lit"
    TTRT_BIN="$(command -v ttrt || true)"
    BUILD_LIT_BASE="$TTMLIR_ROOT/build/test/ttmlir/Silicon/TTNN"
    LIT_TAIL="optimizer/single_block_layer_perf_tests"
    if [[ -z "$LIT_BIN" ]]; then
        echo "WARNING: llvm-lit not found; skipping lit / ttrt."
    elif [[ ! -d "$BUILD_LIT_BASE" ]]; then
        echo "WARNING: $BUILD_LIT_BASE not found (build first); skipping lit / ttrt."
    else
        for d in n150 llmbox galaxy; do
            LIT_DIR="$BUILD_LIT_BASE/$d/$LIT_TAIL"
            [[ -d "$LIT_DIR" ]] || continue
            echo "  lit: $LIT_DIR"
            "$LIT_BIN" -v "$LIT_DIR" || echo "  WARNING: lit failed in $LIT_DIR (report-only)."
            if [[ -n "$TTRT_BIN" ]]; then
                echo "  ttrt run: $LIT_DIR"
                "$TTRT_BIN" run "$LIT_DIR" || echo "  WARNING: ttrt run failed in $LIT_DIR (report-only)."
            else
                echo "  WARNING: ttrt not found; skipping ttrt run for $LIT_DIR."
            fi
        done
    fi
fi
echo "Done."
