#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate single-block / single-layer model fixtures from a tt-xla sweep.
#
# Rebases tt-mlir HEAD onto tt-xla's pinned SHA in a throwaway worktree, pushes
# the rebased branch to 'origin' (must be tenstorrent/tt-mlir), then invokes
# tt-xla/tests/benchmark/single_layer/regen.sh. The sibling update_models.sh
# copies the regenerated TTIRs into test/ttmlir/models/single_blocks_and_layers/.
# Lit test files are NOT touched here — use the sibling update_lit_tests.sh.
#
# --local skips the push (offline / no push access); --delete-branch removes
# the pushed branch on success (default keeps it, e.g. for CI handoff). The
# branch is never deleted on failure regardless of flags. See ./README.md for
# the full workflow.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TTMLIR_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TTXLA_REMOTE="https://github.com/tenstorrent/tt-xla.git"

SUBSET=""; SKIP_LIT=0; TTXLA_DIR=""; USE_LOCAL=0; DELETE_BRANCH=0
while (( $# )); do
    case "$1" in
        --subset)        SUBSET="$2"; shift 2 ;;
        --ttxla-dir)     TTXLA_DIR="$2"; shift 2 ;;
        --skip-lit)      SKIP_LIT=1; shift ;;
        --local)         USE_LOCAL=1; shift ;;
        --delete-branch) DELETE_BRANCH=1; shift ;;
        -h|--help)
            cat <<EOF
Usage: $0 [--subset name[,...]] [--ttxla-dir dir] [--skip-lit] [--local] [--delete-branch]

  --subset         tt-xla subset (single,llmbox,galaxy). Default: single.
  --ttxla-dir      tt-xla checkout. Default: ../tt-xla, else clone main into /tmp.
  --skip-lit       Don't run lit / ttrt at the end.
  --local          Skip the push; pass TT_MLIR_LOCAL_PATH=<this checkout>
                   to tt-xla instead. Use when 'origin' isn't
                   tenstorrent/tt-mlir or when offline.
  --delete-branch  Delete the pushed branch on success. Default: keep.
                   Never deletes on failure.

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

# Default flow pushes to origin; tt-xla clones tt-mlir from a fixed URL, so
# origin must resolve to tenstorrent/tt-mlir for the SHA to be fetchable.
if (( ! USE_LOCAL )); then
    ORIGIN_URL="$(git -C "$TTMLIR_ROOT" remote get-url origin 2>/dev/null || true)"
    case "$ORIGIN_URL" in
        *tenstorrent/tt-mlir|*tenstorrent/tt-mlir.git) ;;
        *) echo "ERROR: 'origin' must point at tenstorrent/tt-mlir (got: '$ORIGIN_URL')." >&2
           echo "       Re-run with --local to skip the push." >&2
           exit 1 ;;
    esac
fi

# Resolve tt-xla: --ttxla-dir > ../tt-xla > shallow clone of main.
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

BASE="$(sed -nE 's/.*set\(TT_MLIR_VERSION[^"]*"([0-9a-fA-F]+)".*/\1/p' \
    "$TTXLA_DIR/third_party/CMakeLists.txt" | head -n1)"
[[ -n "$BASE" ]] || { echo "ERROR: couldn't read TT_MLIR_VERSION from tt-xla." >&2; exit 1; }

cd "$TTMLIR_ROOT"
ORIG_HEAD="$(git rev-parse HEAD)"
ORIG_LABEL="$(git branch --show-current || true)"
ORIG_LABEL="${ORIG_LABEL:-detached-${ORIG_HEAD:0:8}}"
# Sanitize for use in a ref name (whitespace and ~^: aren't valid).
ORIG_LABEL="${ORIG_LABEL//[^A-Za-z0-9._\/-]/-}"
REGEN_BRANCH="regen/${ORIG_LABEL}-on-${BASE:0:8}"
WORKTREE="/tmp/ttmlir-regen-$$"

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "WARNING: uncommitted changes will not be rebased."
fi
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

# Cleanup: worktree always removed; pushed branch deleted only when
# --delete-branch is set AND the build succeeded.
PUSHED=0; BUILD_RC=1
cleanup() {
    git worktree remove --force "$WORKTREE" 2>/dev/null || true
    if (( PUSHED )); then
        if (( DELETE_BRANCH && BUILD_RC == 0 )); then
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

if ! git -C "$WORKTREE" rebase "$BASE"; then
    trap - EXIT
    echo "ERROR: rebase conflicts. Inspect/abort in $WORKTREE." >&2
    exit 1
fi

REBASED_SHA="$(git -C "$WORKTREE" rev-parse HEAD)"
echo "Rebased HEAD: $REBASED_SHA"

if (( USE_LOCAL )); then
    # Worktrees share objects with the parent checkout, so passing
    # TT_MLIR_LOCAL_PATH lets tt-xla fetch the rebased SHA locally.
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

TT_MLIR_COMMIT_OVERRIDE="$REBASED_SHA" \
TT_MLIR_LOCAL_PATH="$TT_MLIR_LOCAL_PATH_ARG" \
SUBSET="$SUBSET" \
"$TTXLA_DIR/tests/benchmark/single_layer/regen.sh"
BUILD_RC=$?
(( BUILD_RC == 0 )) || echo "WARNING: tt-xla regen.sh exited $BUILD_RC."

# Copy refreshed TTIRs into the tt-mlir models tree.
TTIRS_DIR="$TTXLA_DIR/tests/benchmark/single_layer/generated_${REBASED_SHA:0:8}/ttir"
"$SCRIPT_DIR/update_models.sh" "$TTIRS_DIR" \
    || echo "WARNING: update_models.sh exited non-zero."

if (( SKIP_LIT )); then
    echo "--skip-lit: skipping lit / ttrt run."
else
    LIT_BIN="$(command -v llvm-lit || true)"
    [[ -z "$LIT_BIN" && -x "$TTMLIR_ROOT/build/bin/llvm-lit" ]] && \
        LIT_BIN="$TTMLIR_ROOT/build/bin/llvm-lit"
    TTRT_BIN="$(command -v ttrt || true)"
    # lit writes .ttnn flatbuffers under the build-tree mirror; ttrt reads them
    # from there too. Mirror CI's per-device optimizer test path.
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
