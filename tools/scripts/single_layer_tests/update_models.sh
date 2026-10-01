#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Copy single-block / single-layer model TTIR fixtures from a regenerated
# directory (typically tt-xla's
# tests/benchmark/single_layer/generated_<sha>/ttir/) into
# test/ttmlir/models/single_blocks_and_layers/.
#
# Steps:
#   1. Resolve <source_dir> (absolute, or relative to the tt-mlir repo root).
#   2. Walk <source_dir>/*.mlir, accept every name matching tt-xla's
#      *_1lyr_* naming or legacy *_block / *_layer; skip everything else.
#   3. Copy each accepted file into test/ttmlir/models/single_blocks_and_layers/
#      (NEW if it didn't exist, UPDATED otherwise) and print a summary.
#
# This script is also called by regen.sh at the end of a successful sweep, so
# it has to be safe to run with no fixtures matching (e.g. a partial run that
# produced nothing) — the loop is silent in that case.

set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODELS_DIR="$REPO_ROOT/test/ttmlir/models/single_blocks_and_layers"

# -----------------------------------------------------------------------------
# Argument parsing + source dir resolution
# -----------------------------------------------------------------------------

if [[ -z "${1:-}" || "$1" == -h || "$1" == --help ]]; then
    cat <<EOF
Usage: $0 <source_dir>

Copy *_1lyr_* (and legacy *_block / *_layer) TTIRs from <source_dir> into
test/ttmlir/models/single_blocks_and_layers/. Relative paths are resolved
against the tt-mlir repo root.
EOF
    # No-arg invocation is an error; -h is help (zero exit).
    [[ -z "${1:-}" ]] && exit 1 || exit 0
fi

# Accept either an absolute path or a path relative to the repo root, so users
# can pass the same string they'd give to regen.sh.
SOURCE_DIR="$1"
[[ -d "$SOURCE_DIR" ]] || SOURCE_DIR="$REPO_ROOT/$SOURCE_DIR"
[[ -d "$SOURCE_DIR" ]] || { echo "ERROR: '$1' is not a directory." >&2; exit 1; }
SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"

mkdir -p "$MODELS_DIR"

# -----------------------------------------------------------------------------
# Walk and copy
# -----------------------------------------------------------------------------

# nullglob makes the loop body skip cleanly when no .mlir files match.
# The name filter rejects unrelated artefacts that tt-xla may also drop in the
# same dir (logs, schemas, etc.) so we never copy junk into the models tree.
new=0; upd=0
shopt -s nullglob
for f in "$SOURCE_DIR"/*.mlir; do
    name="$(basename "$f" .mlir)"
    [[ "$name" == *_1lyr_* || "$name" == *_block || "$name" == *_layer ]] || continue

    if [[ -f "$MODELS_DIR/$name.mlir" ]]; then
        upd=$((upd+1)); status=UPDATED
    else
        new=$((new+1)); status=NEW
    fi
    cp "$f" "$MODELS_DIR/"
    printf "  [%-7s] %s\n" "$status" "$name"
done
shopt -u nullglob

echo
echo "Models NEW=$new UPDATED=$upd"
echo "Models -> $MODELS_DIR"
