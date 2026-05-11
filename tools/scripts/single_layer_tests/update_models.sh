#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Copy single-block / single-layer model TTIR fixtures from a regenerated
# directory (typically tt-xla's tests/benchmark/single_layer_tests_<sha>/)
# into test/ttmlir/models/single_blocks_and_layers/.
#
# Accepts every *.mlir matching tt-xla's *_1lyr_* naming or legacy *_block /
# *_layer; all other files are silently skipped.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ -z "${1:-}" || "$1" == -h || "$1" == --help ]]; then
    cat <<EOF
Usage: $0 <source_dir>

Copy *_1lyr_* (and legacy *_block / *_layer) TTIRs from <source_dir> into
test/ttmlir/models/single_blocks_and_layers/. Relative paths are resolved
against the tt-mlir repo root.
EOF
    [[ -z "${1:-}" ]] && exit 1 || exit 0
fi

SOURCE_DIR="$1"
[[ -d "$SOURCE_DIR" ]] || SOURCE_DIR="$REPO_ROOT/$SOURCE_DIR"
[[ -d "$SOURCE_DIR" ]] || { echo "ERROR: '$1' is not a directory." >&2; exit 1; }
SOURCE_DIR="$(cd "$SOURCE_DIR" && pwd)"

MODELS_DIR="$REPO_ROOT/test/ttmlir/models/single_blocks_and_layers"
mkdir -p "$MODELS_DIR"

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
