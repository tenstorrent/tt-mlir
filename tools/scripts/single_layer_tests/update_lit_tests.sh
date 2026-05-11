#!/bin/bash
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
#
# Manage lit test files for single-block / single-layer model fixtures in
# test/ttmlir/models/single_blocks_and_layers/.
#
# Two modes:
#   * <fixture_name> [<name>...] (default) Create a lit file for each listed
#                                fixture. Each name must correspond to a file
#                                under single_blocks_and_layers/. Existing
#                                lit files are never overwritten (HAVE).
#   * --status                   Read-only: walk every fixture and report
#                                HAVE / MISSING per file. Nothing modified.
#
# Each fixture is routed to a per-device lit dir under
# test/ttmlir/Silicon/TTNN/{n150,llmbox,galaxy}/optimizer/single_block_layer_perf_tests/:
#   *galaxy*  -> galaxy
#   *tp*      -> llmbox
#   else      -> n150

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

STATUS_ONLY=0
if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    cat <<EOF
Usage: $0 <fixture_name> [<name> ...]    # create lit files for these fixtures
       $0 --status                       # report HAVE / MISSING per fixture

Each <fixture_name> is the basename of a file under
  test/ttmlir/models/single_blocks_and_layers/
(with or without the .mlir suffix), e.g. mistral_7b_1lyr_bs1_decode or
llama_3_1_8b_1lyr_bs1_prefill_isl128 — NOT a subset (single/llmbox/galaxy)
and NOT a high-level model name (mistral_7b). Existing lit files are never
overwritten.
EOF
    exit 0
elif [[ "${1:-}" == --status ]]; then
    (( $# > 1 )) && { echo "ERROR: --status takes no other arguments." >&2; exit 2; }
    STATUS_ONLY=1
elif (( $# == 0 )); then
    echo "ERROR: no fixtures given. Pass fixture names to create lit files," >&2
    echo "       or pass --status to report HAVE/MISSING. See -h for usage." >&2
    exit 2
fi

MODELS_DIR="$REPO_ROOT/test/ttmlir/models/single_blocks_and_layers"
LIT_BASE="$REPO_ROOT/test/ttmlir/Silicon/TTNN"
LIT_TAIL="optimizer/single_block_layer_perf_tests"
N150="$LIT_BASE/n150/$LIT_TAIL"
LLMBOX="$LIT_BASE/llmbox/$LIT_TAIL"
GALAXY="$LIT_BASE/galaxy/$LIT_TAIL"

[[ -d "$MODELS_DIR" ]] || { echo "ERROR: no models dir at $MODELS_DIR" >&2; exit 1; }

# *galaxy* -> galaxy; *tp* -> llmbox; else n150.
route_for() {
    case "$1" in
        *galaxy*) echo "$GALAXY galaxy" ;;
        *tp*)     echo "$LLMBOX llmbox" ;;
        *)        echo "$N150 n150" ;;
    esac
}

# n150 lit config has no target gate; llmbox/galaxy do.
make_lit_config() {
    local dir="$1" kind="$2" cfg="$1/lit.local.cfg"
    [[ -f "$cfg" ]] && return
    mkdir -p "$dir"
    {
        echo "# Single-block / single-layer perf tests. Requires single-threaded"
        echo "# execution (physical device with opmodel)."
        echo 'config.parallelism_group = "opmodel"'
        if [[ "$kind" != n150 ]]; then
            echo "if \"$kind\" not in config.targets:"
            echo "    config.unsupported = True"
        fi
    } > "$cfg"
    echo "  Created: $cfg"
}

write_lit_test() {
    local name="$1" td="$2"
    cat > "$td/$name.mlir" <<EOF
// REQUIRES: opmodel, perf
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% optimization-level=2 experimental-weight-dtype=bfp_bf8 enable-permute-matmul-fusion=false" -o ${name}_ttnn.mlir %models/single_blocks_and_layers/${name}.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn ${name}_ttnn.mlir
EOF
}

# Single status+print helper used by both modes. do_create=1 writes missing
# lit files (UPDATE mode); do_create=0 just reports (STATUS mode).
new=0; have=0; missing=0
report() {
    local name="$1" do_create="$2" td kind status
    read -r td kind <<< "$(route_for "$name")"
    if [[ -f "$td/$name.mlir" ]]; then
        have=$((have+1)); status=HAVE
    elif (( do_create )); then
        make_lit_config "$td" "$kind"
        write_lit_test "$name" "$td"
        new=$((new+1)); status=NEW
    else
        missing=$((missing+1)); status=MISSING
    fi
    printf "  [%-7s] %-50s -> %s\n" "$status" "$name" "$kind"
}

if (( STATUS_ONLY )); then
    # STATUS MODE: walk all fixtures, report HAVE/MISSING.
    shopt -s nullglob
    for f in "$MODELS_DIR"/*.mlir; do report "$(basename "$f" .mlir)" 0; done
    shopt -u nullglob
    echo
    summary="Lit tests: HAVE=$have"
    (( missing )) && summary+=" MISSING=$missing (pass names as args to create)"
    echo "$summary"
    exit 0
fi

# UPDATE MODE: validate every name first; refuse to create any if one is bad.
bad=()
for arg in "$@"; do [[ -f "$MODELS_DIR/${arg%.mlir}.mlir" ]] || bad+=("${arg%.mlir}"); done
if (( ${#bad[@]} )); then
    echo "ERROR: no such fixture(s) under $MODELS_DIR:" >&2
    printf "  - %s\n" "${bad[@]}" >&2
    exit 1
fi
for arg in "$@"; do report "${arg%.mlir}" 1; done
echo
echo "Lit tests: NEW=$new HAVE=$have"
