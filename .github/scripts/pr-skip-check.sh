#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -ex

function main() {
    local draft=$1
    # See github action log for json shape
    local doc_only_changed=$(echo $changed_files_json | jq -r '.doc_only_changed' )

    if $draft == false; then
        echo "false"
        exit 0
    elif $doc_only_changed == false; then
        echo "false"
        exit 0
    fi
    echo "true"
}

printenv

main $1 
