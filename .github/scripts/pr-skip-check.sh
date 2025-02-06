#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -ex

function main() {
    local draft=$1
    local doc_only_changed=$2

    if $draft == true; then
        echo "false"
        exit 0
    elif $doc_only_changed == true; then
        echo "false"
        exit 0
    fi
    echo "true"
}

main $1 $2
