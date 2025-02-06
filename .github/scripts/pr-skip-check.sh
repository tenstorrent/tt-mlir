#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -ex

function main() {
    local is_draft=$1
    local is_docs=$2

    if "$is_draft" == "true"; then
        echo "true"
        exit 0
    elif "$is_docs" == "true"; then
        echo "true"
        exit 0
    fi
    echo "false"
}

main $1 $2