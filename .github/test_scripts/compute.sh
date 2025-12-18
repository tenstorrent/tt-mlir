#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

echo "Running COMPUTE tests"
for number in {1..5}; do
    echo "Test $number"
    sleep 1
done
