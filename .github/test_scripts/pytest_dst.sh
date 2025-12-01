#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Run TTMetal golden pytests with all DST allocation strategies (basic, greedy, chaitin-briggs)
# Note: allocation-strategy is only available in ttir-to-ttmetal-pipeline, not TTNN

set -e -o pipefail

STRATEGIES=("basic" "greedy" "chaitin-briggs")

for strategy in "${STRATEGIES[@]}"; do
    echo "Running TTMetal pytests with DST_ALLOCATION_STRATEGY=$strategy"
    report_path="${TEST_REPORT_PATH%.xml}_${strategy}.xml"
    DST_ALLOCATION_STRATEGY=$strategy pytest -v -k "ttmetal" "$@" --junit-xml="$report_path"
    echo; echo "--- ⊤⊥ --------- ⊤⊥ --------- end of dst-strategy=$strategy test --------- ⊤⊥ --------- ⊤⊥ ---"; echo;
done
