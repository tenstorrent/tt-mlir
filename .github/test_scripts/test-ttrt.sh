#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

# mv ttrt-artifacts ttrt-artifacts_backup
pytest -ssv "test/python/op_by_op" --junit-xml=$TEST_REPORT_PATH
# rm -rf ttrt-artifacts
# mv ttrt-artifacts_backup ttrt-artifacts
# restore system desc as ttrt test clears it
ttrt query --save-artifacts
