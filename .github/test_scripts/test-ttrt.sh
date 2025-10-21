#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

pytest -ssv "test/python/op_by_op" --junit-xml=$TEST_REPORT_PATH
# restore system desc as ttrt test clears it
ttrt query --save-artifacts
