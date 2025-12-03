#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e -o pipefail

if [ -n "$REQUIREMENTS" ]; then
    eval "pip install $REQUIREMENTS"
fi

pytest -v --junit-xml="$TEST_REPORT_PATH" "$@"
