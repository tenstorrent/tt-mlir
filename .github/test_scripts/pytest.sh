#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


if [ -n "$3" ]; then
    pip install $3
fi
pytest -ssv $1 $2 --junit-xml=$TEST_REPORT_PATH
