# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from enum import Enum


class ExecutionType(Enum):
    DEVICE = "device"
    GOLDEN = "golden"
