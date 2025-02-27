# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum


class CompileStep(Enum):
    STABLE_HLO = 1
    TTIR = 2
    TTNN = 3
    FLATBUFFER = 4
