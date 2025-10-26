# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .mapping import *

__all__ = [
    "GoldenMapTensor",
    "unpack_mlir_attr",
    "get_golden_function",
    "GOLDEN_MAPPINGS",
]
