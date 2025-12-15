# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from ttmlir.dialects import ttir, ttcore, tensor, quant, func


class ReduceType(Enum):
    Sum = ttcore.ir.ReduceType.Sum
    Mean = ttcore.ir.ReduceType.Mean
    Max = ttcore.ir.ReduceType.Max
    Min = ttcore.ir.ReduceType.Min
    Std = ttcore.ir.ReduceType.Std
    Var = ttcore.ir.ReduceType.Var
    Prod = ttcore.ir.ReduceType.Prod
    Invalid = ttcore.ir.ReduceType.Invalid
