# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Python-side enum wrappers for TTIR / TTCore attribute enums.

These enums mirror the C++-generated ``ttcore.ir`` enum values and provide a
convenient Python interface used by builder methods such as
:meth:`TTIRBuilder.reduce_scatter` and :meth:`TTIRBuilder.mesh_shard`.
"""

from enum import Enum
from ttmlir.dialects import ttir, ttcore, tensor, quant, func


class ReduceType(Enum):
    """Type of reduction operation for collective and reduction ops."""

    Sum = ttcore.ir.ReduceType.Sum
    Mean = ttcore.ir.ReduceType.Mean
    Max = ttcore.ir.ReduceType.Max
    Min = ttcore.ir.ReduceType.Min
    Std = ttcore.ir.ReduceType.Std
    Var = ttcore.ir.ReduceType.Var
    Prod = ttcore.ir.ReduceType.Prod
    Invalid = ttcore.ir.ReduceType.Invalid


class MeshShardType(Enum):
    """Strategy for distributing a tensor across the device mesh."""

    Identity = ttcore.ir.MeshShardType.Identity
    Replicate = ttcore.ir.MeshShardType.Replicate
    Maximal = ttcore.ir.MeshShardType.Maximal
    Devices = ttcore.ir.MeshShardType.Devices


class MeshShardDirection(Enum):
    """Direction of a mesh-shard operation (full→shard or shard→full)."""

    FullToShard = ttcore.ir.MeshShardDirection.FullToShard
    ShardToFull = ttcore.ir.MeshShardDirection.ShardToFull
