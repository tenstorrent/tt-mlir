# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-op configuration registry for chisel preOp/postOp dispatch.

Register a ChiselOpConfig to override default preOp/postOp behavior for specific
MLIR op types. Registrations at module level are applied on import.

Usage:
    register_op_config(MyOp, ChiselOpConfig(skip_pcc=True))
    register_op_config(MyOp, ChiselOpConfig(pre_op=my_pre, post_op=my_post))
"""
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

from ttmlir.dialects import func, ttcore, ttnn

logger = logging.getLogger("chisel")


@dataclass
class ChiselOpConfig:
    """Per-op behavior overrides for chisel preOp/postOp dispatch.

    Attributes:
        skip:           Entirely bypass preOp and postOp for this op type.
                        No records are written and the tensor pool is not updated.
                        Use for ops whose IR/FB output counts disagree or whose
                        tensors are not retrievable (e.g. cache identifiers).
        skip_pcc:       Run isolation golden but skip numerical PCC comparison.
                        Useful for ops whose device output is intentionally undefined
                        (e.g. ttnn.empty produces uninitialized memory).
        skip_accum_pcc: Same as skip_pcc but for the accumulation golden check.
        pre_op:   Callable replacing the default preOp body after op iterator
                  advance. Signature: (binary, program_context, op_context) -> None.
        post_op:  Callable replacing the entire default postOp body.
                  Signature: (binary, program_context, op_context) -> None.
                  When set, skip_pcc and skip_accum_pcc are ignored (custom fn
                  owns PCC logic).
    """

    skip: bool = False
    skip_pcc: bool = False
    skip_accum_pcc: bool = False
    pre_op: Optional[Callable] = None
    post_op: Optional[Callable] = None


_OP_CONFIGS: Dict[Type, ChiselOpConfig] = {}


def register_op_config(op_type: Type, config: ChiselOpConfig) -> None:
    """Register a ChiselOpConfig for a specific MLIR op class."""
    _OP_CONFIGS[op_type] = config


def get_op_config(op: object) -> ChiselOpConfig:
    """Return the ChiselOpConfig for op's type, or a default config if not registered."""
    return _OP_CONFIGS.get(type(op), ChiselOpConfig())


def get_skipped_op_names() -> frozenset:
    """MLIR op names (e.g. 'ttcore.load_cached') for all ops registered with skip=True."""
    return frozenset(
        op_type.OPERATION_NAME
        for op_type, config in _OP_CONFIGS.items()
        if config.skip
    )


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------

# ttnn.empty produces an uninitialized device tensor — PCC comparison is
# meaningless for both isolation and accumulation checks.
register_op_config(ttnn.EmptyOp, ChiselOpConfig(skip_pcc=True, skip_accum_pcc=True))

# ttcore.load_cached: IR output count > 0 but FB output count = 0.
# Inputs are cache identifiers, not retrievable tensors.
register_op_config(ttcore.LoadCachedOp, ChiselOpConfig(skip=True))

# ttnn.generic: IR output count = 0 but FB output count = 1.
register_op_config(ttnn.GenericOp, ChiselOpConfig(skip=True))

# func.call: not a device op — runtime does not fire callbacks for it.
register_op_config(func.CallOp, ChiselOpConfig(skip=True))

# ttnn.paged_update_cache / ttnn.fill_cache: in-place cache writes with no
# retrievable output tensors.
register_op_config(ttnn.PagedUpdateCacheOp, ChiselOpConfig(skip=True))
register_op_config(ttnn.FillCacheOp, ChiselOpConfig(skip=True))

# scale/zero_point operands are stored as FB attribute structs, not TensorRefs,
# so IR and FB input counts disagree.
register_op_config(ttnn.QuantizeOp, ChiselOpConfig(skip=True))
register_op_config(ttnn.DequantizeOp, ChiselOpConfig(skip=True))
register_op_config(ttnn.RequantizeOp, ChiselOpConfig(skip=True))
