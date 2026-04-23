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
import traceback
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

from ttmlir.dialects import ttcore, ttnn

from .checker import ChiselChecker
from .context import ChiselContext
from .ops import get_op_outputs
from .utils import retrieve_torch_tensor

logger = logging.getLogger("chisel")


@dataclass
class ChiselOpConfig:
    """Per-op behavior overrides for chisel preOp/postOp dispatch.

    Attributes:
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


# ---------------------------------------------------------------------------
# Per-op implementations
# ---------------------------------------------------------------------------


def _load_cached_pre_op(binary, program_context, op_context) -> None:
    """PreOp for ttcore.LoadCachedOp.

    The cached tensor is already resident on device — its inputs are cache
    identifiers, not addressable tensors. Skip input retrieval and pool seeding
    entirely; provide an empty stash so postOp sees consistent state.
    """
    ctx = ChiselContext.get_instance()
    ctx._stashed_inputs = {}


def _load_cached_post_op(binary, program_context, op_context) -> None:
    """PostOp for ttcore.LoadCachedOp.

    Validates output shape/dtype against MLIR IR and the flatbuffer TensorRef.
    Golden comparison is skipped: the cached value is opaque to chisel.
    The output is NOT added to the golden pool — downstream ops that need it
    will fall back to device-value seeding in preop.
    """
    from ttrt import runtime as tt_runtime

    ctx = ChiselContext.get_instance()
    program = ctx.current_program
    binary_state = ctx.current_binary
    op = program.current_op
    op_name = op.name
    checker = ChiselChecker(ctx, op_name)

    op_outputs = get_op_outputs(op)
    if not op_outputs:
        ctx._stashed_inputs = None
        return

    output_refs = tt_runtime.get_op_output_refs(op_context, program_context)
    asm_state = binary_state.ir_module.get_asm_state(program.program_name)

    for mlir_output, output_ref in zip(op_outputs, output_refs, strict=True):
        name = mlir_output.get_name(asm_state)
        checker.check_mlir_vs_tensor_ref(name, mlir_output, output_ref)
        try:
            device_tensor = retrieve_torch_tensor(program_context, output_ref)
        except Exception:
            tb = traceback.format_exc()
            logger.error(
                f"{op_name} {name}: failed to retrieve device output tensor\n{tb}"
            )
            checker._record(name, "retrieve_output", "error", traceback=tb)
            continue
        checker.check_mlir_vs_runtime_tensor(name, mlir_output, device_tensor)
        checker._record(name, "golden", "skipped")
        checker._record(name, "accum_golden", "skipped")

    ctx._stashed_inputs = None


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------

# ttnn.empty produces an uninitialized device tensor — PCC comparison is
# meaningless for both isolation and accumulation checks.
register_op_config(ttnn.EmptyOp, ChiselOpConfig(skip_pcc=True, skip_accum_pcc=True))

# ttcore.load_cached loads a tensor from the device cache. Inputs are cache
# identifiers, not retrievable tensors, and the cached value is opaque.
register_op_config(
    ttcore.LoadCachedOp,
    ChiselOpConfig(pre_op=_load_cached_pre_op, post_op=_load_cached_post_op),
)
