# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

from ttmlir.dialects import func, ttcore, ttnn
from ttmlir.ir import OpView

from .op_handlers import _deallocate_pre_op, _noop_post_op

logger = logging.getLogger("chisel")


PreOrPostFn = Callable[["ChiselContext", "ChiselOpConfig"], bool]


@dataclass
class ChiselOpConfig:
    """Per-op chisel behavior overrides.

    no_golden: skip golden calculation; emit a NoGoldenPayload.
    skip_pcc: run golden, check shape/dtype, but skip PCC checks.
    pre_op / post_op: overrides for default handlers; must be @chisel_safe.
    """

    no_golden: bool = False
    skip_pcc: bool = False
    pre_op: Optional[PreOrPostFn] = None
    post_op: Optional[PreOrPostFn] = None


def default_configs() -> Dict[Type[OpView], ChiselOpConfig]:
    """Fresh dict of default per-op configs."""
    return {
        # ttnn.empty produces uninitialized memory - PCC comparison is meaningless.
        ttnn.EmptyOp: ChiselOpConfig(skip_pcc=True),
        # ttnn.generic: IR output count = 0 but FB output count = 1.
        ttnn.GenericOp: ChiselOpConfig(no_golden=True),
        # ttnn.tt_lang_op: opaque tt-lang kernel lowered to a GenericOp FB
        # record. It has IR results, but the runtime reports GenericOp as
        # having no output refs (same bucket as ttnn.generic), and it has
        # no golden to validate against.
        ttnn.TTLangOp: ChiselOpConfig(no_golden=True),
        # Non-executable ops (device handles, I/O, control flow): no golden to run.
        func.CallOp: ChiselOpConfig(no_golden=True),
        ttnn.GetDeviceOp: ChiselOpConfig(no_golden=True),
        ttnn.LoadTensorOp: ChiselOpConfig(no_golden=True),
        # Custom pre_op evicts inputs from the golden pool
        ttnn.DeallocateOp: ChiselOpConfig(
            pre_op=_deallocate_pre_op,
            post_op=_noop_post_op,
        ),
        ttcore.LoadCachedOp: ChiselOpConfig(no_golden=True),
        # Trace / control-flow / I/O ops.
        ttnn.AllocOp: ChiselOpConfig(no_golden=True),
        ttnn.BeginTraceCaptureOp: ChiselOpConfig(no_golden=True),
        ttnn.EndTraceCaptureOp: ChiselOpConfig(no_golden=True),
        ttnn.ExecuteTraceOp: ChiselOpConfig(no_golden=True),
        ttnn.CaptureOrExecuteTraceOp: ChiselOpConfig(no_golden=True),
        ttnn.CreateGlobalSemaphoreOp: ChiselOpConfig(no_golden=True),
        ttnn.ResetGlobalSemaphoreOp: ChiselOpConfig(no_golden=True),
        # In-place ops with no golden.
        ttnn.WriteTensorOp: ChiselOpConfig(no_golden=True),
        ttnn.PointToPointOp: ChiselOpConfig(no_golden=True),
        # Quantization ops not currently supported.
        ttnn.QuantizeOp: ChiselOpConfig(no_golden=True),
        ttnn.DequantizeOp: ChiselOpConfig(no_golden=True),
        ttnn.RequantizeOp: ChiselOpConfig(no_golden=True),
        # No golden registered yet.
        ttnn.NLPCreateQKVHeadsDecodeOp: ChiselOpConfig(no_golden=True),
        ttnn.RotaryEmbeddingLlamaOp: ChiselOpConfig(no_golden=True),
        ttnn.RotaryEmbeddingOp: ChiselOpConfig(no_golden=True),
        ttnn.ConstantOp: ChiselOpConfig(no_golden=True),
        ttnn.MeshPartitionOp: ChiselOpConfig(no_golden=True),
        ttnn.NLPConcatHeadsOp: ChiselOpConfig(no_golden=True),
        ttnn.NLPConcatHeadsDecodeOp: ChiselOpConfig(no_golden=True),
        ttnn.BitcastConvertOp: ChiselOpConfig(no_golden=True),
        ttnn.AsinhOp: ChiselOpConfig(no_golden=True),
        ttnn.DistributedLayerNormOp: ChiselOpConfig(no_golden=True),
        ttnn.RMSNormPreAllGatherOp: ChiselOpConfig(no_golden=True),
        ttnn.SamplingOp: ChiselOpConfig(no_golden=True),
        ttnn.SelectiveReduceCombineOp: ChiselOpConfig(no_golden=True),
        ttnn.AllToAllDispatchMetadataOp: ChiselOpConfig(no_golden=True),
        ttnn.D2MSubgraphOp: ChiselOpConfig(no_golden=True),
    }


def get_op_names_no_golden() -> frozenset[str]:
    return frozenset(
        op_type.OPERATION_NAME
        for op_type, config in default_configs().items()
        if config.no_golden
    )
