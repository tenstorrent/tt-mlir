# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass
from typing import Dict, Type

from ttmlir.dialects import func, ttcore, ttnn

logger = logging.getLogger("chisel")


@dataclass
class ChiselOpConfig:
    # no_golden: do not run the isolation golden; emit a NoGoldenPayload
    # record per op so the report shows the op was visited but skipped.
    # skip_isolated_pcc: run the isolation golden (shape/dtype still checked)
    # but skip the PCC comparison.
    no_golden: bool = False
    skip_isolated_pcc: bool = False


def default_configs() -> Dict[Type, ChiselOpConfig]:
    # Returns a fresh dict each call so callers cannot mutate the defaults.
    configs: Dict[Type, ChiselOpConfig] = {}

    # ttnn.empty produces uninitialized memory — PCC comparison is meaningless.
    configs[ttnn.EmptyOp] = ChiselOpConfig(skip_isolated_pcc=True)

    # ttnn.generic: IR output count = 0 but FB output count = 1.
    configs[ttnn.GenericOp] = ChiselOpConfig(no_golden=True)

    # Non-executable ops (device handles, I/O, control flow): no golden to run.
    configs[func.CallOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.GetDeviceOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.LoadTensorOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.DeallocateOp] = ChiselOpConfig(no_golden=True)
    configs[ttcore.LoadCachedOp] = ChiselOpConfig(no_golden=True)

    # Trace / control-flow / I/O ops.
    configs[ttnn.AllocOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.BeginTraceCaptureOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.EndTraceCaptureOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.ExecuteTraceOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.CaptureOrExecuteTraceOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.CreateGlobalSemaphoreOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.ResetGlobalSemaphoreOp] = ChiselOpConfig(no_golden=True)

    # In-place ops (see CHISEL_INPLACE_OPS). TODO: chisel does not yet
    # validate in-place tensor mutations; skip until support lands.
    configs[ttnn.UpdateCacheOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.PagedUpdateCacheOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.FillCacheOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.PagedFillCacheOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.WriteTensorOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.BatchNormTrainingOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.PointToPointOp] = ChiselOpConfig(no_golden=True)

    # Quantization ops not currently supported.
    configs[ttnn.QuantizeOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.DequantizeOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.RequantizeOp] = ChiselOpConfig(no_golden=True)

    # No golden registered yet
    configs[ttnn.NLPCreateQKVHeadsDecodeOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.RotaryEmbeddingLlamaOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.RotaryEmbeddingOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.ConstantOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.MeshShardOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.MeshPartitionOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.NLPConcatHeadsOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.NLPConcatHeadsDecodeOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.BitcastConvertOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.AsinhOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.DistributedLayerNormOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.RMSNormPreAllGatherOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.SamplingOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.SelectiveReduceCombineOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.AllToAllDispatchMetadataOp] = ChiselOpConfig(no_golden=True)
    configs[ttnn.D2MSubgraphOp] = ChiselOpConfig(no_golden=True)

    return configs


def get_no_golden_op_names() -> frozenset[str]:
    return frozenset(
        op_type.OPERATION_NAME
        for op_type, config in default_configs().items()
        if config.no_golden
    )
