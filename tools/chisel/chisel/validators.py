# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass, field
from typing import Optional

import torch
from _ttmlir_runtime.runtime import TensorRef
from ttmlir.ir import Value

from golden import GoldenMapTensor
from golden.mapping import mlir_datatype_to_torch_dtype, mlir_type_to_torch_dtype
from golden.metrics import get_atol_rtol_pcc

from .exceptions import ChiselFailure, DtypeMismatch, ShapeMismatch
from .report import (
    ChiselRecord,
    NumericsMode,
    NumericsPayload,
    RecordStatus,
    SkippedNumericsPayload,
)
from .utils import cached_retrieve_tensor

logger = logging.getLogger("chisel")


@dataclass
class PCCConfig:
    # Settings consumed by the PCC computation itself.

    # Fail if computed PCC < min_pcc.
    min_pcc: float = 0.99

    # atol/rtol are NOT failure thresholds - they are forwarded to get_pcc ->
    # torch.isclose to short-circuit two degenerate cases:
    #   * single-element tensors (one scalar value to compare)
    #   * constant tensors (all elements identical in golden and device)
    # In both cases torch.isclose with these tolerances yields the PCC verdict
    # directly, because correlation is undefined / 1.0 by convention.
    atol: float = 1e-8
    rtol: float = 1e-5


@dataclass
class ChiselChecksConfig:
    # Settings for chisel's checks; override via
    # chisel.configure(checks_config=...).

    pcc: PCCConfig = field(default_factory=PCCConfig)

    # When set, the numerics check also fails if the computed worst-case
    # absolute/relative diff exceeds the threshold. None = disabled.
    max_atol: Optional[float] = None
    max_rtol: Optional[float] = None

    # Per-op golden modes:
    #   isolation:   re-run the golden each op from device inputs.
    #   accumulation: chain goldens through a program-scoped pool.
    isolation: bool = False
    accumulation: bool = True


def _extract_shape_dtype(source) -> tuple[list, torch.dtype]:
    # Accepts a GoldenMapTensor, an MLIR IR Value, a flatbuffer TensorRef, or
    # a torch.Tensor.
    if isinstance(source, GoldenMapTensor):
        t = next(iter(source.shard_map.values()))
        return list(t.shape), t.dtype
    if isinstance(source, torch.Tensor):
        return list(source.shape), source.dtype
    if hasattr(source, "type") and hasattr(source.type, "shape"):
        return (
            list(source.type.shape),
            mlir_type_to_torch_dtype(source.type.element_type),
        )
    if hasattr(source, "get_shape") and hasattr(source, "get_dtype"):
        return (
            list(source.get_shape()),
            mlir_datatype_to_torch_dtype(source.get_dtype()),
        )
    raise TypeError(f"cannot extract shape/dtype from {type(source).__name__}")


def emit_pcc(
    ctx,
    op,
    ssa: str,
    mlir_value,
    golden_out: GoldenMapTensor,
    device_tensor: GoldenMapTensor,
    *,
    mode: NumericsMode,
    skip_pcc: bool,
) -> None:
    """Shape/dtype + PCC for one (golden, device) pair under `mode`."""
    check_shape_dtype(op, "mlir_vs_golden", mlir_value, golden_out)
    if skip_pcc:
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check="numerics",
                ssa=ssa,
                payload=SkippedNumericsPayload(mode=mode),
            )
        )
        return
    check_numerics(ctx, op, ssa, golden_out, device_tensor, mode=mode)


def check_shape_dtype(op, check: str, expected, actual) -> None:
    exp_shape, exp_dtype = _extract_shape_dtype(expected)
    act_shape, act_dtype = _extract_shape_dtype(actual)
    if exp_shape != act_shape:
        raise ShapeMismatch(op, check, exp_shape, act_shape)
    if exp_dtype != act_dtype:
        raise DtypeMismatch(op, check, exp_dtype, act_dtype)


def validate_and_retrieve_tensor(
    ctx, mlir_value: Value, rt_tensor_ref: TensorRef
) -> GoldenMapTensor:
    """Shape/dtype-check `rt_tensor_ref` against the IR, pull a host copy
    (cached by SSA), and re-check it. Shared by the default callbacks and the
    per-op handlers (op_handlers.py)."""
    op = ctx.op
    check_shape_dtype(op, "mlir_vs_tensor_ref", mlir_value, rt_tensor_ref)
    ssa = mlir_value.get_name(ctx.asm_state)
    tensor = cached_retrieve_tensor(ctx, ssa, rt_tensor_ref, ctx.mesh_shape)
    check_shape_dtype(op, "mlir_vs_runtime_tensor", mlir_value, tensor)
    return tensor


def check_numerics(
    ctx,
    op,
    ssa: str,
    golden: GoldenMapTensor,
    device: GoldenMapTensor,
    mode: NumericsMode = NumericsMode.ISOLATED,
) -> None:
    """Per-shard PCC check; emits one record per shard, tagged by `mode`."""
    check = "numerics"
    check_shape_dtype(op, check, golden, device)

    cfg = ctx.checks_config
    pcc_cfg = cfg.pcc
    golden_shards = golden.shard_map
    device_shards = device.shard_map

    if golden_shards.keys() != device_shards.keys():
        raise ChiselFailure(
            op,
            check,
            f"shard id mismatch: golden={sorted(golden_shards)} "
            f"device={sorted(device_shards)}",
        )

    for device_id, golden_shard in golden_shards.items():
        device_shard = device_shards[device_id]
        atol, rtol, pcc = get_atol_rtol_pcc(
            golden_shard, device_shard, pcc_cfg.atol, pcc_cfg.rtol
        )
        failed = pcc < pcc_cfg.min_pcc
        if cfg.max_atol is not None and atol > cfg.max_atol:
            failed = True
        if cfg.max_rtol is not None and rtol > cfg.max_rtol:
            failed = True
        status = RecordStatus.NUMERICS_FAIL if failed else RecordStatus.OK
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check=check,
                ssa=ssa,
                payload=NumericsPayload(
                    status=status,
                    mode=mode,
                    pcc=pcc,
                    atol=atol,
                    rtol=rtol,
                    device_id=device_id,
                ),
            )
        )
