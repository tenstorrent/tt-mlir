# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging
from dataclasses import dataclass
from typing import Optional

import torch

from golden import GoldenMapTensor
from golden.mapping import mlir_datatype_to_torch_dtype, mlir_type_to_torch_dtype
from golden.metrics import get_atol_rtol_pcc

from .exceptions import DtypeMismatch, ShapeMismatch
from .report import ChiselRecord, NumericsPayload, Status

logger = logging.getLogger("chisel")


@dataclass
class ChiselChecksConfig:
    # Settings for chisel's checks; override via
    # chisel.configure(checks_config=...)

    # Fail if computed PCC < threshold.
    threshold: float = 0.99

    # atol/rtol below are NOT failure thresholds — they are forwarded to
    # get_pcc → torch.isclose to short-circuit two edge cases:
    #   * single-element tensors (one scalar value to compare)
    #   * constant tensors (all elements identical in golden and device)
    # In both cases torch.isclose with these tolerances yields the PCC verdict
    # directly, because correlation is undefined / 1.0 by convention. They do
    # Use max_atol/max_rtol below to gate failure on the worst-case diffs.
    atol: float = 1e-8
    rtol: float = 1e-5

    # When set, the numerics check also fails if the computed worst-case
    # absolute/relative diff exceeds the threshold. None = disabled.
    max_atol: Optional[float] = None
    max_rtol: Optional[float] = None


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


def check_shape_dtype(op, check: str, expected, actual) -> None:
    exp_shape, exp_dtype = _extract_shape_dtype(expected)
    act_shape, act_dtype = _extract_shape_dtype(actual)
    if exp_shape != act_shape:
        raise ShapeMismatch(op, check, exp_shape, act_shape)
    if exp_dtype != act_dtype:
        raise DtypeMismatch(op, check, exp_dtype, act_dtype)


def check_numerics(
    ctx, op, ssa: str, golden: GoldenMapTensor, device: GoldenMapTensor
) -> None:
    """Per-shard PCC check.

    Writes one record per shard; multi-shard tensors are compared shard-by-shard
    rather than collapsed to a single comparison.
    """
    check = "numerics"
    check_shape_dtype(op, check, golden, device)

    cfg = ctx.checks_config
    golden_shards = golden.shard_map
    device_shards = device.shard_map

    for device_id, golden_shard in golden_shards.items():
        if device_id not in device_shards:
            continue
        device_shard = device_shards[device_id]
        atol, rtol, pcc = get_atol_rtol_pcc(
            golden_shard, device_shard, cfg.atol, cfg.rtol
        )
        failed = pcc < cfg.threshold
        if cfg.max_atol is not None and atol > cfg.max_atol:
            failed = True
        if cfg.max_rtol is not None and rtol > cfg.max_rtol:
            failed = True
        status = Status.NUMERICS_FAIL if failed else Status.OK
        ctx.write_record(
            ChiselRecord(
                op=op.name,
                check=check,
                ssa=ssa,
                payload=NumericsPayload(
                    status=status, pcc=pcc, atol=atol, rtol=rtol, device_id=device_id
                ),
            )
        )
