# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChiselChecker: per-op validator that reports every mismatch precisely and never raises.
"""
import logging
import traceback
from typing import Optional

import torch

from golden import GoldenMapTensor
from golden.mapping import mlir_datatype_to_torch_dtype, mlir_type_to_torch_dtype
from golden.metrics import compute_atol, compute_pcc, compute_rtol

from .context import ChiselContext

logger = logging.getLogger("chisel")

_PCC_THRESHOLD = 0.99


# ---------------------------------------------------------------------------
# Shape/dtype extractors — global vs local
# ---------------------------------------------------------------------------

def _extract_global_shape_dtype(source) -> tuple[list, torch.dtype]:
    """Extract the *global* (full-tensor) shape and dtype.

    Accepted source types:
      - MLIR IR Value  (.type.shape is always global)
      - TensorRef      (.get_shape() returns global shape from TensorDesc.shape)
      - torch.Tensor   (shape is the tensor's own shape)

    GoldenMapTensor is not accepted here — the global shape across shards is
    not recoverable without sharding-dimension metadata.
    """
    if isinstance(source, torch.Tensor):
        return list(source.shape), source.dtype
    if hasattr(source, "type") and hasattr(source.type, "shape"):
        # MLIR IR Value
        return (
            list(source.type.shape),
            mlir_type_to_torch_dtype(source.type.element_type),
        )
    if hasattr(source, "get_shape") and hasattr(source, "get_dtype"):
        # TensorRef — get_shape() is the global shape
        return (
            list(source.get_shape()),
            mlir_datatype_to_torch_dtype(source.get_dtype()),
        )
    raise TypeError(
        f"cannot extract global shape/dtype from {type(source).__name__}"
    )


def _extract_local_shape_dtype(source) -> tuple[list, torch.dtype]:
    """Extract the *local* (per-shard) shape and dtype.

    Accepted source types:
      - GoldenMapTensor  (first-shard shape — all shards are uniform by invariant)
      - TensorRef        (.get_local_shape() returns TensorDesc.local_shape;
                          falls back to global shape for single-device tensors)
      - torch.Tensor     (no shard concept — shape returned as-is)
    """
    if isinstance(source, GoldenMapTensor):
        # .shape and .dtype forward to the first shard via __getattr__
        return list(source.shape), source.dtype
    if hasattr(source, "get_local_shape") and hasattr(source, "get_dtype"):
        # TensorRef with local_shape binding
        return (
            list(source.get_local_shape()),
            mlir_datatype_to_torch_dtype(source.get_dtype()),
        )
    if isinstance(source, torch.Tensor):
        return list(source.shape), source.dtype
    raise TypeError(
        f"cannot extract local shape/dtype from {type(source).__name__}"
    )


def _compare_shape_dtype(expected, actual) -> Optional[tuple[str, dict]]:
    """Pure comparator. Returns None on match, else (status, extras) for recording."""
    exp_shape, exp_dtype = expected
    act_shape, act_dtype = actual
    if exp_shape != act_shape:
        return "shape_mismatch", {
            "expected_shape": exp_shape,
            "actual_shape": act_shape,
        }
    if exp_dtype != act_dtype:
        return "dtype_mismatch", {
            "expected_dtype": str(exp_dtype),
            "actual_dtype": str(act_dtype),
        }
    return None


# ---------------------------------------------------------------------------
# ChiselChecker
# ---------------------------------------------------------------------------

class ChiselChecker:
    """
    Per-op validator instantiated once per callback pair (preOp/postOp).

    All public methods return bool and never raise — every exception is caught,
    logged, and written to the JSONL results file so nothing is silently dropped.

    The single exception: check_golden_vs_runtime_tensor raises AssertionError
    when ctx.strict=True, which preserves pytest-subtest integration.
    """

    def __init__(self, ctx: ChiselContext, op_name: str) -> None:
        self.ctx = ctx
        self.op_name = op_name
        self._op_asm = self._get_op_asm(ctx)

    @staticmethod
    def _get_op_asm(ctx: ChiselContext) -> str:
        try:
            op = ctx.current_program.current_op if ctx.current_program else None
            return op.get_asm(use_local_scope=True).strip() if op else ""
        except Exception:
            return ""

    def record(self, slot: str, check: str, status: str, **extra) -> None:
        """Append one JSON record via the context's shared writer."""
        self.ctx.write_record({
            "op": self.op_name,
            "slot": slot,
            "check": check,
            "status": status,
            **extra,
            "op_asm": self._op_asm,
        })

    def _check_shape_dtype_impl(
        self, slot: str, check: str, exp_tuple, act_tuple,
    ) -> bool:
        fail = _compare_shape_dtype(exp_tuple, act_tuple)
        if fail is None:
            self.record(slot, check, "ok")
            return True
        status, extras = fail
        detail = (
            f"expected={extras.get('expected_shape', extras.get('expected_dtype'))} "
            f"actual={extras.get('actual_shape', extras.get('actual_dtype'))}"
        )
        logger.warning(f"{self.op_name} {slot} [{check}]: {status.upper()} {detail}")
        self.record(slot, check, status, **extras)
        return False

    def check_global_shape_dtype(
        self, slot: str, check: str, expected, actual,
    ) -> bool:
        """Compare *global* shapes and dtypes.

        Both `expected` and `actual` must support global shape extraction:
        MLIR Value, TensorRef, or torch.Tensor.  Typical use: verifying that
        the MLIR IR shape matches the TensorRef recorded in the flatbuffer.
        """
        try:
            exp = _extract_global_shape_dtype(expected)
            act = _extract_global_shape_dtype(actual)
            return self._check_shape_dtype_impl(slot, check, exp, act)
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{self.op_name} {slot} [{check}]: ERROR\n{tb}")
            self.record(slot, check, "error", traceback=tb)
            return False

    def check_local_shape_dtype(
        self, slot: str, check: str, expected, actual,
    ) -> bool:
        """Compare *local* (per-shard) shapes and dtypes.

        Accepts TensorRef (uses get_local_shape()) and GoldenMapTensor (uses
        first-shard shape).  Typical use: verifying that the per-device shard
        shape recorded in the flatbuffer matches the shape of the retrieved
        runtime tensor or golden output.
        """
        try:
            exp = _extract_local_shape_dtype(expected)
            act = _extract_local_shape_dtype(actual)
            return self._check_shape_dtype_impl(slot, check, exp, act)
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{self.op_name} {slot} [{check}]: ERROR\n{tb}")
            self.record(slot, check, "error", traceback=tb)
            return False

    def _compare_shard(
        self,
        slot: str,
        check: str,
        log_tag: str,
        golden: torch.Tensor,
        device: torch.Tensor,
    ) -> bool:
        """Compare a single (golden shard, device shard) pair and record the result."""
        fail = _compare_shape_dtype(
            _extract_local_shape_dtype(golden),
            _extract_local_shape_dtype(device),
        )
        if fail is not None:
            status, extras = fail
            msg = (
                f"{self.op_name} {slot} [{check}]: {status.upper()} "
                f"expected={extras.get('expected_shape', extras.get('expected_dtype'))} "
                f"actual={extras.get('actual_shape', extras.get('actual_dtype'))}"
            )
            self.record(slot, check, status, **extras)
            logger.warning(msg)
            if self.ctx.strict:
                raise AssertionError(msg)
            return False

        pcc = compute_pcc(golden, device)
        atol = compute_atol(golden, device)
        rtol = compute_rtol(golden, device)

        if pcc >= _PCC_THRESHOLD:
            self.record(slot, check, "ok", pcc=pcc, atol=atol, rtol=rtol)
            logger.info(
                f"{self.op_name} {slot}{log_tag}: OK  "
                f"pcc={pcc:.6f} atol={atol:.6e} rtol={rtol:.6e}"
            )
            return True

        msg = (
            f"{self.op_name} {slot} [{check}]: PCC FAIL "
            f"pcc={pcc:.6f} (threshold={_PCC_THRESHOLD}) "
            f"atol={atol:.6e} rtol={rtol:.6e}"
        )
        self.record(slot, check, "pcc_fail", pcc=pcc, atol=atol, rtol=rtol)
        logger.warning(msg)
        if self.ctx.strict:
            raise AssertionError(msg)
        return False

    def check_golden_vs_runtime_tensor(
        self,
        slot: str,
        golden: GoldenMapTensor,
        device: GoldenMapTensor,
        *,
        accum: bool = False,
    ) -> bool:
        """Per-shard PCC/atol/rtol comparison between golden and device tensors.

        For single-device tensors one comparison is recorded under `slot`.
        For multi-device tensors each shard is recorded under `slot[shard_id]`
        so failures are traceable to individual devices.

        When accum=True records under "accum_golden_vs_runtime_tensor".
        Raises AssertionError if ctx.strict=True and any shard fails.
        """
        check = "accum_golden_vs_runtime_tensor" if accum else "golden_vs_runtime_tensor"
        log_tag = " [accum]" if accum else ""
        try:
            golden_shards = golden.golden_map_tensor_as_torch_tensors()
            device_shards = device.golden_map_tensor_as_torch_tensors()
            num_shards = len(golden_shards)
            all_ok = True
            for shard_id in sorted(golden_shards.keys()):
                shard_slot = f"{slot}[{shard_id}]" if num_shards > 1 else slot
                d_shard = device_shards.get(shard_id)
                if d_shard is None:
                    msg = f"missing device shard {shard_id}"
                    self.record(shard_slot, check, "error", traceback=msg)
                    logger.error(f"{self.op_name} {shard_slot} [{check}]: {msg}")
                    all_ok = False
                    continue
                ok = self._compare_shard(
                    shard_slot, check, log_tag,
                    golden_shards[shard_id], d_shard,
                )
                all_ok = all_ok and ok
            return all_ok

        except AssertionError:
            raise
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{self.op_name} {slot} [{check}]: ERROR\n{tb}")
            self.record(slot, check, "error", traceback=tb)
            return False
