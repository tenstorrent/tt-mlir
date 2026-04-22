# tools/chisel/chisel/checker.py
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
ChiselChecker: per-op validator that reports every mismatch precisely and never raises.
"""
import json
import logging
import traceback

import torch

from golden.mapping import mlir_datatype_to_torch_dtype, mlir_type_to_torch_dtype
from golden.metrics import compute_atol, compute_pcc, compute_rtol

from .context import ChiselContext

logger = logging.getLogger("chisel")

_PCC_THRESHOLD = 0.99


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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_op_asm(ctx: ChiselContext) -> str:
        try:
            return ctx._current_op.get_asm(use_local_scope=True).strip()
        except Exception:
            return ""

    def _record(self, slot: str, check: str, status: str, **extra) -> None:
        """Append one JSON record to ctx.results_path (no-op if path is None)."""
        if self.ctx.results_path is None:
            return
        record = {
            "op": self.op_name,
            "slot": slot,
            "check": check,
            "status": status,
            **extra,
            "op_asm": self._op_asm,
        }
        with open(self.ctx.results_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _check_shape_dtype(
        self,
        slot: str,
        check: str,
        expected_shape: list,
        actual_shape: list,
        expected_dtype: torch.dtype,
        actual_dtype: torch.dtype,
    ) -> bool:
        ok = True
        if expected_shape != actual_shape:
            msg = (
                f"{self.op_name} {slot} [{check}]: shape MISMATCH "
                f"expected={expected_shape} actual={actual_shape}"
            )
            self._record(
                slot, check, "shape_mismatch",
                expected_shape=expected_shape, actual_shape=actual_shape,
            )
            logger.warning(msg)
            ok = False
        if expected_dtype != actual_dtype:
            msg = (
                f"{self.op_name} {slot} [{check}]: dtype MISMATCH "
                f"expected={expected_dtype} actual={actual_dtype}"
            )
            self._record(
                slot, check, "dtype_mismatch",
                expected_dtype=str(expected_dtype), actual_dtype=str(actual_dtype),
            )
            logger.warning(msg)
            ok = False
        if ok:
            self._record(slot, check, "ok")
        return ok

    # ------------------------------------------------------------------
    # 5 public check methods
    # ------------------------------------------------------------------

    def check_mlir_vs_tensor_ref(self, slot: str, mlir_value, tensor_ref) -> bool:
        """Shape/dtype: MLIR IR type vs TensorRef metadata from flatbuffer."""
        try:
            mlir_shape = list(mlir_value.type.shape)
            mlir_dtype = mlir_type_to_torch_dtype(mlir_value.type.element_type)
            ref_shape = list(tensor_ref.get_shape())
            ref_dtype = mlir_datatype_to_torch_dtype(tensor_ref.get_dtype())
            return self._check_shape_dtype(
                slot, "mlir_vs_tensor_ref",
                mlir_shape, ref_shape,
                mlir_dtype, ref_dtype,
            )
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{self.op_name} {slot} [mlir_vs_tensor_ref]: ERROR\n{tb}")
            self._record(slot, "mlir_vs_tensor_ref", "error", traceback=tb)
            return False

    def check_mlir_vs_runtime_tensor(
        self, slot: str, mlir_value, tensor: torch.Tensor
    ) -> bool:
        """Shape/dtype: MLIR IR type vs retrieved torch tensor."""
        try:
            mlir_shape = list(mlir_value.type.shape)
            mlir_dtype = mlir_type_to_torch_dtype(mlir_value.type.element_type)
            return self._check_shape_dtype(
                slot, "mlir_vs_runtime_tensor",
                mlir_shape, list(tensor.shape),
                mlir_dtype, tensor.dtype,
            )
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{self.op_name} {slot} [mlir_vs_runtime_tensor]: ERROR\n{tb}")
            self._record(slot, "mlir_vs_runtime_tensor", "error", traceback=tb)
            return False

    def check_golden_vs_tensor_ref(
        self, slot: str, golden: torch.Tensor, tensor_ref
    ) -> bool:
        """Shape/dtype: golden output vs TensorRef metadata."""
        try:
            ref_shape = list(tensor_ref.get_shape())
            ref_dtype = mlir_datatype_to_torch_dtype(tensor_ref.get_dtype())
            return self._check_shape_dtype(
                slot, "golden_vs_tensor_ref",
                ref_shape, list(golden.shape),
                ref_dtype, golden.dtype,
            )
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{self.op_name} {slot} [golden_vs_tensor_ref]: ERROR\n{tb}")
            self._record(slot, "golden_vs_tensor_ref", "error", traceback=tb)
            return False

    def check_mlir_vs_golden(
        self, slot: str, mlir_value, golden: torch.Tensor
    ) -> bool:
        """Shape/dtype: MLIR IR type vs golden tensor."""
        try:
            mlir_shape = list(mlir_value.type.shape)
            mlir_dtype = mlir_type_to_torch_dtype(mlir_value.type.element_type)
            return self._check_shape_dtype(
                slot, "mlir_vs_golden",
                mlir_shape, list(golden.shape),
                mlir_dtype, golden.dtype,
            )
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{self.op_name} {slot} [mlir_vs_golden]: ERROR\n{tb}")
            self._record(slot, "mlir_vs_golden", "error", traceback=tb)
            return False

    def check_golden_vs_runtime_tensor(
        self, slot: str, golden: torch.Tensor, device: torch.Tensor
    ) -> bool:
        """Full comparison: shape, dtype, PCC, atol, rtol.

        Raises AssertionError if ctx.strict=True and the check fails,
        so pytest-subtest integration continues to work.
        """
        try:
            if list(golden.shape) != list(device.shape):
                msg = (
                    f"{self.op_name} {slot} [golden_vs_runtime_tensor]: shape MISMATCH "
                    f"expected={list(golden.shape)} actual={list(device.shape)}"
                )
                self._record(
                    slot, "golden_vs_runtime_tensor", "shape_mismatch",
                    expected_shape=list(golden.shape), actual_shape=list(device.shape),
                )
                logger.warning(msg)
                if self.ctx.strict:
                    raise AssertionError(msg)
                return False

            if golden.dtype != device.dtype:
                msg = (
                    f"{self.op_name} {slot} [golden_vs_runtime_tensor]: dtype MISMATCH "
                    f"expected={golden.dtype} actual={device.dtype}"
                )
                self._record(
                    slot, "golden_vs_runtime_tensor", "dtype_mismatch",
                    expected_dtype=str(golden.dtype), actual_dtype=str(device.dtype),
                )
                logger.warning(msg)
                if self.ctx.strict:
                    raise AssertionError(msg)
                return False

            pcc = compute_pcc(golden, device)
            atol = compute_atol(golden, device)
            rtol = compute_rtol(golden, device)

            if pcc >= _PCC_THRESHOLD:
                self._record(
                    slot, "golden_vs_runtime_tensor", "ok",
                    pcc=pcc, atol=atol, rtol=rtol,
                )
                logger.info(
                    f"{self.op_name} {slot}: OK  "
                    f"pcc={pcc:.6f} atol={atol:.6e} rtol={rtol:.6e}"
                )
                return True

            msg = (
                f"{self.op_name} {slot} [golden_vs_runtime_tensor]: PCC FAIL "
                f"pcc={pcc:.6f} (threshold={_PCC_THRESHOLD}) "
                f"atol={atol:.6e} rtol={rtol:.6e}"
            )
            self._record(
                slot, "golden_vs_runtime_tensor", "pcc_fail",
                pcc=pcc, atol=atol, rtol=rtol,
            )
            logger.warning(msg)
            if self.ctx.strict:
                raise AssertionError(msg)
            return False

        except AssertionError:
            raise
        except Exception:
            tb = traceback.format_exc()
            logger.error(
                f"{self.op_name} {slot} [golden_vs_runtime_tensor]: ERROR\n{tb}"
            )
            self._record(slot, "golden_vs_runtime_tensor", "error", traceback=tb)
            return False
