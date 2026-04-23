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

from golden.mapping import mlir_datatype_to_torch_dtype, mlir_type_to_torch_dtype
from golden.metrics import compute_atol, compute_pcc, compute_rtol

from .context import ChiselContext

logger = logging.getLogger("chisel")

_PCC_THRESHOLD = 0.99


def _extract_shape_dtype(source) -> tuple[list, torch.dtype]:
    """Type-dispatched (shape, torch dtype) extractor.

    Accepts an MLIR IR Value, a flatbuffer TensorRef, or a torch.Tensor.
    """
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

    def check_shape_dtype(self, slot: str, check: str, expected, actual) -> bool:
        """Shape/dtype comparison between any two of: MLIR Value, TensorRef, torch.Tensor."""
        try:
            exp = _extract_shape_dtype(expected)
            act = _extract_shape_dtype(actual)
            fail = _compare_shape_dtype(exp, act)
            if fail is None:
                self.record(slot, check, "ok")
                return True
            status, extras = fail
            if status == "shape_mismatch":
                detail = f"expected={exp[0]} actual={act[0]}"
            else:
                detail = f"expected={exp[1]} actual={act[1]}"
            logger.warning(
                f"{self.op_name} {slot} [{check}]: {status.upper()} {detail}"
            )
            self.record(slot, check, status, **extras)
            return False
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{self.op_name} {slot} [{check}]: ERROR\n{tb}")
            self.record(slot, check, "error", traceback=tb)
            return False

    def check_golden_vs_runtime_tensor(
        self, slot: str, golden: torch.Tensor, device: torch.Tensor,
        *, accum: bool = False,
    ) -> bool:
        """Full comparison: shape, dtype, PCC, atol, rtol.

        When accum=True, records under check="accum_golden_vs_runtime_tensor"
        so isolation and accumulation results are distinguishable in the JSONL
        output.

        Raises AssertionError if ctx.strict=True and the check fails,
        so pytest-subtest integration continues to work.
        """
        check = "accum_golden_vs_runtime_tensor" if accum else "golden_vs_runtime_tensor"
        log_tag = " [accum]" if accum else ""
        try:
            fail = _compare_shape_dtype(
                _extract_shape_dtype(golden), _extract_shape_dtype(device)
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

        except AssertionError:
            raise
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"{self.op_name} {slot} [{check}]: ERROR\n{tb}")
            self.record(slot, check, "error", traceback=tb)
            return False
