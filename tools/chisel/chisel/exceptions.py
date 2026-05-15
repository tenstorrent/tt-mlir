# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Optional, Sequence

from .report import (
    ChiselRecord,
    DtypeMismatchPayload,
    ErrorPayload,
    IrRuntimeMismatchPayload,
    NoGoldenPayload,
    Payload,
    ShapeMismatchPayload,
    RecordStatus,
)
from .utils import get_op_asm


class UnexpectedStateError(RuntimeError):
    # Raised when a ChiselContext property is accessed outside its lifecycle
    # scope (e.g. ctx.op outside an op callback). Programming error, not a
    # check failure.

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(
            f"ChiselContext.{field} is unset (accessed outside its lifecycle scope)"
        )


class ChiselFailure(Exception):
    # Carries the offending op + a typed payload; converted to a ChiselRecord
    # via to_record(), either by the caller or by chisel_safe on raise.

    status: RecordStatus = RecordStatus.ERROR

    def __init__(
        self,
        op,
        check: str,
        detail: str = "",
        payload: Optional[Payload] = None,
    ) -> None:
        self.op = op
        self.op_name = op.name
        self.op_asm = get_op_asm(op)
        self.check = check
        self.payload: Payload = payload if payload is not None else ErrorPayload()
        header = f"{self.op_name} [{check}]: {self.status.value.upper()}"
        body = f"{header}\n  {detail}" if detail else header
        super().__init__(f"{body}\n  op: {self.op_asm}" if self.op_asm else body)

    def to_record(self) -> ChiselRecord:
        return ChiselRecord(
            op=self.op_name,
            check=self.check,
            op_asm=self.op_asm,
            payload=self.payload,
        )


class ShapeMismatch(ChiselFailure):
    status = RecordStatus.SHAPE_MISMATCH

    def __init__(
        self,
        op,
        check: str,
        expected_shape: Sequence[int],
        actual_shape: Sequence[int],
    ) -> None:
        exp = list(expected_shape)
        act = list(actual_shape)
        super().__init__(
            op,
            check,
            f"expected={exp} actual={act}",
            payload=ShapeMismatchPayload(expected_shape=exp, actual_shape=act),
        )


class DtypeMismatch(ChiselFailure):
    status = RecordStatus.DTYPE_MISMATCH

    def __init__(self, op, check: str, expected_dtype, actual_dtype) -> None:
        super().__init__(
            op,
            check,
            f"expected={expected_dtype} actual={actual_dtype}",
            payload=DtypeMismatchPayload(
                expected_dtype=str(expected_dtype),
                actual_dtype=str(actual_dtype),
            ),
        )


class GoldenNotImplementedError(ChiselFailure):
    status = RecordStatus.NO_GOLDEN

    def __init__(self, op) -> None:
        super().__init__(
            op,
            "golden_not_implemented",
            f"no golden registered for {op.name}",
            payload=NoGoldenPayload(),
        )


class IrRuntimeMismatch(ChiselFailure):
    status = RecordStatus.IR_RUNTIME_MISMATCH

    def __init__(self, op, check: str, runtime_debug: str) -> None:
        super().__init__(
            op,
            check,
            f"runtime debug_info:\n{runtime_debug}",
            payload=IrRuntimeMismatchPayload(runtime_debug=runtime_debug),
        )
