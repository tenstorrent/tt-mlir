# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from collections import deque
from enum import Enum
from typing import (
    Annotated,
    Callable,
    Iterable,
    Iterator,
    List,
    Literal,
    Optional,
    Union,
)

from pydantic import BaseModel, ConfigDict, Field


class RecordStatus(str, Enum):
    OK = "ok"
    SHAPE_MISMATCH = "shape_mismatch"
    DTYPE_MISMATCH = "dtype_mismatch"
    NUMERICS_FAIL = "numerics_fail"
    ERROR = "error"
    SKIPPED_NUMERICS = "skipped_numerics"
    NO_GOLDEN = "no_golden"
    IR_RUNTIME_MISMATCH = "ir_runtime_mismatch"
    CHISEL_BUG = "chisel_bug"
    GOLDEN_PROMOTED = "golden_promoted"
    GOLDEN_EVICTED = "golden_evicted"


PASS_STATUS = frozenset(
    {
        RecordStatus.OK,
        RecordStatus.SKIPPED_NUMERICS,
        RecordStatus.NO_GOLDEN,
        RecordStatus.GOLDEN_PROMOTED,
        RecordStatus.GOLDEN_EVICTED,
    }
)


class NumericsMode(str, Enum):
    """How a numerics record's golden was produced.

    ISOLATED: per-op golden from device inputs.
    ACCUMULATED: program-scoped chain from prior goldens' outputs.
    """

    ISOLATED = "isolated"
    ACCUMULATED = "accumulated"


class _Payload(BaseModel):
    """Base for status-discriminated payloads.

    Each payload declares only the fields valid for its RecordStatus;
    extra="forbid" so wrong-field-for-status mistakes raise at construction,
    not at the downstream consumer.
    """

    model_config = ConfigDict(extra="forbid")


class NumericsPayload(_Payload):
    """Shared by OK and NUMERICS_FAIL; status set by check_numerics."""

    status: Literal[RecordStatus.OK, RecordStatus.NUMERICS_FAIL]
    mode: NumericsMode
    pcc: float
    atol: float
    rtol: float
    device_id: Optional[int] = None


class ShapeMismatchPayload(_Payload):
    status: Literal[RecordStatus.SHAPE_MISMATCH] = RecordStatus.SHAPE_MISMATCH
    expected_shape: List[int]
    actual_shape: List[int]


class DtypeMismatchPayload(_Payload):
    status: Literal[RecordStatus.DTYPE_MISMATCH] = RecordStatus.DTYPE_MISMATCH
    expected_dtype: str
    actual_dtype: str


class ErrorPayload(_Payload):
    status: Literal[RecordStatus.ERROR] = RecordStatus.ERROR


class SkippedNumericsPayload(_Payload):
    status: Literal[RecordStatus.SKIPPED_NUMERICS] = RecordStatus.SKIPPED_NUMERICS
    mode: NumericsMode


class NoGoldenPayload(_Payload):
    status: Literal[RecordStatus.NO_GOLDEN] = RecordStatus.NO_GOLDEN


class GoldenPromotedPayload(_Payload):
    """Audit-only: emitted when a device tensor is seeded into the golden pool.

    Ideally only function args are promoted; intermediate-SSA promotions
    mean a producer op is missing a golden.
    """

    status: Literal[RecordStatus.GOLDEN_PROMOTED] = RecordStatus.GOLDEN_PROMOTED


class GoldenEvictedPayload(_Payload):
    """Audit-only: one record per SSA removed from the golden pool."""

    status: Literal[RecordStatus.GOLDEN_EVICTED] = RecordStatus.GOLDEN_EVICTED


class IrRuntimeMismatchPayload(_Payload):
    status: Literal[RecordStatus.IR_RUNTIME_MISMATCH] = RecordStatus.IR_RUNTIME_MISMATCH
    runtime_debug: str


class ChiselBugPayload(_Payload):
    status: Literal[RecordStatus.CHISEL_BUG] = RecordStatus.CHISEL_BUG
    traceback: str


Payload = Annotated[
    Union[
        NumericsPayload,
        ShapeMismatchPayload,
        DtypeMismatchPayload,
        ErrorPayload,
        SkippedNumericsPayload,
        NoGoldenPayload,
        IrRuntimeMismatchPayload,
        ChiselBugPayload,
        GoldenPromotedPayload,
        GoldenEvictedPayload,
    ],
    Field(discriminator="status"),
]


class ChiselRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: str
    check: str
    op_asm: Optional[str] = None
    ssa: Optional[str] = None
    binary_id: Optional[int] = None
    program_name: Optional[str] = None
    program_index: Optional[int] = None
    payload: Payload

    @property
    def status(self) -> RecordStatus:
        return self.payload.status


class ChiselReport:
    def __init__(self, capacity: int = 10_000) -> None:
        self._records: deque[ChiselRecord] = deque(maxlen=capacity)

    def append(self, record: ChiselRecord) -> None:
        self._records.append(record)

    def clear(self) -> None:
        self._records.clear()

    def __iter__(self) -> Iterator[ChiselRecord]:
        return iter(self._records)

    def __len__(self) -> int:
        return len(self._records)

    @property
    def capacity(self) -> Optional[int]:
        return self._records.maxlen

    @property
    def records(self) -> List[ChiselRecord]:
        return list(self._records)

    def resize(self, capacity: int) -> None:
        self._records = deque(self._records, maxlen=capacity)

    def _from_filtered(
        self, predicate: Callable[[ChiselRecord], bool]
    ) -> "ChiselReport":
        # Preserve the larger of len(filtered) and source capacity so chained
        # filters never silently drop records into a maxlen=0 deque.
        filtered = [r for r in self._records if predicate(r)]
        out = ChiselReport(capacity=max(len(filtered), self.capacity or 1, 1))
        for r in filtered:
            out.append(r)
        return out

    def filter(self, predicate: Callable[[ChiselRecord], bool]) -> "ChiselReport":
        return self._from_filtered(predicate)

    def by_op(self, name: str) -> "ChiselReport":
        return self._from_filtered(lambda r: r.op == name)

    def by_check(self, name: str) -> "ChiselReport":
        return self._from_filtered(lambda r: r.check == name)

    def by_program(self, name: str) -> "ChiselReport":
        return self._from_filtered(lambda r: r.program_name == name)

    def by_status(
        self, status: Union[RecordStatus, str, Iterable[Union[RecordStatus, str]]]
    ) -> "ChiselReport":
        if isinstance(status, (RecordStatus, str)):
            wanted = {RecordStatus(status)}
        else:
            wanted = {RecordStatus(s) for s in status}
        return self._from_filtered(lambda r: r.status in wanted)

    def failures(self) -> "ChiselReport":
        return self._from_filtered(lambda r: r.status not in PASS_STATUS)

    def passes(self) -> "ChiselReport":
        return self._from_filtered(lambda r: r.status in PASS_STATUS)

    @classmethod
    def from_jsonl(cls, path: str, *, capacity: Optional[int] = None) -> "ChiselReport":
        # Default capacity to len(loaded) so offline loads never truncate;
        # the 10k runtime default is a buffer cap, not an analysis cap.
        records: List[ChiselRecord] = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(ChiselRecord.model_validate_json(line))
        cap = capacity if capacity is not None else max(len(records), 1)
        report = cls(capacity=cap)
        for r in records:
            report.append(r)
        return report
