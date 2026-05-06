# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Chisel record schema (pydantic) and bounded in-memory report buffer.

Every record produced by chisel callbacks/checkers flows through
`ChiselContext.write_record`, which validates against `ChiselRecord` and
appends to a `ChiselReport` ring buffer. File output is a separate, opt-in
mirror — set `results_path=None` to disable it and rely solely on the report.
"""
from collections import deque
from typing import Iterable, Iterator, List, Literal, Optional

from pydantic import BaseModel, ConfigDict

Status = Literal[
    "ok",
    "shape_mismatch",
    "dtype_mismatch",
    "pcc_fail",
    "error",
    "skipped",
    "skipped_pcc",
    "applied",
    "chisel_bug",
]

# Statuses that represent a non-failure outcome. `failures()` excludes these.
_NON_FAILURE_STATUSES = frozenset({"ok", "applied", "skipped", "skipped_pcc"})


class ChiselRecord(BaseModel):
    """One chisel check result.

    Required fields are populated for every record. Optional fields appear
    only for the check/status combinations that produce them (e.g. `pcc`
    is set only by golden-vs-runtime tensor comparisons).
    """

    model_config = ConfigDict(extra="forbid")

    op: str
    slot: str
    check: str
    status: Status
    op_asm: str

    pcc: Optional[float] = None
    atol: Optional[float] = None
    rtol: Optional[float] = None

    expected_shape: Optional[List[int]] = None
    actual_shape: Optional[List[int]] = None

    expected_dtype: Optional[str] = None
    actual_dtype: Optional[str] = None

    traceback: Optional[str] = None


class ChiselReport:
    """Bounded in-memory store of `ChiselRecord` instances."""

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

    def by_op(self, op: str) -> List[ChiselRecord]:
        return [r for r in self._records if r.op == op]

    def by_status(self, *statuses: str) -> List[ChiselRecord]:
        wanted = set(statuses)
        return [r for r in self._records if r.status in wanted]

    def failures(self) -> List[ChiselRecord]:
        return [r for r in self._records if r.status not in _NON_FAILURE_STATUSES]
