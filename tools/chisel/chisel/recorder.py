# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import logging
import os
from typing import Tuple, Optional, Set, TextIO

from _ttmlir_runtime.binary import Binary

from .report import ChiselRecord, ChiselReport, PASS_STATUS

logger = logging.getLogger("chisel")

_UNSET = object()


class JsonlSink:
    # Per-record append+flush so partial results survive a hang/crash mid-program.

    def __init__(self, path: str) -> None:
        self.path = path
        self._file: Optional[TextIO] = None

    def write(self, record: ChiselRecord) -> None:
        if self._file is None:
            self._file = open(self.path, "a")
        self._file.write(record.model_dump_json(exclude_none=True) + "\n")
        self._file.flush()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


class ChiselRecorder:
    """Owns the in-memory ring buffer, the optional JSONL sink, and the
    per-binary debug-dump policy."""

    def __init__(self) -> None:
        self.report: ChiselReport = ChiselReport()
        self._sink: Optional[JsonlSink] = None
        self._debug_chisel_dir: Optional[str] = None
        self._dumped_binaries: Set[int] = set()

    @property
    def results_path(self) -> Optional[str]:
        return self._sink.path if self._sink is not None else None

    @property
    def debug_chisel_dir(self) -> Optional[str]:
        return self._debug_chisel_dir

    def configure(
        self,
        *,
        results_path=_UNSET,
        report_capacity=_UNSET,
        debug_chisel_dir=_UNSET,
    ) -> None:
        if results_path is not _UNSET and results_path != self.results_path:
            self.close()
            if results_path is not None:
                self._sink = JsonlSink(results_path)
        if report_capacity is not _UNSET:
            self.report.resize(report_capacity)
        if debug_chisel_dir is not _UNSET:
            self._debug_chisel_dir = debug_chisel_dir

    def close(self) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None

    def write(
        self,
        record: ChiselRecord,
        *,
        program: Optional["ProgramState"],
        binary_state: Optional["BinaryState"],
    ) -> None:
        if binary_state is not None and record.binary_id is None:
            record.binary_id = binary_state.binary_id
        if program is not None:
            if record.program_name is None:
                record.program_name = program.program_name
            if record.program_index is None:
                record.program_index = program.program_index
        self.report.append(record)
        if self._sink is not None:
            self._sink.write(record)
        self._dump_debug(record, program, binary_state)

    def _dump_debug(
        self,
        record: ChiselRecord,
        program: Optional["ProgramState"],
        binary_state: Optional["BinaryState"],
    ) -> None:
        """On the first failure for a given binary, dump the source MLIR and
        flatbuffer so the user can debug offline. Records carry binary_id +
        program_name/program_index, so one dump per binary is enough."""
        if self._debug_chisel_dir is None:
            return
        if record.status in PASS_STATUS:
            return
        if program is None or binary_state is None or program._rt_binary is None:
            return
        binary_id = binary_state.binary_id
        if binary_id in self._dumped_binaries:
            return
        self._dumped_binaries.add(binary_id)
        paths = _dump_debug_artifacts(
            self._debug_chisel_dir,
            binary_id=binary_id,
            mlir_source=binary_state.mlir_source,
            rt_binary=program._rt_binary,
        )
        if paths is None:
            return
        mlir_path, fb_path = paths
        logger.warning(
            "chisel debug: dumped MLIR to %s and flatbuffer to %s "
            "(triggered by status=%s on op=%s program=%s[%d])",
            mlir_path,
            fb_path,
            record.status.value,
            record.op,
            program.program_name,
            program.program_index,
        )


def _dump_debug_artifacts(
    debug_dir: str,
    *,
    binary_id: int,
    mlir_source: str,
    rt_binary: Binary,
) -> Optional[Tuple[str, str]]:
    # Writes binary_{id}.mlir and binary_{id}.ttnn under debug_dir. Returns
    # the paths, or None on failure - failure is logged, never raised, since
    # this is a debug aid that must not break the run.
    try:
        os.makedirs(debug_dir, exist_ok=True)
        stem = f"binary_{binary_id}"
        mlir_path = os.path.join(debug_dir, stem + ".mlir")
        fb_path = os.path.join(debug_dir, stem + ".ttnn")
        with open(mlir_path, "w") as f:
            f.write(mlir_source)
        rt_binary.store(fb_path)
        return mlir_path, fb_path
    except Exception:
        logger.exception("Chisel debug dump failed.")
        return None
