# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest

from chisel.recorder import JsonlSink
from chisel.report import (
    ChiselBugPayload,
    ChiselRecord,
    ChiselReport,
    DtypeMismatchPayload,
    ErrorPayload,
    NoGoldenPayload,
    NumericsMode,
    NumericsPayload,
    ShapeMismatchPayload,
    SkippedNumericsPayload,
    RecordStatus,
)


def _records_for_each_status():
    return [
        ChiselRecord(
            op="ttnn.linear",
            check="numerics",
            ssa="%0",
            program_name="prog0",
            program_index=0,
            payload=NumericsPayload(
                status=RecordStatus.OK,
                mode=NumericsMode.ISOLATED,
                pcc=0.999,
                atol=1e-8,
                rtol=1e-5,
                device_id=0,
            ),
        ),
        ChiselRecord(
            op="ttnn.add",
            check="mlir_vs_tensor_ref",
            ssa="%1",
            payload=ShapeMismatchPayload(expected_shape=[1, 32], actual_shape=[1, 64]),
        ),
        ChiselRecord(
            op="ttnn.multiply",
            check="mlir_vs_tensor_ref",
            payload=DtypeMismatchPayload(expected_dtype="bf16", actual_dtype="f32"),
        ),
        ChiselRecord(
            op="ttnn.linear",
            check="numerics",
            op_asm="some asm",
            payload=NumericsPayload(
                status=RecordStatus.NUMERICS_FAIL,
                mode=NumericsMode.ACCUMULATED,
                pcc=0.42,
                atol=1.0,
                rtol=1.0,
                device_id=1,
            ),
        ),
        ChiselRecord(op="ttnn.linear", check="numerics", payload=ErrorPayload()),
        ChiselRecord(
            op="ttnn.add",
            check="numerics",
            payload=SkippedNumericsPayload(mode=NumericsMode.ISOLATED),
        ),
        ChiselRecord(
            op="ttnn.unknown",
            check="numerics",
            payload=ChiselBugPayload(traceback="boom\n"),
        ),
        ChiselRecord(
            op="ttnn.constant",
            check="numerics",
            payload=NoGoldenPayload(),
        ),
    ]


def test_model_json_roundtrip_per_status():
    for original in _records_for_each_status():
        line = original.model_dump_json(exclude_none=True)
        restored = ChiselRecord.model_validate_json(line)
        assert restored == original


def test_from_jsonl_roundtrip(tmp_path):
    path = tmp_path / "chisel_result.jsonl"
    sink = JsonlSink(str(path))
    originals = _records_for_each_status()
    for r in originals:
        sink.write(r)
    sink.close()

    loaded = ChiselReport.from_jsonl(str(path))
    assert len(loaded) == len(originals)
    # Default capacity matches the loaded count so analysis loads don't truncate.
    assert loaded.capacity == len(originals)
    assert loaded.records == originals


def test_from_jsonl_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.touch()
    loaded = ChiselReport.from_jsonl(str(path))
    assert len(loaded) == 0


def test_from_jsonl_respects_explicit_capacity(tmp_path):
    path = tmp_path / "cap.jsonl"
    sink = JsonlSink(str(path))
    for r in _records_for_each_status():
        sink.write(r)
    sink.close()

    loaded = ChiselReport.from_jsonl(str(path), capacity=3)
    assert loaded.capacity == 3
    # ring buffer evicts oldest - verify by tail-match.
    assert loaded.records == _records_for_each_status()[-3:]


def _build_report():
    report = ChiselReport(capacity=100)
    for r in _records_for_each_status():
        report.append(r)
    return report


def test_by_op_and_by_check_and_by_program():
    report = _build_report()

    linears = report.by_op("ttnn.linear")
    assert len(linears) == 3
    assert {r.op for r in linears} == {"ttnn.linear"}

    numerics = report.by_check("numerics")
    assert {r.check for r in numerics} == {"numerics"}
    # 6 out of 8 records use the "numerics" check.
    assert len(numerics) == 6

    prog0 = report.by_program("prog0")
    assert len(prog0) == 1
    assert prog0.records[0].program_name == "prog0"


def test_by_status_accepts_enum_str_and_iterable():
    report = _build_report()

    assert len(report.by_status(RecordStatus.OK)) == 1
    assert len(report.by_status("ok")) == 1

    multi = report.by_status([RecordStatus.OK, "numerics_fail"])
    assert {r.status for r in multi} == {RecordStatus.OK, RecordStatus.NUMERICS_FAIL}
    assert len(multi) == 2


def test_failures_and_passes_complement():
    report = _build_report()
    failures = report.failures()
    passes = report.passes()
    # Non-failure statuses: ok, skipped_numerics, no_golden -> 3 of the 8 records.
    assert len(passes) == 3
    assert len(failures) == 5
    assert len(failures) + len(passes) == len(report)
    # No overlap.
    failing_ids = {id(r) for r in failures}
    passing_ids = {id(r) for r in passes}
    assert failing_ids.isdisjoint(passing_ids)
