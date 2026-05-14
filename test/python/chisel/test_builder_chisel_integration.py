# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import List, Optional

import torch

import chisel
from builder.base.builder_apis import compile_and_execute_ttnn
from builder.base.builder_utils import Operand
from builder.ttnn.ttnn_builder import TTNNBuilder


def test_chisel_records_one_layer_nn(request, device, tmp_path):
    # multiply(y, y) instead of an activation: the TTNN optimizer fuses
    # activations into ttnn.linear, collapsing two ops into one.
    x_shape = (64, 128)
    w_shape = (256, 128)  # transpose_b=True → effective rhs is (128, 256)
    b_shape = (256,)

    def module(builder: TTNNBuilder):
        @builder.func(
            [x_shape, w_shape, b_shape],
            [torch.float32, torch.float32, torch.float32],
        )
        def one_layer_nn(
            x: Operand,
            w: Operand,
            b: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            y = builder.linear(x, w, bias=b, transpose_b=True)
            return builder.multiply(y, y)

    with chisel.session(results_path="chisel_result.jsonl") as report:
        compile_and_execute_ttnn(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    assert records, "chisel report is empty"

    ops_seen = {r.op for r in records}
    expected = {"ttnn.linear", "ttnn.multiply"}
    missing = expected - ops_seen
    assert not missing, (
        f"chisel report missing op records for: {sorted(missing)}. "
        f"Saw: {sorted(ops_seen)}"
    )

    PCC_THRESHOLD = 0.99
    pcc_records = [r for r in records if r.check == "numerics"]
    assert pcc_records, "no numerics (PCC) records produced"

    for op in expected:
        op_pcc = [r for r in pcc_records if r.op == op]
        assert op_pcc, f"no PCC record for {op}"
        for r in op_pcc:
            assert (
                r.status == "ok"
            ), f"{op}: PCC check status={r.status} payload={r.payload}"
            assert (
                r.payload.pcc >= PCC_THRESHOLD
            ), f"{op}: PCC {r.payload.pcc} below threshold {PCC_THRESHOLD}"


def test_chisel_dumps_debug_artifacts_on_pcc_fail(request, device, tmp_path):
    # Set the PCC threshold above the [-1, 1] range so every numerics check
    # records NUMERICS_FAIL, and verify the recorder dumps the MLIR source
    # and flatbuffer for the offending binary to debug_chisel_dir.
    x_shape = (64, 128)
    w_shape = (256, 128)
    b_shape = (256,)

    def module(builder: TTNNBuilder):
        @builder.func(
            [x_shape, w_shape, b_shape],
            [torch.float32, torch.float32, torch.float32],
        )
        def one_layer_nn(
            x: Operand,
            w: Operand,
            b: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            y = builder.linear(x, w, bias=b, transpose_b=True)
            return builder.multiply(y, y)

    debug_dir = tmp_path / "chisel_debug"

    with chisel.session(
        debug_chisel_dir=str(debug_dir),
        checks_config=chisel.ChiselChecksConfig(threshold=2.0),
    ) as report:
        compile_and_execute_ttnn(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    pcc_records = [r for r in records if r.check == "numerics"]
    assert pcc_records, "no numerics (PCC) records produced"
    assert all(
        r.status == chisel.Status.NUMERICS_FAIL for r in pcc_records
    ), f"expected all PCC records to fail with threshold=2.0, got {[r.status for r in pcc_records]}"

    assert debug_dir.is_dir(), f"debug dir not created at {debug_dir}"
    mlir_files = sorted(debug_dir.glob("binary_*.mlir"))
    fb_files = sorted(debug_dir.glob("binary_*.ttnn"))
    assert mlir_files, f"no MLIR dumped under {debug_dir}: {list(debug_dir.iterdir())}"
    assert (
        fb_files
    ), f"no flatbuffer dumped under {debug_dir}: {list(debug_dir.iterdir())}"
    # One dump per binary — the policy is first-failure-wins.
    assert len(mlir_files) == 1
    assert len(fb_files) == 1
    for mlir_path, fb_path in zip(mlir_files, fb_files):
        assert mlir_path.stat().st_size > 0, f"empty MLIR dump at {mlir_path}"
        assert fb_path.stat().st_size > 0, f"empty flatbuffer dump at {fb_path}"


def test_chisel_records_chisel_bug_on_callback_raise(
    request, device, tmp_path, monkeypatch
):
    # Monkey-patch the numerics check that _default_post_op calls so it
    # raises. chisel_safe should swallow the exception, write a chisel_bug
    # record carrying the traceback, and the ttmlir runtime should finish
    # the program normally so the test exits the session cleanly.
    import chisel.callbacks as chisel_callbacks

    boom_message = "synthetic chisel bug for testing chisel_safe"

    def boom(*args, **kwargs):
        raise RuntimeError(boom_message)

    monkeypatch.setattr(chisel_callbacks, "check_numerics", boom)

    x_shape = (64, 128)
    w_shape = (256, 128)
    b_shape = (256,)

    def module(builder: TTNNBuilder):
        @builder.func(
            [x_shape, w_shape, b_shape],
            [torch.float32, torch.float32, torch.float32],
        )
        def one_layer_nn(
            x: Operand,
            w: Operand,
            b: Operand,
            builder: TTNNBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            y = builder.linear(x, w, bias=b, transpose_b=True)
            return builder.multiply(y, y)

    with chisel.session() as report:
        # Must not raise — chisel_safe is supposed to keep the runtime alive
        # even when a chisel callback explodes.
        compile_and_execute_ttnn(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    assert records, "chisel report is empty — runtime did not produce any records"

    bug_records = [r for r in records if r.status == chisel.Status.CHISEL_BUG]
    assert bug_records, (
        "expected at least one chisel_bug record, got statuses: "
        f"{[r.status for r in records]}"
    )

    for r in bug_records:
        tb = r.payload.traceback
        assert tb, f"chisel_bug record has empty traceback for op={r.op}"
        assert (
            boom_message in tb
        ), f"traceback for op={r.op} missing the raised message; got:\n{tb}"
        assert (
            "RuntimeError" in tb
        ), f"traceback for op={r.op} missing the exception type; got:\n{tb}"

    bug_ops = {r.op for r in bug_records}
    assert bug_ops == {
        "ttnn.linear",
        "ttnn.multiply",
    }, f"expected chisel_bug for both ops, got {bug_ops}"
