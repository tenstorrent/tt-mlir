# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Chisel coverage for CPU-hoisted func.CallOps.

A hoisted op is compiled into a CpuOp wrapping a dylib; chisel's func.CallOp
override re-runs the dylib via tt_runtime.invoke_cpu_op with the propagated
goldens from the accumulation pool, seeds the call's outputs back into the
pool, and PCC-checks against the device tensor. We assert the accumulated
numerics record is present and OK, and that no NoGoldenPayload was emitted
for the hoisted call (the override replaces the previous no_golden behavior).
"""
import platform
from typing import List, Optional

import pytest
import torch

import chisel
from builder.base.builder_apis import compile_and_execute_ttir
from builder.base.builder_utils import Operand
from builder.ttir.ttir_builder import TTIRBuilder


def test_chisel_records_cpu_hoisted_call(request, device, tmp_path):
    x_shape = (64, 128)
    y_shape = (64, 128)

    def module(builder: TTIRBuilder):
        @builder.func([x_shape, y_shape], [torch.float32, torch.float32])
        def hoisted_add(
            x: Operand,
            y: Operand,
            builder: TTIRBuilder,
            unit_attrs: Optional[List[str]] = None,
        ):
            # The add gets hoisted into a CPU dylib by HoistCPUOps; downstream
            # multiply runs on device against the hoisted output, exercising
            # propagation of the cpu-invoked golden through the pool.
            hoisted = builder.add(x, y, unit_attrs=["ttir.should_hoist"])
            return builder.multiply(hoisted, hoisted)

    with chisel.session(
        results_path=str("chisel_result.jsonl"),
        checks_config=chisel.ChiselChecksConfig(accumulation=True),
    ) as report:
        compile_and_execute_ttir(
            module,
            test_base=request.node.name,
            output_root=str(tmp_path),
            target="ttnn",
            device=device,
        )
        records = report.records

    assert records, "chisel report is empty"

    hoisted_records = [r for r in records if r.op == "func.call"]
    assert hoisted_records, (
        "no func.call records produced - hoisting did not take effect or "
        "chisel skipped the op"
    )

    # The new override should not emit NoGoldenPayload for a hoisted call.
    no_golden = [r for r in hoisted_records if r.check == "golden_not_implemented"]
    assert not no_golden, (
        f"func.call still emitted golden_not_implemented: {no_golden}; "
        "expected the cpu-hoist handler to take over"
    )

    PCC_THRESHOLD = 0.99
    accum_pcc = [
        r
        for r in hoisted_records
        if r.check == "numerics" and r.payload.mode == chisel.NumericsMode.ACCUMULATED
    ]
    assert accum_pcc, "no accumulated PCC record for the hoisted func.call"
    for r in accum_pcc:
        assert (
            r.status == chisel.RecordStatus.OK
        ), f"hoisted func.call accumulated PCC status={r.status} payload={r.payload}"
        assert (
            r.payload.pcc >= PCC_THRESHOLD
        ), f"hoisted func.call PCC {r.payload.pcc} below {PCC_THRESHOLD}"

    # Downstream multiply consumes the cpu-invoked golden through the pool;
    # if propagation works it produces an accumulated PCC record too.
    mul_accum = [
        r
        for r in records
        if r.op == "ttnn.multiply"
        and r.check == "numerics"
        and r.payload.mode == chisel.NumericsMode.ACCUMULATED
    ]
    assert mul_accum, (
        "no accumulated PCC record for downstream ttnn.multiply - "
        "cpu-invoked golden likely failed to propagate through the pool"
    )
    for r in mul_accum:
        assert r.status == chisel.RecordStatus.OK
        assert r.payload.pcc >= PCC_THRESHOLD
