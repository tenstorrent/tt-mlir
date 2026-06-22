# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for skip-mode predicates and the golden->runtime tensor helper.

The full skip path (device write-back via update_tensor_in_pool) is exercised
on silicon by the builder chisel integration suite; here we cover the pure,
device-free units: the `chisel.skip` predicates, the SkippedOpPayload record.
"""
import torch

from golden import GoldenMapTensor

from chisel import skip
from chisel.report import (
    ChiselRecord,
    NumericsMode,
    NumericsPayload,
    RecordStatus,
    SkippedOpPayload,
)
from chisel.utils import get_torch_tensor, golden_to_runtime_tensor


def _numerics(mode, failed, pcc=0.5):
    return NumericsPayload(
        status=RecordStatus.NUMERICS_FAIL if failed else RecordStatus.OK,
        mode=mode,
        pcc=pcc,
        atol=0.0,
        rtol=0.0,
        device_id=0,
    )


# --- fakes (avoid constructing real MLIR ops for predicate tests) ----------


class _FakeType:
    def __init__(self, shape):
        self.shape = list(shape)
        self.element_type = "f32"


class _FakeResult:
    def __init__(self, name, shape=(2, 3)):
        self._name = name
        self.type = _FakeType(shape)

    def get_name(self, _asm_state):
        return self._name


class _FakeOp:
    def __init__(self, name, results=()):
        self.name = name
        self.results = list(results)


class _FakeCtx:
    def __init__(self, op, op_numerics=None):
        self.op = op
        self.asm_state = None
        # Mirror ChiselContext.op_numerics: list of NumericsPayload, or None.
        self.op_numerics = op_numerics


# --- skip_op_names ---------------------------------------------------------


def test_skip_op_names_matches():
    pred = skip.skip_op_names("ttnn.matmul", "ttnn.add")
    assert pred(_FakeCtx(_FakeOp("ttnn.matmul"))) is True
    assert pred(_FakeCtx(_FakeOp("ttnn.add"))) is True


def test_skip_op_names_no_match():
    pred = skip.skip_op_names("ttnn.matmul")
    assert pred(_FakeCtx(_FakeOp("ttnn.multiply"))) is False


# --- skip_ssa --------------------------------------------------------------


def test_skip_ssa_matches_output():
    op = _FakeOp("ttnn.add", results=[_FakeResult("%5")])
    assert skip.skip_ssa("%5")(_FakeCtx(op)) is True


def test_skip_ssa_no_match():
    op = _FakeOp("ttnn.add", results=[_FakeResult("%5")])
    assert skip.skip_ssa("%9")(_FakeCtx(op)) is False


# --- skip_on_bad_pcc -------------------------------------------------------


def test_skip_on_bad_pcc_fires_on_failure():
    ctx = _FakeCtx(
        _FakeOp("ttnn.matmul"), [_numerics(NumericsMode.ACCUMULATED, failed=True)]
    )
    assert skip.skip_on_bad_pcc()(ctx) is True


def test_skip_on_bad_pcc_quiet_when_passing():
    ctx = _FakeCtx(
        _FakeOp("ttnn.matmul"), [_numerics(NumericsMode.ACCUMULATED, failed=False)]
    )
    assert skip.skip_on_bad_pcc()(ctx) is False


def test_skip_on_bad_pcc_none_numerics():
    # No numerics recorded (e.g. skip_pcc op): predicate must not fire / raise.
    assert skip.skip_on_bad_pcc()(_FakeCtx(_FakeOp("ttnn.matmul"))) is False


def test_skip_on_bad_pcc_respects_mode():
    # Failed under ISOLATED only; an ACCUMULATED predicate must not fire.
    ctx = _FakeCtx(
        _FakeOp("ttnn.matmul"), [_numerics(NumericsMode.ISOLATED, failed=True)]
    )
    assert skip.skip_on_bad_pcc(NumericsMode.ACCUMULATED)(ctx) is False
    assert skip.skip_on_bad_pcc(NumericsMode.ISOLATED)(ctx) is True


# --- combinators -----------------------------------------------------------

_FAIL_ACC = [_numerics(NumericsMode.ACCUMULATED, failed=True)]


def test_any_of():
    pred = skip.any_of(skip.skip_op_names("ttnn.matmul"), skip.skip_on_bad_pcc())
    assert pred(_FakeCtx(_FakeOp("ttnn.matmul"))) is True
    assert pred(_FakeCtx(_FakeOp("ttnn.add"), _FAIL_ACC)) is True
    assert pred(_FakeCtx(_FakeOp("ttnn.add"))) is False


def test_all_of():
    pred = skip.all_of(skip.skip_op_names("ttnn.matmul"), skip.skip_on_bad_pcc())
    assert pred(_FakeCtx(_FakeOp("ttnn.matmul"), _FAIL_ACC)) is True
    # name matches but pcc passed -> False
    assert pred(_FakeCtx(_FakeOp("ttnn.matmul"))) is False
    # pcc failed but wrong name -> False
    assert pred(_FakeCtx(_FakeOp("ttnn.add"), _FAIL_ACC)) is False


# --- SkippedOpPayload record ----------------------------------------------


def test_skipped_op_payload_roundtrip():
    record = ChiselRecord(
        op="ttnn.matmul", check="skipped_op", payload=SkippedOpPayload()
    )
    assert record.status is RecordStatus.SKIPPED_OP
    # skip is an action, not a failure.
    assert record.status not in {RecordStatus.NUMERICS_FAIL, RecordStatus.ERROR}
    restored = ChiselRecord.model_validate_json(
        record.model_dump_json(exclude_none=True)
    )
    assert restored == record
