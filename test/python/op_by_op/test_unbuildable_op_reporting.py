# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Ops that cannot be wrapped in a module must still be reported.

Regression tests: `execute_extracted_ops` used to `continue` past such an op, so
it appeared in no result -- neither passed nor failed. The op vanished from the
totals and the failure survived only as a line in a job log, which meant the
op-by-op data could not be used to count how many ops were actually attempted.
"""

import pytest

from op_by_op_infra.execution_result import (
    MAX_ERROR_MESSAGE_CHARS,
    _truncate_error_message,
)
from op_by_op_infra.workflow_internal import build_unbuildable_op_model


@pytest.fixture(scope="session", autouse=True)
def _bootstrap_system_desc():
    """Override the package-wide fixture: these tests never touch a device.

    The conftest fixture generates a system descriptor, which needs hardware.
    Everything here is pure record construction, so requiring a device would only
    make the tests unrunnable on a dev box and on any CPU-only CI runner.
    """
    yield


class _FakeOp:
    """Duck-typed stand-in for OpWrapper.

    Only `op_name` and `origin_model` are read when building the record, and a
    real OpWrapper cannot be constructed without a live MLIR context, so the two
    attributes are supplied directly. `origin_model` is a list on OpWrapper.
    """

    def __init__(self, op_name, origin_model):
        self.op_name = op_name
        self.origin_model = origin_model

    def as_module_str(self):
        # Present so this fake stands in for a real OpWrapper on both the old and
        # the new failure path; the old one printed the whole module here.
        return "module { }"


def test_unbuildable_op_is_reported_as_a_failure():
    model = build_unbuildable_op_model(
        _FakeOp("stablehlo.custom_call", ["pytorch_SomeModel"]),
        RuntimeError("operand #0 has no defining op"),
    )

    # The point of the fix: the op is present, and present as a failure.
    assert model.success is False
    assert model.op_name == "stablehlo.custom_call"
    assert model.model_name == "pytorch_SomeModel"
    assert "Failed to create module from op" in model.error_message
    assert "operand #0 has no defining op" in model.error_message

    # No compiled module exists for these ops, so there are no tensor descriptions
    # to report. Empty rather than absent, so the record still validates.
    assert model.inputs == []
    assert model.outputs == []
    assert model.test_start_ts is not None and model.test_end_ts is not None


def test_multi_model_origin_is_joined_like_the_executed_path():
    """Deduplicated ops carry several origin models; both paths must agree.

    `ModuleWrapper` joins with ", " (utils.py), so an unbuildable op has to use
    the same separator or the two paths would write different shapes of value for
    the same column.
    """
    model = build_unbuildable_op_model(
        _FakeOp("stablehlo.dot_general", ["model_a", "model_b"]),
        ValueError("nope"),
    )
    assert model.model_name == "model_a, model_b"


def test_empty_origin_model_becomes_none_not_empty_string():
    model = build_unbuildable_op_model(_FakeOp("stablehlo.add", [""]), ValueError("x"))
    assert model.model_name is None


def test_op_name_is_coerced_to_str():
    """`op_name` can be an MLIR StringAttr rather than a plain str."""

    class _AttrLike:
        def __str__(self):
            return "stablehlo.reshape"

    model = build_unbuildable_op_model(_FakeOp(_AttrLike(), ["m"]), ValueError("boom"))
    assert model.op_name == "stablehlo.reshape"
    assert isinstance(model.op_name, str)


def test_execute_extracted_ops_reports_rather_than_drops(monkeypatch):
    """The actual regression: the op must appear in the returned list.

    Stubs the device path so this stays a test of the loop's bookkeeping -- the
    executor and system descriptor are irrelevant to whether an op that failed
    `as_module()` survives into the results.
    """
    from op_by_op_infra import workflow

    good = _FakeOp("stablehlo.add", ["model_a"])
    bad = _FakeOp("stablehlo.custom_call", ["model_b"])
    good.as_module = lambda: "module-for-good"

    def _raise():
        raise RuntimeError("cannot build module")

    bad.as_module = _raise

    monkeypatch.setattr(workflow, "_ensure_system_desc", lambda: None)

    class _StubExecutor:
        def __init__(self, *args, **kwargs):
            pass

        def execute(self, module):
            return module

    monkeypatch.setattr(workflow.workflow_internal, "MLIRModuleExecutor", _StubExecutor)
    monkeypatch.setattr(
        workflow.workflow_internal,
        "convert_results_to_pydantic_models",
        lambda results, frontend=None: [
            build_unbuildable_op_model(good, RuntimeError("executed-placeholder"))
            for _ in results
        ],
    )

    models = workflow.execute_extracted_ops([good, bad], frontend="tt-xla")

    # One record for the executed op, one for the op that could not be built.
    # Before the fix the second was dropped and this returned a single record.
    assert len(models) == 2

    unbuildable = [m for m in models if m.op_name == "stablehlo.custom_call"]
    assert len(unbuildable) == 1, "the unbuildable op must be reported, not dropped"
    assert unbuildable[0].success is False
    assert "cannot build module" in unbuildable[0].error_message
    # Frontend still gets stamped on the synthesized record.
    assert unbuildable[0].frontend == "tt-xla"


def test_short_error_messages_are_left_alone():
    message = "operand #0 has no defining op"
    assert _truncate_error_message(message) == message


def test_long_error_messages_are_truncated_and_say_so():
    """Compiler errors can embed a whole module; one real one was 57k characters."""
    message = "x" * (MAX_ERROR_MESSAGE_CHARS + 5000)

    truncated = _truncate_error_message(message)

    assert len(truncated) < len(message)
    assert truncated.startswith("x" * 100)
    # The amount dropped is stated, so a truncated message is never mistaken for
    # the whole error.
    assert "truncated 5000 of" in truncated


def test_unbuildable_op_error_message_is_bounded():
    model = build_unbuildable_op_model(
        _FakeOp("stablehlo.add", ["m"]), RuntimeError("y" * 50_000)
    )
    assert len(model.error_message) < 50_000
    assert "truncated" in model.error_message
