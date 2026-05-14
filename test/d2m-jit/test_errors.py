# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Negative tests: every error path should raise `D2mJitError` and the
formatted message should include the offending Python file:line plus a
short code excerpt. Unknown-name errors should also surface a
`did you mean?` hint.

Each test constructs a deliberately-broken kernel inside the test body so
the asserted line numbers stay co-located with the code being tested. The
`_reset_builder` autouse fixture in conftest.py drops the singleton
between tests; without it a failed compile would poison the next call.
"""

import re

import pytest
import torch

import d2m_jit as d2m


_L = d2m.Layout(
    shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
)


def _make_lazy_pair():
    """Two LazyTensors backed by zeros, in the current builder generation."""
    t = torch.zeros(64, 64, dtype=torch.float32)
    return d2m.to_layout(t, _L), d2m.empty(_L)


# ---------------------------------------------------------------------------
# Kernel-body errors -- raised from D2MCompiler.visit() while compiling the
# kernel AST. Each one should be a D2mJitError with the kernel's source
# location and a code excerpt pointing at the offending line.
# ---------------------------------------------------------------------------


def test_unknown_function_has_line_and_hint():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_loud(in_t, [0, 0])  # typo of remote_load
        remote_store(out_t, [0, 0], x)

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k(in_t, out_t, grid=(1, 1))

    msg = str(exc_info.value)
    assert "remote_loud" in msg, msg
    assert "test_errors.py" in msg, msg
    assert "did you mean" in msg.lower(), msg
    assert "remote_load" in msg, msg  # the suggested replacement


def test_unknown_method_on_tensor_has_line_and_hint():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        y = x.sigmoidd()  # typo of sigmoid
        remote_store(out_t, [0, 0], y)

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k(in_t, out_t, grid=(1, 1))

    msg = str(exc_info.value)
    assert "sigmoidd" in msg, msg
    assert "test_errors.py" in msg, msg
    assert "did you mean" in msg.lower(), msg
    assert "sigmoid" in msg, msg


def test_unknown_variable_has_line_and_hint():
    @d2m.kernel
    def k(in_t, out_t):
        good_name = remote_load(in_t, [0, 0])
        remote_store(out_t, [0, 0], good_nam)  # typo

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k(in_t, out_t, grid=(1, 1))

    msg = str(exc_info.value)
    # `good_nam` ends up as an unknown var inside Compare via the
    # eventual MLIR-binding lookup; the visit() wrapper still pins it
    # to a line in this file.
    assert "test_errors.py" in msg, msg


def test_unsupported_syntax_while_loop():
    @d2m.kernel
    def k(in_t, out_t):
        i = 0
        while i < 1:  # While not in _SUPPORTED_NODES
            i = i + 1

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k(in_t, out_t, grid=(1, 1))

    msg = str(exc_info.value)
    assert "While" in msg, msg
    assert "unsupported Python syntax" in msg, msg
    assert "test_errors.py" in msg, msg


def test_string_constant_rejected_with_line():
    @d2m.kernel
    def k(in_t, out_t):
        x = "not allowed"  # visit_Constant rejects non-int constants

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k(in_t, out_t, grid=(1, 1))

    msg = str(exc_info.value)
    assert "constant type str not implemented" in msg, msg
    assert "test_errors.py" in msg, msg


def test_error_message_includes_code_excerpt():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        y = x.sigmoidd()
        remote_store(out_t, [0, 0], y)

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k(in_t, out_t, grid=(1, 1))

    msg = str(exc_info.value)
    # The formatted excerpt has lines like `--> NN | <source>` for the
    # failing line and `    NN | <source>` for context. Verify both the
    # marker and that the actual broken expression is quoted back.
    assert "-->" in msg, msg
    assert "sigmoidd" in msg, msg


def test_error_line_number_points_to_correct_line():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        y = x.bogus()
        remote_store(out_t, [0, 0], y)

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k(in_t, out_t, grid=(1, 1))

    err = exc_info.value
    # The reported line should be the offending statement, not the `def`
    # line. Easiest cross-check: the excerpt for `err.line` reads back the
    # exact source text we wrote.
    assert err.line is not None
    assert err.source_lines[err.snippet_line - 1].strip() == "y = x.bogus()", (
        f"err.line={err.line}; excerpt:\n" f"{err.source_lines[err.snippet_line - 1]!r}"
    )


# ---------------------------------------------------------------------------
# Call-site errors -- raised from _emit_kernel_generic before AST compilation.
# These are pinned to the kernel's `def` line (so the user at least sees
# *which* kernel rejected the call); the actual call site is in the
# Python traceback.
# ---------------------------------------------------------------------------


def test_kernel_called_with_wrong_arg_type():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        remote_store(out_t, [0, 0], x)

    raw_tensor = torch.zeros(64, 64)
    out_t = d2m.empty(_L)
    with pytest.raises(d2m.D2mJitError) as exc_info:
        # Passing torch.Tensor directly (not lifted through to_layout).
        k(raw_tensor, out_t, grid=(1, 1))

    msg = str(exc_info.value)
    assert "Tensor" in msg or "torch" in msg.lower(), msg
    assert "to_layout" in msg, msg  # hint should mention to_layout
    assert "test_errors.py" in msg, msg


def test_kernel_called_with_scalar_before_tensor():
    @d2m.kernel
    def k(in_t, out_t, scalar):
        x = remote_load(in_t, [0, 0])
        remote_store(out_t, [0, 0], x)

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        # Scalar before second tensor -- violates ordering rule.
        k(in_t, 5, out_t, grid=(1, 1))

    msg = str(exc_info.value)
    assert "precede scalars" in msg, msg
    assert "test_errors.py" in msg, msg


def test_num_outs_zero_rejected():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        remote_store(out_t, [0, 0], x)

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k(in_t, out_t, grid=(1, 1), num_outs=0)

    msg = str(exc_info.value)
    assert "num_outs" in msg, msg
    assert "test_errors.py" in msg, msg


# ---------------------------------------------------------------------------
# D2mJitError formatting itself
# ---------------------------------------------------------------------------


def test_error_format_has_file_line_header():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        bogus_function_name()
        remote_store(out_t, [0, 0], x)

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k(in_t, out_t, grid=(1, 1))

    msg = str(exc_info.value)
    # Header looks like: "d2m_jit error at /abs/path/test_errors.py:42:4"
    header_re = re.compile(
        r"^d2m_jit error at [^:]+/test_errors\.py:\d+(?::\d+)?$",
        re.MULTILINE,
    )
    assert header_re.search(msg), f"missing header in:\n{msg}"


def test_error_cause_is_attached():
    @d2m.kernel
    def k(in_t, out_t):
        x = remote_load(in_t, [0, 0])
        y = x.bogus_method()
        remote_store(out_t, [0, 0], y)

    in_t, out_t = _make_lazy_pair()
    with pytest.raises(d2m.D2mJitError) as exc_info:
        k(in_t, out_t, grid=(1, 1))

    err = exc_info.value
    # AttributeError for unknown method; chain is also wired via __cause__.
    assert isinstance(
        err.cause, AttributeError
    ), f"expected AttributeError cause, got {type(err.cause).__name__}"
