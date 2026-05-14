# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Smoke for `d2m.config` debug knobs: flags fire and don't break the
round-trip path."""

import io
import contextlib
import torch
import d2m_jit as d2m


def make_layout():
    return d2m.Layout(
        shape=(64, 64),
        dtype=d2m.float32,
        block_shape=[1, 1],
        grid_shape=[2, 2],
    )


def test_print_pipeline_fires():
    d2m.config.print_pipeline = True
    d2m.config.print_ir_before_pipeline = False
    d2m.config.print_ir_after_pipeline = False

    t = torch.randn(64, 64, dtype=torch.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = d2m.to_layout(t, make_layout()).to_host()
    output = buf.getvalue()
    assert (
        "[d2m-jit] pipeline:" in output
    ), "config.print_pipeline=True did not produce the expected stdout marker"
    assert torch.allclose(t, out), "round-trip broke under config.print_pipeline=True"

    d2m.config.print_pipeline = False


def test_print_ir_before_after():
    d2m.config.print_ir_before_pipeline = True
    d2m.config.print_ir_after_pipeline = True

    t = torch.randn(64, 64, dtype=torch.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = d2m.to_layout(t, make_layout()).to_host()
    output = buf.getvalue()
    assert "[d2m-jit] IR before pipeline:" in output
    assert "[d2m-jit] IR after pipeline:" in output
    assert torch.allclose(t, out)

    d2m.config.print_ir_before_pipeline = False
    d2m.config.print_ir_after_pipeline = False


def test_default_is_quiet():
    """With no flags set, no [d2m-jit] markers should appear in stdout."""
    # Reset to defaults explicitly.
    d2m.config.print_pipeline = False
    d2m.config.print_ir_before_pipeline = False
    d2m.config.print_ir_after_pipeline = False
    d2m.config.print_ir_after_each_pass = False

    t = torch.randn(64, 64, dtype=torch.float32)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = d2m.to_layout(t, make_layout()).to_host()
    output = buf.getvalue()
    assert "[d2m-jit]" not in output, f"unexpected debug output: {output!r}"
    assert torch.allclose(t, out)
