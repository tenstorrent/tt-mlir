# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generic, data-driven runner for bundled d2m-jit pattern tests.

This file is intentionally tiny and pattern-agnostic: it just executes the
specs each pattern file declares (see runner). It replaces the
hand-written, per-pattern test_pattern_eltwise.py + lit/*_pattern.py files.

  test_pattern_rewrite : TTIR -> apply this file's pattern(s) -> FileCheck.
                         No device. Covers the rewrite path (lit replacement).
  test_kernel_device   : drive the @d2m.kernel directly -> PCC vs torch.
                         On silicon (direct-kernel path).
"""

import pytest

import d2m_jit as d2m
from runner import (
    assert_pcc,
    filecheck,
    run_bench,
    run_e2e,
    run_rewrite,
)

# These cover the MLIR pattern-rewrite path and the on-silicon e2e path -- both
# compiler/device, not the torch simulator. Skip under D2M_JIT_SIM so the same
# `pytest test/d2m-jit/` invocation is green in both backends.
pytestmark = pytest.mark.skipif(
    d2m.config.simulator,
    reason="compiler-path test; not applicable to the torch simulator (D2M_JIT_SIM)",
)


def test_pattern_rewrite(pattern_test):
    """Apply the pattern file's rewrites and FileCheck the resulting IR."""
    ir_text = run_rewrite(pattern_test)
    if not pattern_test.expect_match and not pattern_test.check:
        raise AssertionError(
            "Negative PatternTest cases (expect_match=False) must provide FileCheck directives via `check`."
        )
    if pattern_test.check:
        filecheck(pattern_test.check, ir_text)


def test_kernel_device(kernel_bench):
    """Run the kernel on device at its default config and PCC-compare."""
    actual, expected = run_bench(kernel_bench)
    assert_pcc(expected, actual, kernel_bench.pcc)


def test_e2e_device(e2e_spec, e2e_device):
    """True e2e: the rewritten module is compiled to a flatbuffer and run on
    device IN-PROCESS (no ttrt subprocess, no files). PCC the device output
    against the golden computed from the deterministic inputs."""
    pcc, _, _ = run_e2e(e2e_spec, e2e_device)
    assert pcc >= e2e_spec.pcc, f"e2e PCC {pcc} < {e2e_spec.pcc} for {e2e_spec.name}"
