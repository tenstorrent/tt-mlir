# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Generic, data-driven runner for bundled d2m-jit pattern tests.

This file is intentionally tiny and pattern-agnostic: it just executes the
specs each pattern file declares (see d2m_jit.testing). It replaces the
hand-written, per-pattern test_pattern_eltwise.py + lit/*_pattern.py files.

  test_pattern_rewrite : TTIR -> apply this file's pattern(s) -> FileCheck.
                         No device. Covers the rewrite path (lit replacement).
  test_kernel_device   : drive the @d2m.kernel directly -> PCC vs torch.
                         On silicon (direct-kernel path).
"""

from d2m_jit.testing import assert_pcc, filecheck, run_bench, run_rewrite


def test_pattern_rewrite(pattern_test):
    """Apply the pattern file's rewrites and FileCheck the resulting IR."""
    ir_text = run_rewrite(pattern_test)
    if pattern_test.check:
        filecheck(pattern_test.check, ir_text)


def test_kernel_device(kernel_bench):
    """Run the kernel on device at its default config and PCC-compare."""
    actual, expected = run_bench(kernel_bench)
    assert_pcc(expected, actual, kernel_bench.pcc)
