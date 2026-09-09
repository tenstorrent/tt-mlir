# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

"""FileCheck the exact shape of a D2mJitError message.

This complements `test/d2m-jit/test_errors.py` (pytest, runtime-facing)
by locking down the formatted error layout (header, gutter, marker,
caret, cause line, hint) without depending on pytest internals or the
device runtime."""

import torch
import d2m_jit as d2m


_L = d2m.Layout(
    shape=(64, 64), dtype=d2m.float32, block_shape=[1, 1], grid_shape=[1, 1]
)


@d2m.kernel
def k(in_t, out_t):
    x = remote_load(in_t, [0, 0])
    y = x.sigmoidd()  # typo
    remote_store(out_t, [0, 0], y)


t = torch.zeros(64, 64, dtype=torch.float32)
in_lt = d2m.to_layout(t, _L)
out_lt = d2m.empty(_L)

try:
    k(in_lt, out_lt, grid=(1, 1))
except d2m.D2mJitError as e:
    print(e)


# Header points back at THIS source file with the offending line.
# CHECK: d2m_jit error at {{.*}}/errors.py:{{[0-9]+}}:{{[0-9]+}}

# Code excerpt: context line + arrowed bad line + context line, with caret
# pinning the bad expression.
# CHECK:    {{[0-9]+}} |     x = remote_load(in_t, [0, 0])
# CHECK: --> {{[0-9]+}} |     y = x.sigmoidd()  # typo
# CHECK: ^
# CHECK:    {{[0-9]+}} |     remote_store(out_t, [0, 0], y)

# Cause and hint trail the excerpt.
# CHECK: AttributeError: `x` of type !tensor has no method 'sigmoidd'
# CHECK: hint: did you mean `sigmoid`?
