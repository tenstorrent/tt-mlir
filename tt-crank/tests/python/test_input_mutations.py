# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch

from tt_crank.torch._compile import CompileOption


def _run_tt(fn, inputs, zero_copy):
    torch._dynamo.reset()
    tt_inputs = [t.clone().to("tt") for t in inputs]
    options = {CompileOption.ENABLE_ZERO_COPY_INPUT_MUTATIONS: zero_copy}
    out = torch.compile(fn, backend="tt", options=options)(*tt_inputs)
    return out.cpu(), [t.cpu() for t in tt_inputs]


def _check(fn, *inputs):
    cpu_inputs = [t.clone() for t in inputs]
    cpu_out = fn(*cpu_inputs)

    on_out, on_inputs = _run_tt(fn, inputs, zero_copy=True)
    off_out, off_inputs = _run_tt(fn, inputs, zero_copy=False)

    torch.testing.assert_close(on_out, off_out)
    torch.testing.assert_close(on_inputs, off_inputs)

    torch.testing.assert_close(on_out, cpu_out)
    torch.testing.assert_close(on_inputs, cpu_inputs)


def test_input_mutations():
    def fn(x, y):
        x.add_(y)
        x.mul_(2)
        return x + y

    _check(fn, torch.randn(32, 32, dtype=torch.bfloat16), torch.randn(32, 32, dtype=torch.bfloat16))


def test_input_mutation_through_view():
    def fn(x, y):
        z = x.view(64, 16)
        z.add_(y)
        return x * 2

    _check(fn, torch.randn(32, 32, dtype=torch.bfloat16), torch.randn(64, 16, dtype=torch.bfloat16))


def test_input_mutation_through_view_of_view():
    def fn(x, y):
        z = x.view(64, 16)
        t = z.view(32, 2, 16)
        z.mul_(2)
        t.add_(y)
        return x
    
    _check(fn, torch.randn(32, 32, dtype=torch.bfloat16), torch.randn(32, 2, 16, dtype=torch.bfloat16))


def test_multiple_inputs_mutated():
    def fn(x, y):
        x.add_(1)
        y.mul_(2)
        return x * 2

    _check(fn, torch.randn(32, 32, dtype=torch.bfloat16), torch.randn(32, 32, dtype=torch.bfloat16))


def test_input_mutation_through_slice():
    def fn(x):
        x[32:].add_(1)
        return x * 2

    _check(fn, torch.randn(64, 32, dtype=torch.bfloat16))


def test_input_mutation_through_transpose():
    def fn(x):
        x.t().add_(1)
        return x * 2

    _check(fn, torch.randn(32, 64, dtype=torch.bfloat16))


def test_mutation_with_different_dtype_operand():
    def fn(x, y):
        x.add_(y)
        return x * 2

    _check(fn, torch.randn(32, 32, dtype=torch.bfloat16), torch.randn(32, 32, dtype=torch.float32))

