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

    _check(
        fn,
        torch.randn(32, 32, dtype=torch.bfloat16),
        torch.randn(32, 32, dtype=torch.bfloat16),
    )


def test_input_mutation_through_view():
    def fn(x, y):
        z = x.view(64, 16)
        z.add_(y)
        return x * 2

    _check(
        fn,
        torch.randn(32, 32, dtype=torch.bfloat16),
        torch.randn(64, 16, dtype=torch.bfloat16),
    )


def test_input_mutation_through_view2():
    def fn(x, y):
        z = x.view(64, 16)
        z.add_(y)
        x.mul_(3)
        return z * 2

    _check(
        fn,
        torch.randn(32, 32, dtype=torch.bfloat16),
        torch.randn(64, 16, dtype=torch.bfloat16),
    )


def test_input_mutation_through_view_of_view():
    def fn(x, y):
        z = x.view(64, 16)
        t = z.view(32, 2, 16)
        z.mul_(2)
        t.add_(y)
        return x

    _check(
        fn,
        torch.randn(32, 32, dtype=torch.bfloat16),
        torch.randn(32, 2, 16, dtype=torch.bfloat16),
    )


def test_multiple_inputs_mutated():
    def fn(x, y):
        x.add_(1)
        y.mul_(2)
        return x * 2

    _check(
        fn,
        torch.randn(32, 32, dtype=torch.bfloat16),
        torch.randn(32, 32, dtype=torch.bfloat16),
    )


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

    _check(
        fn,
        torch.randn(32, 32, dtype=torch.bfloat16),
        torch.randn(32, 32, dtype=torch.float32),
    )


def test_input_mutation_non_tile_aligned_shape():
    def fn(x, y):
        x.add_(y)
        return x * 2

    _check(
        fn,
        torch.randn(5, 7, dtype=torch.bfloat16),
        torch.randn(5, 7, dtype=torch.bfloat16),
    )


def test_repeated_calls_accumulate_mutation():
    def fn(x):
        x.add_(1)
        return x * 2

    for zero_copy in (True, False):
        torch._dynamo.reset()
        tt_x = torch.zeros(32, 32, dtype=torch.bfloat16).to("tt")
        options = {CompileOption.ENABLE_ZERO_COPY_INPUT_MUTATIONS: zero_copy}
        compiled = torch.compile(fn, backend="tt", options=options)
        for _ in range(3):
            out = compiled(tt_x)

        expected_x = torch.full((32, 32), 3.0, dtype=torch.bfloat16)
        torch.testing.assert_close(tt_x.cpu(), expected_x)
        torch.testing.assert_close(out.cpu(), expected_x * 2)


def test_same_tensor_passed_twice():
    def fn(x, y):
        x.add_(1)
        return x + y

    for zero_copy in (True, False):
        torch._dynamo.reset()
        tt_x = torch.zeros(32, 32, dtype=torch.bfloat16).to("tt")
        options = {CompileOption.ENABLE_ZERO_COPY_INPUT_MUTATIONS: zero_copy}
        out = torch.compile(fn, backend="tt", options=options)(tt_x, tt_x)

        ones = torch.ones(32, 32, dtype=torch.bfloat16)
        torch.testing.assert_close(tt_x.cpu(), ones)
        torch.testing.assert_close(out.cpu(), ones * 2)
