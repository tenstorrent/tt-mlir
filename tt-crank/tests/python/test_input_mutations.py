# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import torch


def _fn(x, y):
    x.add_(y)
    x.mul_(2)
    return x + y


def test_input_mutations():
    x, y = torch.randn(32, 32, dtype=torch.bfloat16), torch.randn(32, 32, dtype=torch.bfloat16)
    x_tt, y_tt = x.to("tt"), y.to("tt")

    x_ref = x.clone()
    out = _fn(x_ref, y)
    out_tt = torch.compile(_fn, backend="tt")(x_tt, y_tt)

    torch.testing.assert_close(out_tt.cpu(), out)
    torch.testing.assert_close(x_tt.cpu(), x_ref)


def _view_fn(x, y):
    z = x.view(64, 16)
    z.add_(y)
    return x * 2


def test_input_mutation_through_view():
    x, y = torch.randn(32, 32, dtype=torch.bfloat16), torch.randn(64, 16, dtype=torch.bfloat16)
    x_tt, y_tt = x.to("tt"), y.to("tt")
    x_ref = x.clone()
    out = _view_fn(x_ref, y)
    out_tt = torch.compile(_view_fn, backend="tt")(x_tt, y_tt)

    torch.testing.assert_close(out_tt.cpu(), out)
    torch.testing.assert_close(x_tt.cpu(), x_ref)


def _view_of_view_fn(x, y):
    z = x.view(64, 16)
    t = z.view(32, 2, 16)
    z.mul_(2)
    t.add_(y)
    return x


def test_input_mutation_through_view_of_view():
    x, y = torch.randn(32, 32, dtype=torch.bfloat16), torch.randn(32, 2, 16, dtype=torch.bfloat16)
    x_tt, y_tt = x.to("tt"), y.to("tt")

    x_ref = x.clone()
    out = _view_of_view_fn(x_ref, y)
    out_tt = torch.compile(_view_of_view_fn, backend="tt")(x_tt, y_tt)

    torch.testing.assert_close(out_tt.cpu(), out)
    torch.testing.assert_close(x_tt.cpu(), x_ref)