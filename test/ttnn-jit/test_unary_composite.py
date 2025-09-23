# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch

import pytest

from math import *
from utils import _build_golden_map, _get_ttnn_op


def cosh(input_tensor):
    e_pos_x = exp(input_tensor)
    e_neg_x = exp(-input_tensor)
    nr_term = e_pos_x + e_neg_x
    return nr_term * 0.5


def sinh(input_tensor):
    e_pos_x = exp(input_tensor)
    e_neg_x = exp(-input_tensor)
    nr_term = e_pos_x - e_neg_x
    return nr_term * 0.5


def cbrt(input_tensor):
    t_ln_input = log(abs(input_tensor))
    t1 = t_ln_input * 0.3333333333333333
    t2 = exp(t1)
    t3 = t2 * sign(input_tensor)
    return t3


def softsign(input_tensor):
    # softsign = x / (1 + |x|)
    return input_tensor * reciprocal(abs(input_tensor) + 1.0)


# Failing all_close
def selu(input_tensor):
    x_expm1 = expm1(input_tensor)
    # alpha = 1.67326
    result_t2 = x_expm1 * 1.67326

    # DefaultQueueID = 0
    # TODO: can't pass 0.0 to minimum/maximum
    default_queue_id = input_tensor - input_tensor
    result_term2 = minimum(default_queue_id, result_t2)
    x_max = maximum(default_queue_id, input_tensor)
    sum_max_term_2 = x_max + result_term2

    # scale = 1.050701
    result_selu = sum_max_term_2 * 1.050701
    return result_selu


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("op", [cosh, sinh, cbrt, softsign])
def test_unary_ops(device, h, w, op):
    torch.manual_seed(0)
    golden_op = _get_ttnn_op(op)
    torch_input_tensor = torch.randn((h, w), dtype=torch.float32)
    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device
    )

    op_jit = ttnn_jit.jit(backend="ttnn", debug=True)(op)
    output_tensor = op_jit(input_tensor)
    golden_tensor = golden_op(input_tensor)

    all_close = torch.allclose(
        output_tensor.cpu().to_torch(), golden_tensor.cpu().to_torch(), atol=1e-2
    )
    assert all_close


# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


def cosh(input_tensor):
    e_pos_x = exp(input_tensor)
    e_neg_x = exp(-input_tensor)
    nr_term = e_pos_x + e_neg_x
    return nr_term * 0.5


def sinh(input_tensor):
    e_pos_x = exp(input_tensor)
    e_neg_x = exp(-input_tensor)
    nr_term = e_pos_x - e_neg_x
    return nr_term * 0.5


def cbrt(input_tensor):
    t_ln_input = log(abs(input_tensor))
    t1 = t_ln_input * 0.3333333333333333
    t2 = exp(t1)
    t3 = t2 * sign(input_tensor)
    return t3


def softsign(input_tensor):
    # softsign = x / (1 + |x|)
    return input_tensor * reciprocal(abs(input_tensor) + 1.0)


def selu(input_tensor):
    x_expm1 = expm1(input_tensor)
    # alpha = 1.67326
    result_t2 = x_expm1 * 1.67326

    # DefaultQueueID = 0
    # TODO: can't pass 0.0 to minimum/maximum
    default_queue_id = input_tensor - input_tensor
    result_term2 = minimum(default_queue_id, result_t2)
    x_max = maximum(default_queue_id, input_tensor)
    sum_max_term_2 = x_max + result_term2

    # scale = 1.050701
    result_selu = sum_max_term_2 * 1.050701
    return result_selu


@pytest.mark.parametrize("h", [32])
@pytest.mark.parametrize("w", [32])
@pytest.mark.parametrize("op", [cosh, sinh, cbrt, softsign, selu])
def test_unary_ops(device, h, w, op):
    torch.manual_seed(0)
    golden_op = _get_ttnn_op(op)
    torch_input_tensor = torch.randn((h, w), dtype=torch.float32)
    input_tensor = ttnn.from_torch(
        torch_input_tensor, layout=ttnn.TILE_LAYOUT, device=device
    )

    op_jit = ttnn_jit.jit(backend="ttnn", debug=True)(op)
    output_tensor = op_jit(input_tensor)
    golden_tensor = golden_op(input_tensor)

    all_close = torch.allclose(
        output_tensor.cpu().to_torch(), golden_tensor.cpu().to_torch(), atol=1e-2
    )
    assert all_close
