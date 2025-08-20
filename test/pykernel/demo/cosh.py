# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.ttir_ast import *

"""
TTNN composite op code:

Tensor _cosh(const Tensor& input_a, const std::optional<MemoryConfig>& output_mem_config) {
    Tensor e_pos_x = ttnn::exp(input_a, false, output_mem_config);
    Tensor e_neg_x = ttnn::exp(ttnn::neg(input_a, output_mem_config), false, output_mem_config);
    Tensor nr_term = ttnn::add(e_pos_x, e_neg_x, std::nullopt, output_mem_config);
    e_pos_x.deallocate();
    e_neg_x.deallocate();
    return ttnn::multiply(nr_term, 0.5f, std::nullopt, output_mem_config);
}

"""


@ttir_compile(verbose=True, to_flatbuffer_file="cosh.ttm")
def cosh(a):
    e_pos_x = exp(a)
    e_neg_x = exp(-a)
    nr_term = e_pos_x + e_neg_x
    return nr_term * 0.5


input_tensor = Tensor(
    [-1, 0, 1, 2], shape=[1, 1, 1, 4], data_type=ttnn.float32, layout=ttnn.TILE_LAYOUT
)

cosh(input_tensor)
