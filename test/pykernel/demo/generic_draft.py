# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.ast import *


@ttkernel_compile(
    kernel_type="generic",
    grid=[1, 1],
    index_maps=[],
)
def eltwise_add(lhs, rhs, out):
    out = lhs + rhs


@ttkernel_compile(
    kernel_type="generic",
    grid=[1, 1],  # can this be inferred from tensor?
    block_factors=[1, 1],  # this is just defaulted, are we even using this
    index_maps=[],
    iterator_types=[],  # can we abstract this away? if rank < 2, we can just use parallel?
    threads=[],  # abstract this away?
    num_regions=1,  # default to 1, but will need +1 for each fused op i think
)
def eltwise_add_block(lhs_block, rhs_block, out_block):
    for y in lhs_block.shape[0]:
        for x in lhs_block.shape[1]:
            out_block[y, x] = lhs_block[y, x] + rhs_block[y, x]


"""
"""


def cosh(input, output):
    e_pos_x = exp(input)  # tile_exp
    neg_x = neg(input)  # tile_negative
    e_neg_x = exp(neg_x)  # tile_exp
    nr_term = e_pos_x + e_neg_x
    output = nr_term / 2  # tile_div
    return output


"""
"""


def sinh(input, output):
    e_pos_x = exp(input)  # tile_exp
    neg_x = neg(input)  # tile_negative
    e_neg_x = exp(neg_x)  # tile_exp
    nr_term = e_pos_x - e_neg_x
    output = nr_term / 2  # tile_div
    return output


def swish_or_silu(input, output):
    # x / (1 + exp(-x))
    neg_x = neg(input)  # tile_negative
    exp_neg_x = exp(neg_x)  # tile_exp
    output = input / (1 + exp_neg_x)  # tile_add -> tile_div
    return output


def cbrt(input, output):
    # constexpr float scale = (float)(1.0 / 3.0);
    # Tensor t_ln_input =
    #     ttnn::log(ttnn::abs(input_tensor, output_mem_config), output_mem_config);  // negative log is not useful here
    # Tensor t1 = ttnn::multiply(t_ln_input, scale, std::nullopt, output_mem_config);
    # t_ln_input.deallocate();
    # Tensor t2 = ttnn::exp(t1, false, output_mem_config);
    # t1.deallocate();
    # Tensor t3 = ttnn::multiply(t2, ttnn::sign(input_tensor, output_mem_config), std::nullopt, output_mem_config);
    # return t3;

    abs_input = abs(input)
    t_ln_input = log(abs_input)
    t1 = t_ln_input * 1 / 3
    t2 = exp(t1)
    t3 = t2 * sign(input)
    return t3


# // polygamma support for the range of input(1, 10) and n(1, 10)
# Tensor _polygamma(const Tensor& input_a, int32_t k, const std::optional<MemoryConfig>& output_mem_config) {
#     float k_der = 1.0f + k;
#     float fact_val = std::tgamma(k_der);
#     float pos_neg = 1.0f;
#     if (k == 2 || k == 4 || k == 6 || k == 8 || k == 10) {
#         pos_neg = -1.0f;
#     }
#     Tensor temp(input_a);
#     {
#         Tensor z1 = ttnn::reciprocal(ttnn::power(input_a, k_der, output_mem_config), output_mem_config);
#         temp = z1;
#         for (int idx = 1; idx < 11; idx++) {
#             z1 = ttnn::reciprocal(
#                 ttnn::power(ttnn::add(input_a, idx, std::nullopt, output_mem_config), k_der, output_mem_config),
#                 output_mem_config);
#             temp = ttnn::add(temp, z1, std::nullopt, output_mem_config);
#         }
#     }
#     fact_val *= pos_neg;
#     return ttnn::multiply(temp, fact_val, std::nullopt, output_mem_config);
# }
