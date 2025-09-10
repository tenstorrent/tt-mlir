# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pykernel
import ttnn


@pykernel.jit(backend="ttnn", debug=True, dump_flatbuffer=True)
def cosh(a):
    e_pos_x = exp(a)
    e_neg_x = exp(-a)
    nr_term = e_pos_x + e_neg_x
    return nr_term * 0.5


input_tensor = ttnn.Tensor(
    [-1, 0, 1, 2], shape=[1, 1, 1, 4], data_type=ttnn.float32, layout=ttnn.TILE_LAYOUT
)

cosh(input_tensor)
