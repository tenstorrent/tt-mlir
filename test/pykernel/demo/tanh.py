# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pykernel.ttir_ast import *
import numpy as np

"""
This already exists as a TTIRNamedOp..

tanh = (e^x - e^-x) / (e^x + e^-x)

"""

"""
2025-08-07 18:54:49,706 - DEBUG - tensor([[[[ 1.5410, -0.2934, -2.1788,  0.5684]]]])

2025-08-07 18:54:49,706 - DEBUG - output tensors for program=0
2025-08-07 18:54:49,707 - DEBUG - tensor([[[[ 0.9116, -0.2734, -0.9731,  0.5112]]]])
"""


@ttir_compile(verbose=True, to_flatbuffer_file="tanh_composite.ttm")
def tanh_composite(a):
    e_pos_x = exp(a)
    e_neg_x = exp(-a)
    return (e_pos_x - e_neg_x) / (e_pos_x + e_neg_x)


"""
2025-08-07 18:54:09,663 - DEBUG - input tensors for program=0
2025-08-07 18:54:09,665 - DEBUG - tensor([[[[ 1.5410, -0.2934, -2.1788,  0.5684]]]])

2025-08-07 18:54:09,665 - DEBUG - output tensors for program=0
2025-08-07 18:54:09,666 - DEBUG - tensor([[[[ 0.9116, -0.2734, -0.9731,  0.5112]]]])
"""


@ttir_compile(verbose=True, to_flatbuffer_file="tanh.ttm")
def tanh(a):
    return tanh(a)


dummy_values = [1.0, 2.0, 3.0, 4.0]
a = Tensor(
    dummy_values, shape=[1, 1, 1, 4], data_type=ttnn.float32, layout=ttnn.TILE_LAYOUT
)
k = Tensor(
    dummy_values, shape=[1, 1, 1, 4], data_type=ttnn.float32, layout=ttnn.TILE_LAYOUT
)

# tanh(a)
tanh_composite(a)
