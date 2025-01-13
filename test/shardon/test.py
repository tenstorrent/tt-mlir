# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# CHECK: %[[CONST:.*]] = "arith.constant"() <{value = 10 : i32}> : () -> i32

from shardon.shardon_ast import *

@ttkernel_compile
def test_constant():
    a = 10

test_constant()



# class Tensor:
#     def __init__(self, shape, dtype):
#         self.shape = shape
#         self.dtype = dtype

# @ttkernel_compile
# def eltwise(
#     in0,
#     in1,
#     out,
#     index_maps=[
#         lambda *dn, m, n: (*dn, m, n),
#         lambda *dn, m, n: (*dn, m, n),
#         lambda *dn, m, n: (*dn, m, n),
#     ],
#     iterator_types=["parallel", "parallel", "parallel"],
#     dynamic_shapes=False,
# ):
#     t6 = Tensix(in0, in1, out)
#     for dn in range(in0.shape[-3]):
#         for m in range(in0.shape[-2]):
#             for n in range(in0.shape[-1]):
#                 in0.wait()
#                 in1.wait()
#                 out.reserve()
#                 t6.tile_regs_acquire()
#                 t6.unpack_ab(in0, 0, in1, 0)
#                 t6.add(0)
#                 t6.pack(0, out, 0)
#                 t6.tile_regs_release()
#                 in0.pop()
#                 in1.pop()
#                 out.push()


# a = Tensor((8, 128, 128), "float32")
# b = Tensor((8, 128, 128), "float32")
# out = Tensor((8, 128, 128), "float32")
# eltwise(a, b, out)