# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s 2>&1 | FileCheck %s
# REQUIRES: d2m-jit

import ast

import d2m_jit as d2m  # noqa: F401
from d2m_jit._src.ast import D2MCompiler
from ttmlir.ir import Context, Location

SOURCE = """
def k():
    topk_local_sort(0, 1, 5, 0, 0, 0, True)
    topk_merge(0, 0, 32, True, True)
    topk_rebuild(0, 1, 0, 32, 5, 1, True)
"""


with Context() as ctx, Location.unknown(ctx):
    compiler = D2MCompiler("k", "compute")
    compiler.visit(ast.parse(SOURCE))
    compiler.module.operation.verify()
    print(compiler.module)


# CHECK-LABEL: func.func @k
# CHECK: d2m.topk_local_sort
# CHECK-SAME: stable_sort = true
# CHECK: d2m.topk_merge
# CHECK-SAME: sort_direction = true
# CHECK-SAME: stable_sort = true
# CHECK: d2m.topk_rebuild
# CHECK-SAME: stable_sort = true
