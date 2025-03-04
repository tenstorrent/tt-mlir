# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *


@ttkernel_compile()
def test_assign():
    # CHECK: module {
    # CHECK: func.func @

    # TEST: SSA assignment
    # CHECK: {{.*}} = arith.constant 1 : i32
    # CHECK: {{.*}} = arith.constant 2 : i32
    a = 1
    a = 2

    # TEST: AugAssign with memref
    # CHECK: %[[CONST:.*]] = arith.constant{{.*}} : i32
    # CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<1xi32>
    # CHECK: memref.store %[[CONST]], %[[ALLOCA]]{{.*}} : memref<1xi32>
    b: int = 1
    # CHECK: %[[CONST:.*]] = arith.constant{{.*}} : i32
    # CHECK: memref.store %[[CONST]], %[[ALLOCA]]{{.*}} : memref<1xi32>
    b = 2

    return


@ttkernel_compile()
def test_ifstmt():
    # CHECK: module {
    # CHECK: func.func @

    # TEST: if stmt scope stacks with memref and SSA
    # CHECK: %[[C1:.*]] = arith.constant{{.*}} : i32
    # CHECK: %[[A_a:.*]] = memref.alloca() : memref<1xi32>
    # CHECK: memref.store %[[C1]], %[[A_a]]{{.*}} : memref<1xi32>
    a: int = 1

    # CHECK: %[[L_a:.*]] = memref.load %[[A_a]]{{.*}} : memref<1xi32>
    # CHECK: %[[COND:.*]] = arith.cmpi eq, %[[L_a]]{{.*}}
    # CHECK: scf.if %[[COND]]{{.*}}
    if a == 1:
        # CHECK: %[[C2:.*]] = arith.constant{{.*}} : i32
        # CHECK: memref.store %[[C2]], %[[A_a]]{{.*}} : memref<1xi32>
        a = 10
        # CHECK: %[[C3:.*]] = arith.constant{{.*}} : i32
        # CHECK: %[[A_b:.*]] = memref.alloca() : memref<1xi32>
        # CHECK: memref.store %[[C3]], %[[A_b]]{{.*}} : memref<1xi32>
        b: int = 5
    else:
        # CHECK: %[[C2:.*]] = arith.constant{{.*}} : i32
        # CHECK: memref.store %[[C2]], %[[A_a]]{{.*}} : memref<1xi32>
        a = 20
        # CHECK: %[[C3:.*]] = arith.constant{{.*}} : i32
        # CHECK: %[[A_c:.*]] = memref.alloca() : memref<1xi32>
        # CHECK: memref.store %[[C3]], %[[A_c]]{{.*}} : memref<1xi32>
        c: int = 1

    # CHECK: memref.store {{.*}}, %[[A_a]]{{.*}} : memref<1xi32>
    # CHECK: {{.*}} = arith.constant{{.*}}
    # CHECK: {{.*}} = arith.constant{{.*}}
    a = 1
    b = 2
    c = 3
    return


@ttkernel_compile()
def test_for():
    # CHECK: module {
    # CHECK: func.func @

    # TEST: simple for loop
    # CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
    for i in range(0, 10, 1):
        a = 1

    # TEST: nested for loop with vars and accumulation with memref
    a = 0
    b = 10
    c = 1
    x: int = 0
    y: int = 10
    z: int = 1
    d: int = 1
    for i in range(a, b, c):
        # CHECK: %[[X:.*]] = memref.load{{.*}}
        # CHECK: %[[Y:.*]] = memref.load{{.*}}
        # CHECK: %[[Z:.*]] = memref.load{{.*}}
        # CHECK: scf.for {{.*}} = %[[X]] to %[[Y]] step %[[Z]] {{.*}}
        for j in range(x, y, z):
            # CHECK: {{.*}} = arith.addi{{.*}}
            # CHECK: memref.store{{.*}}
            d = d + 1

    return


@ttkernel_compile()
def test_binops():
    # CHECK: module {
    # CHECK: func.func @
    a = 1
    b = 1
    # CHECK: %{{.*}} = arith.addi{{.*}}
    a + b
    # CHECK: %{{.*}} = arith.subi{{.*}}
    a - b
    # CHECK: %{{.*}} = arith.muli{{.*}}
    a * b
    # CHECK: %{{.*}} = arith.addi{{.*}}
    # CHECK: %{{.*}} = arith.subi{{.*}}
    a + b - a
    # CHECK: %{{.*}} = arith.addi{{.*}}
    # CHECK: %{{.*}} = arith.muli{{.*}}
    # CHECK: %{{.*}} = arith.subi{{.*}}
    a + b - a * b
    return


@ttkernel_compile()
def test_compare_expr():
    # CHECK: module {
    # CHECK: func.func @
    a = 1
    b = 2
    # CHECK: %{{.*}} = arith.cmpi eq{{.*}}
    a == b
    # CHECK: %{{.*}} = arith.cmpi ne{{.*}}
    a != b
    # CHECK: %{{.*}} = arith.cmpi slt{{.*}}
    a < b
    # CHECK: %{{.*}} = arith.cmpi sle{{.*}}
    a <= b
    # CHECK: %{{.*}} = arith.cmpi sgt{{.*}}
    a > b
    # CHECK: %{{.*}} = arith.cmpi sge{{.*}}
    a >= b
    return


test_assign()
test_ifstmt()
test_for()
test_binops()
test_compare_expr()
