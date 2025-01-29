# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *


@ttkernel_compile
def test_assign_constant_int():
    # CHECK: module {
    # CHECK: func.func @[[C:.*]]
    # CHECK: %[[CONST:.*]] = arith.constant{{.*}} : i32
    # CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<1xi32>
    # CHECK: memref.store %[[CONST]], %[[ALLOCA]]{{.*}} : memref<1xi32>
    a = 1

    # CHECK: %[[CONST:.*]] = arith.constant{{.*}} : i32
    # CHECK: memref.store %[[CONST]], %[[ALLOCA]]{{.*}} : memref<1xi32>
    a = 2

    # CHECK: return[[RET:.*]]
    return a


@ttkernel_compile
def test_ifstmt():
    # CHECK: module {
    # CHECK: func.func @[[C:.*]]
    # CHECK: %[[C1:.*]] = arith.constant{{.*}} : i32
    # CHECK: %[[A_a:.*]] = memref.alloca() : memref<1xi32>
    # CHECK: memref.store %[[C1]], %[[A_a]]{{.*}} : memref<1xi32>
    a = 1
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
        b = 5
    else:
        # CHECK: %[[C2:.*]] = arith.constant{{.*}} : i32
        # CHECK: memref.store %[[C2]], %[[A_a]]{{.*}} : memref<1xi32>
        a = 20
        # CHECK: %[[C3:.*]] = arith.constant{{.*}} : i32
        # CHECK: %[[A_c:.*]] = memref.alloca() : memref<1xi32>
        # CHECK: memref.store %[[C3]], %[[A_c]]{{.*}} : memref<1xi32>
        c = 1

    # CHECK: memref.store {{.*}}, %[[A_a]]{{.*}} : memref<1xi32>
    # CHECK: %[[A_b:.*]] = memref.alloca() : memref<1xi32>
    # CHECK: memref.store {{.*}}, %[[A_b]]{{.*}} : memref<1xi32>
    # CHECK: %[[A_c:.*]] = memref.alloca() : memref<1xi32>
    # CHECK: memref.store {{.*}}, %[[A_c]]{{.*}} : memref<1xi32>
    a = 1
    b = 2
    c = 3
    return


@ttkernel_compile
def test_for():
    # CHECK: module {
    # CHECK: func.func @[[C:.*]]
    # CHECK: scf.for {{.*}} to {{.*}} step {{.*}}
    for i in range(0, 10, 1):
        a = 1

    x = 0
    y = 10
    z = 1
    for i in range(0, 10, 1):
        # CHECK: %[[X:.*]] = memref.load{{.*}}
        # CHECK: %[[Y:.*]] = memref.load{{.*}}
        # CHECK: %[[Z:.*]] = memref.load{{.*}}
        # CHECK: scf.for {{.*}} = %[[X]] to %[[Y]] step %[[Z]] {{.*}}
        for j in range(x, y, z):
            a = 1

    return


@ttkernel_compile
def test_binops():
    # CHECK: module {
    # CHECK: func.func @[[C:.*]]
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


@ttkernel_compile
def test_compare_expr():
    # CHECK: module {
    # CHECK: func.func @[[C:.*]]
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


test_assign_constant_int()
test_ifstmt()
test_for()
test_binops()
test_compare_expr()
