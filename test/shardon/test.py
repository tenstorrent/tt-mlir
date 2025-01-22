# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s

from shardon.shardon_ast import *

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
    #CHECK: scf.if{{.*}}
    if(1):
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

test_assign_constant_int()
test_ifstmt()