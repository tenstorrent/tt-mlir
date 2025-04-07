# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel.pykernel_ast import *


@ttkernel_compile(optimize=False)
def test_for_iter_args():
    # CHECK: module {
    # CHECK: func.func @

    # TEST: Simple for loop with a variable modified inside the loop
    # This should automatically use iter_args
    a = 0
    # CHECK: %[[INIT:.*]] = arith.constant 0 : i32
    # CHECK: %[[RESULT:.*]] = scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[ITER_A:.*]] = %[[INIT]])
    for i in range(0, 10, 1):
        # CHECK: %[[NEW_A:.*]] = arith.addi %[[ITER_A]], {{.*}} : i32
        a = a + 1
        # CHECK: scf.yield %[[NEW_A]] : i32

    # CHECK: return
    return


@ttkernel_compile(optimize=False)
def test_for_multiple_iter_args():
    # CHECK: module {
    # CHECK: func.func @

    # TEST: For loop with multiple variables modified inside the loop
    # All should be automatically turned into iter_args
    a = 0
    b = 10
    c = 5

    # CHECK: %[[A_INIT:.*]] = arith.constant 0 : i32
    # CHECK: %[[B_INIT:.*]] = arith.constant 10 : i32
    # CHECK: %[[C_INIT:.*]] = arith.constant 5 : i32
    # CHECK: %[[RESULT:.*]]:3 = scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[ITER_A:.*]] = %[[A_INIT]], %[[ITER_B:.*]] = %[[B_INIT]], %[[ITER_C:.*]] = %[[C_INIT]])
    for i in range(0, 10, 1):
        # CHECK: %[[NEW_A:.*]] = arith.addi %[[ITER_A]], {{.*}} : i32
        a = a + 1
        # CHECK: %[[NEW_B:.*]] = arith.subi %[[ITER_B]], {{.*}} : i32
        b = b - 1
        # CHECK: %[[NEW_C:.*]] = arith.muli %[[ITER_C]], {{.*}} : i32
        c = c * 2
        # CHECK: scf.yield %[[NEW_A]], %[[NEW_B]], %[[NEW_C]] : i32, i32, i32

    # CHECK: return
    return


@ttkernel_compile(optimize=False)
def test_nested_for_with_iter_args():
    # CHECK: module {
    # CHECK: func.func @

    # TEST: Nested for loops with variables modified at different levels
    sum_val = 0

    # CHECK: %[[SUM_INIT:.*]] = arith.constant 0 : i32
    # CHECK: %[[RESULT:.*]] = scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[ITER_SUM:.*]] = %[[SUM_INIT]])
    for i in range(0, 5, 1):
        # CHECK: %[[INNER_RESULT:.*]] = scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args(%[[INNER_ITER_SUM:.*]] = %[[ITER_SUM]])
        for j in range(0, 5, 1):
            # CHECK: %[[NEW_SUM:.*]] = arith.addi %[[INNER_ITER_SUM]], {{.*}} : i32
            sum_val = sum_val + 1
            # CHECK: scf.yield %[[NEW_SUM]] : i32
        # CHECK: scf.yield %[[INNER_RESULT]] : i32

    # CHECK: return
    return


@ttkernel_compile(optimize=False)
def test_mixed_scope_variables():
    # CHECK: module {
    # CHECK: func.func @

    # TEST: For loop with a mix of variables defined outside and inside the loop
    # Only the variables defined outside and modified inside should become iter_args
    outer_var1 = 10
    outer_var2 = 5

    # CHECK: %[[OUTER1_INIT:.*]] = arith.constant 10 : i32
    # CHECK: %[[OUTER2_INIT:.*]] = arith.constant 5 : i32
    # CHECK: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args
    for i in range(0, 10, 1):
        # Variables defined inside the loop should not become iter_args
        inner_var1 = 1

        # Modifying both inner and outer variables
        inner_var1 = inner_var1 + 1

        # Modify outer variables
        outer_var1 = outer_var1 - 1
        outer_var2 = outer_var2 + 1

    # CHECK: return
    return


@ttkernel_compile(optimize=False)
def test_conditional_modification():
    # CHECK: module {
    # CHECK: func.func @

    # TEST: For loop with variables that are conditionally modified
    # These should still be detected as iter_args
    counter = 0

    # CHECK: %[[COUNTER_INIT:.*]] = arith.constant 0 : i32
    # CHECK: scf.for {{.*}} = {{.*}} to {{.*}} step {{.*}} iter_args
    for i in range(0, 10, 1):
        # Simple conditional modification
        if i == 5:
            counter = counter + 1

    # CHECK: return
    return


# Run the tests
test_for_iter_args()
test_for_multiple_iter_args()
test_nested_for_with_iter_args()
test_mixed_scope_variables()
test_conditional_modification()
