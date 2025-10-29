# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python %s | FileCheck %s
# REQUIRES: pykernel

from pykernel import ttkernel_compile, ttkernel_noc_compile

# NOTE: The FileCheck directives in this file use CHECK-LABEL to find function definitions
# regardless of their order in the output. This makes the tests more robust against
# variations in the generated MLIR code.


@ttkernel_compile()
def test_assign():
    # CHECK: module {
    # CHECK: func.func @

    # TEST: SSA assignment
    # CHECK: arith.constant 1 : i32
    # CHECK: arith.constant 2 : i32
    a = 1
    a = 2

    # TEST: AnnAssign with memref
    # CHECK: %[[CONST:.*]] = arith.constant{{.*}} : i32
    # CHECK: %[[ALLOCA:.*]] = memref.alloca() : memref<1xi32>
    # CHECK: memref.store %[[CONST]], %[[ALLOCA]]{{.*}} : memref<1xi32>
    b: int = 1
    # CHECK: %[[CONST:.*]] = arith.constant{{.*}} : i32
    # CHECK: memref.store %[[CONST]], %[[ALLOCA]]{{.*}} : memref<1xi32>
    b = 2

    # TEST: AugAssign with memref
    # CHECK: memref.load %[[ALLOCA]]
    # CHECK: arith.addi
    # CHECK: memref.store {{.*}} %[[ALLOCA]]
    b += 2

    # CHECK: memref.load %[[ALLOCA]]
    # CHECK: arith.subi
    # CHECK: memref.store {{.*}} %[[ALLOCA]]
    b -= 2

    # CHECK: memref.load %[[ALLOCA]]
    # CHECK: arith.muli
    # CHECK: memref.store {{.*}} %[[ALLOCA]]
    b *= 2

    return


@ttkernel_compile(optimize=False)
def test_ifstmt():
    # CHECK: module {
    # CHECK: func.func @

    # TEST: if stmt scope stacks with memref and SSA
    # CHECK: %[[C1:.*]] = arith.constant
    # CHECK: %[[A_a:.*]] = memref.alloca() : memref<1xi32>
    # CHECK: memref.store %[[C1]], %[[A_a]]{{.*}} : memref<1xi32>
    a: int = 1

    # CHECK: %[[L_a:.*]] = memref.load %[[A_a]]{{.*}} : memref<1xi32>
    # CHECK: %[[COND:.*]] = arith.cmpi eq, %[[L_a]]
    # CHECK: scf.if %[[COND]]
    if a == 1:
        # CHECK: %[[C2:.*]] = arith.constant
        # CHECK: memref.store %[[C2]], %[[A_a]]{{.*}} : memref<1xi32>
        a = 10
        # CHECK: %[[C3:.*]] = arith.constant
        # CHECK: %[[A_b:.*]] = memref.alloca() : memref<1xi32>
        # CHECK: memref.store %[[C3]], %[[A_b]]{{.*}} : memref<1xi32>
        b: int = 5
    else:
        # CHECK: %[[C2:.*]] = arith.constant
        # CHECK: memref.store %[[C2]], %[[A_a]]{{.*}} : memref<1xi32>
        a = 20
        # CHECK: %[[C3:.*]] = arith.constant
        # CHECK: %[[A_c:.*]] = memref.alloca() : memref<1xi32>
        # CHECK: memref.store %[[C3]], %[[A_c]]{{.*}} : memref<1xi32>
        c: int = 1

    # CHECK: memref.store {{.*}}, %[[A_a]]{{.*}} : memref<1xi32>
    # CHECK: arith.constant
    # CHECK: arith.constant
    a = 1
    b = 2
    c = 3

    # CHECK: %[[COND2:.*]] = arith.cmpi ne
    # CHECK: scf.if %[[COND2]]
    if a:
        # CHECK: memref.store
        a = 2

    # CHECK: arith.cmpi sgt
    # CHECK: %[[COND3:.*]] = arith.andi
    # CHECK: scf.if %[[COND3]]
    if True and a > 10:
        # CHECK: memref.store
        a = 2
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
        # CHECK: %[[X:.*]] = memref.load
        # CHECK: %[[Y:.*]] = memref.load
        # CHECK: %[[Z:.*]] = memref.load
        # CHECK: scf.for {{.*}} = %[[X]] to %[[Y]] step %[[Z]]
        for j in range(x, y, z):
            # CHECK: arith.addi
            # CHECK: memref.store
            d = d + 1

    return


@ttkernel_compile()
def test_binops():
    # CHECK: module {
    # CHECK: func.func @
    a = 1
    b = 1
    # CHECK: arith.addi
    a + b
    # CHECK: arith.subi
    a - b
    # CHECK: arith.muli
    a * b
    # CHECK: arith.addi
    # CHECK: arith.subi
    a + b - a
    # CHECK: arith.addi
    # CHECK: arith.muli
    # CHECK: arith.subi
    a + b - a * b

    # CHECK: arith.floordivsi
    a // b

    # CHECK: arith.remsi
    a % b

    # CHECK: arith.shrsi
    a >> b

    # CHECK: arith.shli
    a << b

    # CHECK: arith.andi
    a & b

    # CHECK: arith.ori
    a | b

    # CHECK: arith.xori
    a ^ b

    return


@ttkernel_compile(optimize=False)
def test_compare_expr():
    # CHECK: module {
    # CHECK: func.func @
    a = 1
    b = 2
    # CHECK: arith.cmpi eq
    a == b
    # CHECK: arith.cmpi ne
    a != b
    # CHECK: arith.cmpi slt
    a < b
    # CHECK: arith.cmpi sle
    a <= b
    # CHECK: arith.cmpi sgt
    a > b
    # CHECK: arith.cmpi sge
    a >= b
    return


@ttkernel_compile(optimize=False)
def test_bool_ops():
    # CHECK: module {
    # CHECK: func.func @
    a = 1
    b = 2
    # CHECK: arith.cmpi sge
    # CHECK: arith.cmpi sle
    # CHECK: arith.andi
    a >= b and b <= a

    # CHECK: arith.cmpi sgt
    # CHECK: arith.cmpi ne
    # CHECK: arith.andi
    # CHECK: arith.cmpi ne
    # CHECK: arith.ori
    a or b and a > b

    # CHECK: arith.cmpi slt
    # CHECK: arith.andi
    # CHECK: arith.ori
    # CHECK: arith.ori
    # CHECK: arith.andi
    # CHECK: arith.ori
    False and (a < b) or False and (True or False or False)
    return


@ttkernel_compile()
def test_unary_ops():
    # CHECK: module {
    # CHECK: func.func @

    a = 1

    # CHECK: emitc.logical_not
    not a

    # CHECK: emitc.bitwise_not
    ~a

    # CHECK: emitc.unary_minus
    -a

    # CHECK: emitc.unary_plus
    +a

    return


@ttkernel_compile(optimize=False)
def test_array_assign():
    # CHECK-LABEL: func.func @test_array_assign

    # TEST: Array declaration and initialization with size
    # CHECK: memref<5xi32>
    a: [int, 5] = 0

    # TEST: Array initialization with list of values
    # CHECK: memref<3xi32>
    # CHECK: memref.store {{.*}} : memref<3xi32>
    b = [1, 2, 3]

    return


@ttkernel_compile(optimize=False)
def test_array_access():
    # CHECK-LABEL: func.func @test_array_access

    # TEST: Array declaration and element access
    # CHECK: memref<5xi32>
    a = [0, 1, 2, 3, 4]

    # TEST: Array element assignment with constant index
    # CHECK: arith.constant 42 : i32
    # CHECK: memref.store {{.*}} : memref<5xi32>
    a[2] = 42

    # TEST: Array element access with constant index
    # CHECK: memref.load {{.*}} : memref<5xi32>
    b = a[2]

    # TEST: Array element access with variable index
    # CHECK: arith.constant 3 : i32
    i = 3
    # CHECK: arith.index_cast {{.*}} : i32 to index
    # CHECK: memref.load {{.*}} : memref<5xi32>
    c = a[i]

    # TEST: Array element assignment with expression index
    # CHECK: arith.addi
    # CHECK: arith.constant 99 : i32
    a[i + 1] = 99

    return


@ttkernel_noc_compile(optimize=False)
def test_array_iteration():
    # CHECK-LABEL: func.func @test_array_iteration

    # TEST: Array declaration
    # CHECK: memref<5xi32>
    a = [0, 0, 0, 0, 0]

    # Initialize array with values
    # CHECK: arith.constant 1 : i32
    a[0] = 1
    # CHECK: arith.constant 2 : i32
    a[1] = 2
    # CHECK: arith.constant 3 : i32
    a[2] = 3
    # CHECK: arith.constant 4 : i32
    a[3] = 4
    # CHECK: arith.constant 5 : i32
    a[4] = 5

    # TEST: Sum array elements using loop with index
    # CHECK: memref<1xi32>
    # CHECK: arith.constant 0 : i32
    sum: int = 0

    # CHECK: scf.for
    for i in range(0, 5, 1):
        # CHECK: arith.addi
        sum += a[i]

    # TEST: Regular for loop with range
    # CHECK: memref<1xi32>
    # CHECK: arith.constant 1 : i32
    prod: int = 1

    # CHECK: scf.for
    for i in range(0, 5, 1):
        # CHECK: arith.muli
        prod = prod * a[i]

    return


# Additional test for array operations
@ttkernel_compile(optimize=False)
def test_array_additional():
    # CHECK-LABEL: func.func @test_array_additional

    # TEST: Create array with constants
    # CHECK: memref<3xi32>
    a = [2, 4, 6]

    # TEST: Access with variable index
    idx = 1
    # CHECK: arith.index_cast {{.*}} : i32 to index
    # CHECK: memref.load {{.*}} : memref<3xi32>
    b = a[idx]  # Should be 4

    # TEST: Modify array element
    # CHECK: arith.constant 10 : i32
    # CHECK: memref.store {{.*}} : memref<3xi32>
    a[idx] = 10

    return


# Test for multidimensional arrays
@ttkernel_compile(optimize=False)
def test_multidim_arrays():
    # CHECK-LABEL: func.func @test_multidim_arrays

    # TEST: 2D array initialization
    # CHECK: memref<2x3xi32>
    matrix = [[1, 2, 3], [4, 5, 6]]

    return


# Test for array creation with expressions as values
@ttkernel_compile(optimize=False)
def test_array_with_expressions():
    # CHECK-LABEL: func.func @test_array_with_expressions

    # TEST: Define variables for expressions
    # CHECK: arith.constant 5 : i32
    a = 5
    # CHECK: arith.constant 10 : i32
    b = 10

    # TEST: Compute expressions before array creation
    # CHECK: arith.addi
    # CHECK: arith.subi
    # CHECK: arith.muli
    # CHECK: arith.constant 42 : i32

    # TEST: Create array with expressions
    # CHECK: memref<4xi32>
    # CHECK: memref.store {{.*}} : memref<4xi32>
    arr = [a + b, a - b, a * b, 42]

    # TEST: Access array elements
    # CHECK: memref.load {{.*}} : memref<4xi32>
    c = arr[0]  # Should be 15

    return


@ttkernel_compile(optimize=False)
def test_attributes():
    # CHECK-LABEL: func.func @test_attributes
    # TEST: Define variables
    # CHECK: %[[BANK_ID:.*]] = arith.constant
    bank_id = 1
    args = TensorAccessorArgs(2, 0)

    # TEST: Check member function is called correctly
    # CHECK: %[[TA:.*]] = ttkernel.TensorAccessor({{.*}})
    ta = TensorAccessor(args, 0, 1024)

    # CHECK: ttkernel.tensor_accessor_get_noc_addr(%[[TA]], %[[BANK_ID]], {{.*}})
    noc_addr = ta.get_noc_addr(bank_id, 0)
    return


@ttkernel_compile(optimize=False)
def test_print():
    # CHECK-LABEL: func.func @test_print

    # TEST: Define variables
    # CHECK: %[[X:.*]] = arith.constant
    # CHECK: %[[Y:.*]] = arith.constant
    x = 1
    y = 2

    # TEST: dprint with just var
    # CHECK: ttkernel.dprint("{}\\n", %[[X]])
    print(x)

    # TEST: dprint with no args
    # CHECK: ttkernel.dprint("Hello world\\n")
    print("Hello world")

    # TEST: dprint with args
    # CHECK: ttkernel.dprint("Hello world {} {} goodbye.\\n", %[[X]], %[[Y]])
    print("Hello world", x, y, "goodbye.")

    # TEST: dprint with string and format args
    # CHECK: ttkernel.dprint("Hello {} world Goodbye {} world\\n", %[[X]], %[[Y]])
    print("Hello {} world".format(x), "Goodbye {} world".format(y))
    return


test_assign()
test_ifstmt()
test_for()
test_binops()
test_compare_expr()
test_bool_ops()
test_unary_ops()
test_array_assign()
test_array_access()
test_array_iteration()
test_array_additional()
test_multidim_arrays()
test_array_with_expressions()
test_attributes()
test_print()
