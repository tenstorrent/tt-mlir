// REQUIRES: stablehlo
// RUN: ttmlir-opt -split-input-file --decompose-custom-call-tuples -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test stablehlo.tuple op is eliminated by forwarding operands directly.
module @TupleForward {
  func.func @main(%arg0: tensor<3x3xf32>, %arg1: tensor<3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>) {
    // CHECK-NOT: stablehlo.tuple
    %0 = stablehlo.tuple %arg0, %arg1 : tuple<tensor<3x3xf32>, tensor<3xf32>>
    // CHECK-NOT: stablehlo.get_tuple_element
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3x3xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3xf32>
    // CHECK: return %arg0, %arg1 : tensor<3x3xf32>, tensor<3xf32>
    return %1, %2 : tensor<3x3xf32>, tensor<3xf32>
  }
}

// -----

// Test stablehlo.tuple with partial use forwards only the used operand.
module @TuplePartialUse {
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4xi32>, %arg2: tensor<i32>) -> tensor<4xi32> {
    // CHECK-NOT: stablehlo.tuple
    %0 = stablehlo.tuple %arg0, %arg1, %arg2 : tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>
    // CHECK-NOT: stablehlo.get_tuple_element
    %1 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>) -> tensor<4xi32>
    // CHECK: return %arg1 : tensor<4xi32>
    return %1 : tensor<4xi32>
  }
}
