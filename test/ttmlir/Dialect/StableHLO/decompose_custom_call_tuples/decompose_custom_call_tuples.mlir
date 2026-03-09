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

// -----

// Test basic QR decomposition custom_call with tuple return.
module @QrDecompose {
  func.func @main(%arg0: tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>) {
    // CHECK: %[[RESULTS:.*]]:2 = stablehlo.custom_call @Qr(%arg0) : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>)
    %0 = stablehlo.custom_call @Qr(%arg0) : (tensor<3x3xf32>) -> tuple<tensor<3x3xf32>, tensor<3xf32>>
    // CHECK-NOT: stablehlo.get_tuple_element
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3x3xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3xf32>
    // CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1 : tensor<3x3xf32>, tensor<3xf32>
    return %1, %2 : tensor<3x3xf32>, tensor<3xf32>
  }
}

// -----

// Test LU decomposition custom_call with three tuple elements.
module @LuDecompose {
  func.func @main(%arg0: tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xi32>, tensor<i32>) {
    // CHECK: %[[RESULTS:.*]]:3 = stablehlo.custom_call @Lu(%arg0) : (tensor<4x4xf32>) -> (tensor<4x4xf32>, tensor<4xi32>, tensor<i32>)
    %0 = stablehlo.custom_call @Lu(%arg0) : (tensor<4x4xf32>) -> tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>
    // CHECK-NOT: stablehlo.get_tuple_element
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>) -> tensor<4x4xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>) -> tensor<4xi32>
    %3 = stablehlo.get_tuple_element %0[2] : (tuple<tensor<4x4xf32>, tensor<4xi32>, tensor<i32>>) -> tensor<i32>
    // CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1, %[[RESULTS]]#2
    return %1, %2, %3 : tensor<4x4xf32>, tensor<4xi32>, tensor<i32>
  }
}

// -----

// Test that non-tuple custom_call ops are left unchanged.
module @NonTupleUnchanged {
  func.func @main(%arg0: tensor<3x3xf32>) -> tensor<3x3xf32> {
    // CHECK: %[[RESULT:.*]] = stablehlo.custom_call @SomeOp(%arg0) : (tensor<3x3xf32>) -> tensor<3x3xf32>
    %0 = stablehlo.custom_call @SomeOp(%arg0) : (tensor<3x3xf32>) -> tensor<3x3xf32>
    // CHECK: return %[[RESULT]] : tensor<3x3xf32>
    return %0 : tensor<3x3xf32>
  }
}

// -----

// Test custom_call with multiple inputs and tuple return.
module @MultipleInputs {
  func.func @main(%arg0: tensor<8x8xf32>, %arg1: tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8xf32>) {
    // CHECK: %[[RESULTS:.*]]:2 = stablehlo.custom_call @Eigh(%arg0, %arg1) : (tensor<8x8xf32>, tensor<8x8xf32>) -> (tensor<8x8xf32>, tensor<8xf32>)
    %0 = stablehlo.custom_call @Eigh(%arg0, %arg1) : (tensor<8x8xf32>, tensor<8x8xf32>) -> tuple<tensor<8x8xf32>, tensor<8xf32>>
    // CHECK-NOT: stablehlo.get_tuple_element
    %1 = stablehlo.get_tuple_element %0[0] : (tuple<tensor<8x8xf32>, tensor<8xf32>>) -> tensor<8x8xf32>
    %2 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<8x8xf32>, tensor<8xf32>>) -> tensor<8xf32>
    // CHECK: return %[[RESULTS]]#0, %[[RESULTS]]#1
    return %1, %2 : tensor<8x8xf32>, tensor<8xf32>
  }
}

// -----

// Test that only some results of a tuple are used.
module @PartialUse {
  func.func @main(%arg0: tensor<3x3xf32>) -> tensor<3xf32> {
    // CHECK: %[[RESULTS:.*]]:2 = stablehlo.custom_call @Qr(%arg0) : (tensor<3x3xf32>) -> (tensor<3x3xf32>, tensor<3xf32>)
    %0 = stablehlo.custom_call @Qr(%arg0) : (tensor<3x3xf32>) -> tuple<tensor<3x3xf32>, tensor<3xf32>>
    // CHECK-NOT: stablehlo.get_tuple_element
    %1 = stablehlo.get_tuple_element %0[1] : (tuple<tensor<3x3xf32>, tensor<3xf32>>) -> tensor<3xf32>
    // CHECK: return %[[RESULTS]]#1 : tensor<3xf32>
    return %1 : tensor<3xf32>
  }
}
