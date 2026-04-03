// RUN: ttmlir-opt -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @zeros_to_relu(%arg0: tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>{
    %0 = "ttir.zeros"() <{shape = array<i32: 1, 64, 112, 112>}> : () -> tensor<1x64x112x112xbf16>
    %2 = "ttir.maximum"(%arg0, %0) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK-NOT: "ttir.zeros"
    // CHECK-NOT: "ttir.maximum"
    // CHECK: %[[RESULT:.*]] = "ttir.relu"(%arg0) : (tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: return %[[RESULT]] : tensor<1x64x112x112xbf16>
    return %2: tensor<1x64x112x112xbf16>
  }

  func.func @zeros_to_relu_multiple_ops(%arg0: tensor<1x64x112x112xbf16>, %arg1: tensor<1x64x112x112xbf16>) -> (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) {
    %0 = "ttir.zeros"() <{shape = array<i32: 1, 64, 112, 112>}> : () -> tensor<1x64x112x112xbf16>
    %2 = "ttir.maximum"(%arg0, %0) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %4 = "ttir.maximum"(%arg1, %0) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK-NOT: "ttir.zeros"
    // CHECK-NOT: "ttir.maximum"
    // CHECK: %[[RESULT:.*]] = "ttir.relu"(%arg0) : (tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: %[[RESULT2:.*]] = "ttir.relu"(%arg1) : (tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: return %[[RESULT2]], %[[RESULT]] : tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>
    return %4, %2: tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>
  }

  func.func @full_to_relu(%arg0: tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>{
    %0 = "ttir.full"() <{shape = array<i32: 1, 64, 112, 112>, fill_value = 0 : i32}> : () -> tensor<1x64x112x112xbf16>
    %2 = "ttir.maximum"(%arg0, %0) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK-NOT: "ttir.full"
    // CHECK-NOT: "ttir.maximum"
    // CHECK: %[[RESULT:.*]] = "ttir.relu"(%arg0) : (tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: return %[[RESULT]] : tensor<1x64x112x112xbf16>
    return %2: tensor<1x64x112x112xbf16>
  }

  func.func @full_to_relu_multiple_ops(%arg0: tensor<1x64x112x112xbf16>, %arg1: tensor<1x64x112x112xbf16>) -> (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) {
    %0 = "ttir.full"() <{shape = array<i32: 1, 64, 112, 112>, fill_value = 0 : i32}> : () -> tensor<1x64x112x112xbf16>
    %2 = "ttir.maximum"(%arg0, %0) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %4 = "ttir.maximum"(%arg1, %0) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK-NOT: "ttir.full"
    // CHECK-NOT: "ttir.maximum"
    // CHECK: %[[RESULT:.*]] = "ttir.relu"(%arg0) : (tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: %[[RESULT2:.*]] = "ttir.relu"(%arg1) : (tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: return %[[RESULT2]], %[[RESULT]] : tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>
    return %4, %2: tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>
  }

  // Test: scalar full -> reshape pattern (produced by broadcast_in_dim conversion of scalar constants)
  func.func @scalar_full_reshape_to_relu(%arg0: tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>{
    %0 = "ttir.full"() <{fill_value = 0.000000e+00 : f32, shape = array<i32>}> : () -> tensor<bf16>
    %1 = "ttir.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<bf16>) -> tensor<1x1x1x1xbf16>
    %2 = "ttir.maximum"(%arg0, %1) : (tensor<1x64x112x112xbf16>, tensor<1x1x1x1xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK-NOT: "ttir.full"
    // CHECK-NOT: "ttir.reshape"
    // CHECK-NOT: "ttir.maximum"
    // CHECK: %[[RESULT:.*]] = "ttir.relu"(%arg0) : (tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: return %[[RESULT]] : tensor<1x64x112x112xbf16>
    return %2: tensor<1x64x112x112xbf16>
  }

  func.func @ones_to_maximum(%arg0: tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>{
    %0 = "ttir.full"() <{shape = array<i32: 1, 64, 112, 112>, fill_value = 1 : i32}> : () -> tensor<1x64x112x112xbf16>
    %2 = "ttir.maximum"(%arg0, %0) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: "ttir.full"
    // CHECK: "ttir.maximum"
    // CHECK-NOT: "ttir.relu"
    return %2: tensor<1x64x112x112xbf16>
  }
}
