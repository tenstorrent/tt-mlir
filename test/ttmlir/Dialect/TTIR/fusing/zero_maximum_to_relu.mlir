// RUN: ttmlir-opt -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t
module {
  func.func @zeros_to_relu(%arg0: tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>{
    %0 = "ttir.zeros"() <{shape = array<i32: 1, 64, 112, 112>}> : () -> tensor<1x64x112x112xbf16>
    %1 = ttir.empty() : tensor<1x64x112x112xbf16>
    %2 = "ttir.maximum"(%arg0, %0, %1) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK-NOT: "ttir.zeros"
    // CHECK-NOT: "ttir.maximum"
    // CHECK: %[[EMPTY:.*]] = ttir.empty() : tensor<1x64x112x112xbf16>
    // CHECK: %[[RESULT:.*]] = "ttir.relu"(%arg0, %[[EMPTY]]) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: return %[[RESULT]] : tensor<1x64x112x112xbf16>
    return %2: tensor<1x64x112x112xbf16>
  }

  func.func @zeros_to_relu_multiple_ops(%arg0: tensor<1x64x112x112xbf16>, %arg1: tensor<1x64x112x112xbf16>) -> (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) {
    %0 = "ttir.zeros"() <{shape = array<i32: 1, 64, 112, 112>}> : () -> tensor<1x64x112x112xbf16>
    %1 = ttir.empty() : tensor<1x64x112x112xbf16>
    %2 = "ttir.maximum"(%arg0, %0, %1) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %3 = ttir.empty() : tensor<1x64x112x112xbf16>
    %4 = "ttir.maximum"(%arg1, %0, %3) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK-NOT: "ttir.zeros"
    // CHECK-NOT: "ttir.maximum"
    // CHECK: %[[EMPTY:.*]] = ttir.empty() : tensor<1x64x112x112xbf16>
    // CHECK: %[[RESULT:.*]] = "ttir.relu"(%arg0, %[[EMPTY]]) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: %[[EMPTY2:.*]] = ttir.empty() : tensor<1x64x112x112xbf16>
    // CHECK: %[[RESULT2:.*]] = "ttir.relu"(%arg1, %[[EMPTY2]]) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: return %[[RESULT2]], %[[RESULT]] : tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>
    return %4, %2: tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>
  }

  func.func @full_to_relu(%arg0: tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>{
    %0 = "ttir.full"() <{shape = array<i32: 1, 64, 112, 112>, fill_value = 0 : i32}> : () -> tensor<1x64x112x112xbf16>
    %1 = ttir.empty() : tensor<1x64x112x112xbf16>
    %2 = "ttir.maximum"(%arg0, %0, %1) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK-NOT: "ttir.full"
    // CHECK-NOT: "ttir.maximum"
    // CHECK: %[[EMPTY:.*]] = ttir.empty() : tensor<1x64x112x112xbf16>
    // CHECK: %[[RESULT:.*]] = "ttir.relu"(%arg0, %[[EMPTY]]) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: return %[[RESULT]] : tensor<1x64x112x112xbf16>
    return %2: tensor<1x64x112x112xbf16>
  }

  func.func @full_to_relu_multiple_ops(%arg0: tensor<1x64x112x112xbf16>, %arg1: tensor<1x64x112x112xbf16>) -> (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) {
    %0 = "ttir.full"() <{shape = array<i32: 1, 64, 112, 112>, fill_value = 0 : i32}> : () -> tensor<1x64x112x112xbf16>
    %1 = ttir.empty() : tensor<1x64x112x112xbf16>
    %2 = "ttir.maximum"(%arg0, %0, %1) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    %3 = ttir.empty() : tensor<1x64x112x112xbf16>
    %4 = "ttir.maximum"(%arg1, %0, %3) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK-NOT: "ttir.full"
    // CHECK-NOT: "ttir.maximum"
    // CHECK: %[[EMPTY:.*]] = ttir.empty() : tensor<1x64x112x112xbf16>
    // CHECK: %[[RESULT:.*]] = "ttir.relu"(%arg0, %[[EMPTY]]) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: %[[EMPTY2:.*]] = ttir.empty() : tensor<1x64x112x112xbf16>
    // CHECK: %[[RESULT2:.*]] = "ttir.relu"(%arg1, %[[EMPTY2]]) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: return %[[RESULT2]], %[[RESULT]] : tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>
    return %4, %2: tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>
  }

  func.func @ones_to_maximum(%arg0: tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>{
    %0 = "ttir.full"() <{shape = array<i32: 1, 64, 112, 112>, fill_value = 1 : i32}> : () -> tensor<1x64x112x112xbf16>
    %1 = ttir.empty() : tensor<1x64x112x112xbf16>
    %2 = "ttir.maximum"(%arg0, %0, %1) : (tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>, tensor<1x64x112x112xbf16>) -> tensor<1x64x112x112xbf16>
    // CHECK: "ttir.full"
    // CHECK: "ttir.maximum"
    // CHECK-NOT: "ttir.relu"
    return %2: tensor<1x64x112x112xbf16>
  }
}
