// RUN: ttmlir-opt -canonicalize %s | FileCheck %s
module {
  func.func @idempotence_two_in_the_row(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: "ttir.relu"
    // CHECK-NOT: "ttir.relu"
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = "ttir.relu"(%1, %2) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }

  func.func @idempotence_three_in_the_row(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: "ttir.relu"
    // CHECK-NOT: "ttir.relu"
    %0 = tensor.empty() : tensor<64x64xf32>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = "ttir.relu"(%1, %2) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    %4 = tensor.empty() : tensor<64x64xf32>
    %5 = "ttir.relu"(%2, %4) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x64xf32>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %5 : tensor<64x64xf32>
  }

  func.func @not_idempotence_diffrent_types(%arg0: tensor<64x64xf32>) -> tensor<64x64xf32> {
    // CHECK: "ttir.relu"
    // CHECK: "ttir.relu"
    %0 = tensor.empty() : tensor<64x64xbf16>
    %1 = "ttir.relu"(%arg0, %0) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x64xf32>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    %2 = tensor.empty() : tensor<64x64xf32>
    %3 = "ttir.relu"(%1, %2) <{operandSegmentSizes = array<i32: 1, 1>}> : (tensor<64x64xbf16>, tensor<64x64xf32>) -> tensor<64x64xf32>
    return %3 : tensor<64x64xf32>
  }
}
