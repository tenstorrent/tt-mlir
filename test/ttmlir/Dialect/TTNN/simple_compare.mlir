// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @equal(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty"
    // CHECK-SAME: [[TENSOR:tensor<13x31xf32,]]
    %0 = tensor.empty() : tensor<13x31xf32>
    // CHECK: %[[C:.*]] = "ttnn.eq"
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: -> [[TENSOR]]
    %1 = "ttir.eq"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xf32>
    return %1 : tensor<13x31xf32>
  }
}

module attributes {} {
  func.func @not_equal(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty
    // CHECK-SAME: [[TENSOR:tensor<13x31xf32,]]
    %0 = tensor.empty() : tensor<13x31xf32>
    // CHECK: %[[C:.*]] = "ttnn.ne"
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: -> [[TENSOR]]
    %1 = "ttir.ne"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xf32>
    return %1 : tensor<13x31xf32>
  }
}

module attributes {} {
  func.func @greater_equal(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty
    // CHECK-SAME: [[TENSOR:tensor<13x31xf32,]]
    %0 = tensor.empty() : tensor<13x31xf32>
    // CHECK: %[[C:.*]] = "ttnn.ge"
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: -> [[TENSOR]]
    %1 = "ttir.ge"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xf32>
    return %1 : tensor<13x31xf32>
  }
}

module attributes {} {
  func.func @greater_than(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty
    // CHECK-SAME: [[TENSOR:tensor<13x31xf32,]]
    %0 = tensor.empty() : tensor<13x31xf32>
    // CHECK: %[[C:.*]] = "ttnn.gt"
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: -> [[TENSOR]]
    %1 = "ttir.gt"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xf32>
    return %1 : tensor<13x31xf32>
  }
}

module attributes {} {
  func.func @less_equal(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty
    // CHECK-SAME: [[TENSOR:tensor<13x31xf32,]]
    %0 = tensor.empty() : tensor<13x31xf32>
    // CHECK: %[[C:.*]] = "ttnn.le"
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: -> [[TENSOR]]
    %1 = "ttir.le"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xf32>
    return %1 : tensor<13x31xf32>
  }
}

module attributes {} {
  func.func @less_than(%arg0: tensor<13x31xf32>, %arg1: tensor<13x31xf32>) -> tensor<13x31xf32> {
    // CHECK: %[[C:.*]] = "ttnn.empty
    // CHECK-SAME: [[TENSOR:tensor<13x31xf32,]]
    %0 = tensor.empty() : tensor<13x31xf32>
    // CHECK: %[[C:.*]] = "ttnn.lt"
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: [[TENSOR]]
    // CHECK-SAME: -> [[TENSOR]]
    %1 = "ttir.lt"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xf32>, tensor<13x31xf32>, tensor<13x31xf32>) -> tensor<13x31xf32>
    return %1 : tensor<13x31xf32>
  }
}
