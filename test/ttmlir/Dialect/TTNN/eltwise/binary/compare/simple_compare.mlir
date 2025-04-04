// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline %s | FileCheck %s
module attributes {} {
  func.func @equal(%arg0: tensor<13x31xbf16>, %arg1: tensor<13x31xbf16>) -> tensor<13x31xbf16> {
    %0 = ttir.empty() : tensor<13x31xbf16>
    %1 = "ttir.eq"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xbf16>, tensor<13x31xbf16>, tensor<13x31xbf16>) -> tensor<13x31xbf16>
    // CHECK: "ttnn.eq"
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: -> tensor<13x31xbf16
    return %1 : tensor<13x31xbf16>
  }
}

// -----

module attributes {} {
  func.func @not_equal(%arg0: tensor<13x31xbf16>, %arg1: tensor<13x31xbf16>) -> tensor<13x31xbf16> {
    %0 = ttir.empty() : tensor<13x31xbf16>
    %1 = "ttir.ne"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xbf16>, tensor<13x31xbf16>, tensor<13x31xbf16>) -> tensor<13x31xbf16>
    // CHECK: "ttnn.ne"
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: -> tensor<13x31xbf16
    return %1 : tensor<13x31xbf16>
  }
}

// -----

module attributes {} {
  func.func @greater_equal(%arg0: tensor<13x31xbf16>, %arg1: tensor<13x31xbf16>) -> tensor<13x31xbf16> {
    %0 = ttir.empty() : tensor<13x31xbf16>
    %1 = "ttir.ge"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xbf16>, tensor<13x31xbf16>, tensor<13x31xbf16>) -> tensor<13x31xbf16>
    // CHECK: "ttnn.ge"
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: -> tensor<13x31xbf16
    return %1 : tensor<13x31xbf16>
  }
}

// -----

module attributes {} {
  func.func @greater_than(%arg0: tensor<13x31xbf16>, %arg1: tensor<13x31xbf16>) -> tensor<13x31xbf16> {
    %0 = ttir.empty() : tensor<13x31xbf16>
    %1 = "ttir.gt"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xbf16>, tensor<13x31xbf16>, tensor<13x31xbf16>) -> tensor<13x31xbf16>
    // CHECK: "ttnn.gt"
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: -> tensor<13x31xbf16
    return %1 : tensor<13x31xbf16>
  }
}

// -----

module attributes {} {
  func.func @less_equal(%arg0: tensor<13x31xbf16>, %arg1: tensor<13x31xbf16>) -> tensor<13x31xbf16> {
    %0 = ttir.empty() : tensor<13x31xbf16>
    %1 = "ttir.le"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xbf16>, tensor<13x31xbf16>, tensor<13x31xbf16>) -> tensor<13x31xbf16>
    // CHECK: "ttnn.le"
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: -> tensor<13x31xbf16
    return %1 : tensor<13x31xbf16>
  }
}

// -----

module attributes {} {
  func.func @less_than(%arg0: tensor<13x31xbf16>, %arg1: tensor<13x31xbf16>) -> tensor<13x31xbf16> {
    %0 = ttir.empty() : tensor<13x31xbf16>
    %1 = "ttir.lt"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<13x31xbf16>, tensor<13x31xbf16>, tensor<13x31xbf16>) -> tensor<13x31xbf16>
    // CHECK: "ttnn.lt"
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: tensor<13x31xbf16
    // CHECK-SAME: -> tensor<13x31xbf16
    return %1 : tensor<13x31xbf16>
  }
}
