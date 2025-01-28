// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-repeat-folding-workaround-pass=false system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module {
  func.func @main(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK-NOT: ttnn.repeat
    // CHECK: %{{[0-9]+}} = "ttnn.add"
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i32: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.add"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %3 : tensor<1x16x32xf32>
  }
}

module {
  func.func @main(%arg0: tensor<128xf32>, %arg1: tensor<128xf32>) -> tensor<784x128xf32> {
    // CHECK: %{{[0-9]+}} = "ttnn.reshape"
    // CHECK-NOT: "ttnn.repeat"
    // CHECK: %{{[0-9]+}} = "ttnn.reshape"
    // CHECK: %{{[0-9]+}} = "ttnn.repeat"
    // CHECK-SAME: shape = [784 : i32, 1 : i32]
    // CHECK: %{{[0-9]+}} = "ttnn.add"
    %0 = tensor.empty() : tensor<1x128xf32>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %2 = tensor.empty() : tensor<784x128xf32>
    %3 = "ttir.broadcast"(%1, %2) <{broadcast_dimensions = array<i32: 784, 1>}> : (tensor<1x128xf32>, tensor<784x128xf32>) -> tensor<784x128xf32>
    %4 = tensor.empty() : tensor<1x128xf32>
    %5 = "ttir.reshape"(%arg1, %4) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %6 = tensor.empty() : tensor<784x128xf32>
    %7 = "ttir.broadcast"(%5, %6) <{broadcast_dimensions = array<i32: 784, 1>}> : (tensor<1x128xf32>, tensor<784x128xf32>) -> tensor<784x128xf32>
    %8 = tensor.empty() : tensor<784x128xf32>
    %9 = "ttir.add"(%3, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<784x128xf32>, tensor<784x128xf32>, tensor<784x128xf32>) -> tensor<784x128xf32>
    return %9 : tensor<784x128xf32>
  }
}
