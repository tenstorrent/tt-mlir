// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-repeat-folding-workaround-pass=false" %s | FileCheck %s
module {
  func.func @main(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK-NOT: ttnn.repeat
    // CHECK: %{{[0-9]+}} = "ttnn.add"
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
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
    // CHECK-SAME: repeat_dims = #ttnn.shape<784x1>
    // CHECK: %{{[0-9]+}} = "ttnn.add"
    %0 = tensor.empty() : tensor<1x128xf32>
    %1 = "ttir.reshape"(%arg0, %0) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %2 = tensor.empty() : tensor<784x128xf32>
    %3 = "ttir.broadcast"(%1, %2) <{broadcast_dimensions = array<i64: 784, 1>}> : (tensor<1x128xf32>, tensor<784x128xf32>) -> tensor<784x128xf32>
    %4 = tensor.empty() : tensor<1x128xf32>
    %5 = "ttir.reshape"(%arg1, %4) <{shape = [1 : i32, 128 : i32]}> : (tensor<128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    %6 = tensor.empty() : tensor<784x128xf32>
    %7 = "ttir.broadcast"(%5, %6) <{broadcast_dimensions = array<i64: 784, 1>}> : (tensor<1x128xf32>, tensor<784x128xf32>) -> tensor<784x128xf32>
    %8 = tensor.empty() : tensor<784x128xf32>
    %9 = "ttir.add"(%3, %7, %8) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<784x128xf32>, tensor<784x128xf32>, tensor<784x128xf32>) -> tensor<784x128xf32>
    return %9 : tensor<784x128xf32>
  }
}

module {   func.func @main(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.repeat"
    // CHECK-SAME: repeat_dims = #ttnn.shape<1x16x1>
    // CHECK: %{{[0-9]+}} = "ttnn.multiply"(%{{[0-9]+}}, %{{[0-9]+}}, %{{[0-9]+}})
    // CHECK: %{{[0-9]+}} = "ttnn.bitwise_and"([[VAL0]], %{{[0-9]+}}, %{{[0-9]+}})
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.multiply"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %4 = tensor.empty() : tensor<1x16x32xf32>
    %5 = "ttir.bitwise_and"(%1, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %5 : tensor<1x16x32xf32>
  }
}

module {
  func.func @main(%arg0: tensor<1x16x32xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
    // CHECK-NOT: ttnn.repeat
    // CHECK: [[VAL0:%[0-9]+]] = "ttnn.add"
    // CHECK: %{{[0-9]+}} = "ttnn.add"(%{{[0-9]+}}, [[VAL0]], %{{[0-9]+}})
    %0 = tensor.empty() : tensor<1x16x32xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %2 = tensor.empty() : tensor<1x16x32xf32>
    %3 = "ttir.add"(%arg0, %1, %2) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    %4 = tensor.empty() : tensor<1x16x32xf32>
    %5 = "ttir.add"(%1, %3, %4) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<1x16x32xf32>, tensor<1x16x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
    return %5 : tensor<1x16x32xf32>
  }
}
